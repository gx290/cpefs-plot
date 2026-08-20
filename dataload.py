import argparse
import re
from pathlib import Path

import xarray as xr
import yaml


CONFIG_FILE = Path(__file__).with_name("config.yaml")


def get_source_name(variable_config: dict) -> str:
    """返回 NC 中的变量名；未配置 source 时直接使用 name。"""
    return variable_config.get("source", variable_config["name"])


def _validate_number_list(values, field_path: str) -> None:
    """校验色标分级值是严格递增的数字列表。"""
    if not isinstance(values, list) or not values:
        raise ValueError(f"{field_path} must be a non-empty list")
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in values):
        raise ValueError(f"{field_path} must contain only numbers")
    if any(current >= following for current, following in zip(values, values[1:])):
        raise ValueError(f"{field_path} must be strictly increasing")


def validate_config(config: dict) -> None:
    """在读取 NC 前校验 YAML 的必要字段和色标配置。"""
    if not isinstance(config, dict):
        raise ValueError("The YAML root must be a mapping")

    for section in ("source", "batch", "output", "state", "plot", "boundaries"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing or invalid config section: {section}")

    color_templates = config.get("color_templates")
    if not isinstance(color_templates, dict) or not color_templates:
        raise ValueError("color_templates must be a non-empty mapping")
    color_pattern = re.compile(r"#[0-9A-Fa-f]{6}")
    for template_name, template in color_templates.items():
        colors = template.get("colors") if isinstance(template, dict) else None
        if not isinstance(colors, list) or not colors:
            raise ValueError(f"color_templates.{template_name}.colors must be a non-empty list")
        if any(not isinstance(color, str) or not color_pattern.fullmatch(color) for color in colors):
            raise ValueError(
                f"color_templates.{template_name}.colors must contain #RRGGBB colors"
            )

    variables = config.get("variables")
    if not isinstance(variables, list):
        raise ValueError("variables must be a list")

    required_fields = (
        "name",
        "dimension",
        "product",
        "var_code",
        "display_name",
        "plot_unit",
        "color_template",
        "color_mode",
        "invalid_values",
    )
    variable_names = set()
    for index, item in enumerate(variables):
        path = f"variables[{index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{path} must be a mapping")
        missing_fields = [field for field in required_fields if field not in item]
        if missing_fields:
            raise ValueError(f"{path} is missing fields: {missing_fields}")

        variable_name = item["name"]
        if not isinstance(variable_name, str) or not variable_name.strip():
            raise ValueError(f"{path}.name must be a non-empty string")
        if variable_name in variable_names:
            raise ValueError(f"Duplicate variable name: {variable_name}")
        variable_names.add(variable_name)

        if str(item["dimension"]).upper() not in {"2D", "3D"}:
            raise ValueError(f"{path}.dimension must be 2D or 3D")
        if "source" in item and (
            not isinstance(item["source"], str) or not item["source"].strip()
        ):
            raise ValueError(f"{path}.source must be a non-empty string when provided")
        if not isinstance(item["invalid_values"], list):
            raise ValueError(f"{path}.invalid_values must be a list")

        template_name = item["color_template"]
        if template_name not in color_templates:
            raise ValueError(f"{path}.color_template references unknown template: {template_name}")
        colors = color_templates[template_name]["colors"]
        color_mode = str(item["color_mode"]).lower()
        if color_mode == "threshold":
            values = item.get("levels")
            value_path = f"{path}.levels"
            _validate_number_list(values, value_path)
            if len(colors) != len(values):
                raise ValueError(
                    f"{path} threshold mode requires one color for each level: "
                    f"colors={len(colors)}, levels={len(values)}"
                )
        elif color_mode == "interval":
            values = item.get("boundaries")
            value_path = f"{path}.boundaries"
            _validate_number_list(values, value_path)
            if len(colors) != len(values) + 1:
                raise ValueError(
                    f"{path} interval mode requires one more color than boundaries: "
                    f"colors={len(colors)}, boundaries={len(values)}"
                )
        else:
            raise ValueError(f"{path}.color_mode must be threshold or interval")


def load_config(config_path: Path = CONFIG_FILE) -> dict:
    """读取并校验 YAML 配置文件。"""
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    validate_config(config)
    return config


def open_dataset(data_file: str | Path) -> xr.Dataset:
    """打开一个 NC 文件，并让 xarray 自动解码填充值和缺测值。"""
    data_file = Path(data_file)
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    return xr.open_dataset(data_file, decode_cf=True, mask_and_scale=True)


def get_variable_configs(config: dict, dimension: str) -> list[dict]:
    """从统一的 variables 列表中筛选二维或三维要素配置。"""
    target_dimension = dimension.upper()
    if target_dimension not in {"2D", "3D"}:
        raise ValueError(f"Unsupported target dimension: {dimension}")

    result = []
    for item in config["variables"]:
        item_dimension = str(item.get("dimension", "2D")).upper()
        if item_dimension not in {"2D", "3D"}:
            raise ValueError(
                f"Variable {item.get('name', item.get('source', '<unknown>'))} has "
                f"unsupported dimension: {item.get('dimension')!r}"
            )
        if item_dimension == target_dimension:
            result.append(item)
    return result


def squeeze_to_grid(data_array: xr.DataArray, source_name: str) -> xr.DataArray:
    """去掉长度为1的维度，并确保变量最终是二维网格。"""
    result = data_array.squeeze(drop=True)
    if result.ndim != 2:
        raise ValueError(
            f"Variable {source_name} must become a 2-D grid after squeezing; "
            f"actual dimensions are {result.dims}"
        )
    return result


def squeeze_to_volume(data_array: xr.DataArray, source_name: str) -> xr.DataArray:
    """去掉长度为1的维度，并确保变量最终是气压层加二维网格的三维数据。"""
    result = data_array.squeeze(drop=True)
    if result.ndim != 3:
        raise ValueError(
            f"Variable {source_name} must become a 3-D pressure-level volume after "
            f"squeezing; actual dimensions are {result.dims}"
        )
    return result


def extract_data(ds: xr.Dataset, config: dict) -> xr.Dataset:
    """按照配置提取经纬度和绘图变量，并保留 NC 自带的单位属性。"""
    plot_config = config["plot"]
    latitude_name = plot_config["latitude"]
    longitude_name = plot_config["longitude"]
    variable_configs = get_variable_configs(config, "2D")

    required_names = [
        latitude_name,
        longitude_name,
        *(get_source_name(item) for item in variable_configs),
    ]
    missing_names = [name for name in required_names if name not in ds.variables]
    if missing_names:
        raise KeyError(f"Variables not found in dataset: {missing_names}")

    latitude = squeeze_to_grid(ds[latitude_name], latitude_name).copy()
    longitude = squeeze_to_grid(ds[longitude_name], longitude_name).copy()
    if latitude.dims != longitude.dims or latitude.shape != longitude.shape:
        raise ValueError("Latitude and longitude grids do not have matching dimensions")

    data_vars: dict[str, xr.DataArray] = {}
    for item in variable_configs:
        source_name = get_source_name(item)
        output_name = item["name"]
        data_array = squeeze_to_grid(ds[source_name], source_name).copy()
        if data_array.dims != latitude.dims:
            try:
                data_array = data_array.transpose(*latitude.dims)
            except ValueError as exc:
                raise ValueError(
                    f"Variable {source_name} does not match the latitude/longitude grid"
                ) from exc
        if data_array.shape != latitude.shape:
            raise ValueError(f"Variable {source_name} does not match the coordinate grid shape")

        data_array.attrs["source_variable"] = source_name
        data_vars[output_name] = data_array

    result = xr.Dataset(data_vars=data_vars)
    result = result.assign_coords(
        {
            latitude_name: latitude,
            longitude_name: longitude,
        }
    )
    return result


def extract_3d_data(ds: xr.Dataset, config: dict) -> xr.Dataset:
    """提取气压层坐标、经纬度和配置中的三维预报变量。"""
    variable_configs = get_variable_configs(config, "3D")
    if not variable_configs:
        return xr.Dataset()

    plot_config = config["plot"]
    latitude_name = plot_config["latitude"]
    longitude_name = plot_config["longitude"]
    level_name = plot_config["level"]
    required_names = [
        latitude_name,
        longitude_name,
        level_name,
        *(get_source_name(item) for item in variable_configs),
    ]
    missing_names = [name for name in required_names if name not in ds.variables]
    if missing_names:
        raise KeyError(f"3-D variables not found in dataset: {missing_names}")

    latitude = squeeze_to_grid(ds[latitude_name], latitude_name).copy()
    longitude = squeeze_to_grid(ds[longitude_name], longitude_name).copy()
    if latitude.dims != longitude.dims or latitude.shape != longitude.shape:
        raise ValueError("Latitude and longitude grids do not have matching dimensions")

    level = ds[level_name].squeeze(drop=True).copy()
    if level.ndim != 1:
        raise ValueError(
            f"Pressure level {level_name} must be one-dimensional; actual dimensions "
            f"are {level.dims}"
        )
    level_dimension = level.dims[0]
    expected_dimensions = (level_dimension, *latitude.dims)

    data_vars: dict[str, xr.DataArray] = {}
    for item in variable_configs:
        source_name = get_source_name(item)
        output_name = item["name"]
        data_array = squeeze_to_volume(ds[source_name], source_name).copy()
        try:
            data_array = data_array.transpose(*expected_dimensions)
        except ValueError as exc:
            raise ValueError(
                f"Variable {source_name} does not use pressure and horizontal grid "
                f"dimensions {expected_dimensions}"
            ) from exc
        if data_array.shape[1:] != latitude.shape:
            raise ValueError(f"Variable {source_name} does not match the coordinate grid shape")

        data_array.attrs["source_variable"] = source_name
        data_vars[output_name] = data_array

    result = xr.Dataset(data_vars=data_vars)
    return result.assign_coords(
        {
            level_name: level,
            latitude_name: latitude,
            longitude_name: longitude,
        }
    )


def read_data_from_file(data_file: str | Path, config: dict) -> xr.Dataset:
    """读取并加载单个 NC 文件中当前绘图需要的数据。"""
    with open_dataset(data_file) as ds:
        return extract_data(ds, config).load()


def read_product_data_from_file(
    data_file: str | Path,
    config: dict,
) -> tuple[xr.Dataset, xr.Dataset]:
    """一次打开 NC 文件，同时加载二维和三维绘图数据。"""
    with open_dataset(data_file) as ds:
        dataset_2d = extract_data(ds, config).load()
        dataset_3d = extract_3d_data(ds, config).load()
    return dataset_2d, dataset_3d


def read_pressure_levels(data_file: str | Path, config: dict) -> list[float]:
    """仅读取气压层坐标，供批处理提前判断全部三维图片是否存在。"""
    if not get_variable_configs(config, "3D"):
        return []

    level_name = config["plot"]["level"]
    with open_dataset(data_file) as ds:
        if level_name not in ds.variables:
            raise KeyError(f"Pressure level variable not found in dataset: {level_name}")
        level = ds[level_name].squeeze(drop=True)
        if level.ndim != 1:
            raise ValueError(f"Pressure level {level_name} must be one-dimensional")
        return [float(value) for value in level.values]


def read_data(config_path: Path = CONFIG_FILE) -> xr.Dataset:
    """兼容配置中直接指定 data_file 的单文件读取方式。"""
    config = load_config(config_path)
    if "data_file" not in config:
        raise KeyError(
            "The current config does not define data_file. "
            "Use read_data_from_file() or batch_product_generator.py."
        )
    return read_data_from_file(config["data_file"], config)


def save_data(ds: xr.Dataset, output_file: str | Path) -> Path:
    """将二维网格展开并保存为 CSV。"""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    dataframe = ds.to_dataframe().reset_index()
    dataframe.to_csv(output_file, index=False, encoding="utf-8-sig")
    return output_file


def main() -> None:
    """解析一个 NC 文件，并将配置变量保存为 CSV。"""
    parser = argparse.ArgumentParser(description="Parse one KeyMete NC file and save it as CSV.")
    parser.add_argument("data_file", help="Input KeyMete NC file path.")
    parser.add_argument("output_file", help="CSV output file path.")
    parser.add_argument("--config", type=Path, default=CONFIG_FILE, help="Path to config YAML file.")
    args = parser.parse_args()

    config = load_config(args.config)
    cloud_top_dataset = read_data_from_file(args.data_file, config)
    output_file = save_data(cloud_top_dataset, args.output_file)
    print(f"Saved parsed data to: {output_file}")


if __name__ == "__main__":
    main()

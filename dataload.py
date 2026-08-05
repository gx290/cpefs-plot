import argparse
import json
from pathlib import Path

import xarray as xr


CONFIG_FILE = Path(__file__).with_name("config.json")


def load_config(config_path: Path = CONFIG_FILE) -> dict:
    """读取 JSON 配置文件。"""
    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def open_dataset(data_file: str | Path) -> xr.Dataset:
    """打开一个 NC 文件，并让 xarray 自动解码填充值和缺测值。"""
    data_file = Path(data_file)
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    return xr.open_dataset(data_file, decode_cf=True, mask_and_scale=True)


def squeeze_to_grid(data_array: xr.DataArray, source_name: str) -> xr.DataArray:
    """去掉长度为1的维度，并确保变量最终是二维网格。"""
    result = data_array.squeeze(drop=True)
    if result.ndim != 2:
        raise ValueError(
            f"Variable {source_name} must become a 2-D grid after squeezing; "
            f"actual dimensions are {result.dims}"
        )
    return result


def extract_data(ds: xr.Dataset, config: dict) -> xr.Dataset:
    """按照配置提取经纬度和绘图变量，并保留 NC 自带的单位属性。"""
    plot_config = config["plot"]
    latitude_name = plot_config["latitude"]
    longitude_name = plot_config["longitude"]
    variable_configs = config["variables"]

    required_names = [
        latitude_name,
        longitude_name,
        *(item["source"] for item in variable_configs),
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
        source_name = item["source"]
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


def read_data_from_file(data_file: str | Path, config: dict) -> xr.Dataset:
    """读取并加载单个 NC 文件中当前绘图需要的数据。"""
    with open_dataset(data_file) as ds:
        return extract_data(ds, config).load()


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
    parser.add_argument("--config", type=Path, default=CONFIG_FILE, help="Path to config JSON file.")
    args = parser.parse_args()

    config = load_config(args.config)
    cloud_top_dataset = read_data_from_file(args.data_file, config)
    output_file = save_data(cloud_top_dataset, args.output_file)
    print(f"Saved parsed data to: {output_file}")


if __name__ == "__main__":
    main()

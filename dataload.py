import argparse
import json
from pathlib import Path

import xarray as xr


CONFIG_FILE = Path(__file__).with_name("config.json")


def load_config(config_path: Path = CONFIG_FILE) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def open_dataset(data_file: str | Path) -> xr.Dataset:
    data_file = Path(data_file)
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    return xr.open_dataset(data_file)


def extract_data(ds: xr.Dataset, config: dict) -> xr.Dataset:
    # 需要解析的变量全部从 config.json 中读取。
    variable_configs = config["variables"]
    source_names = [item["source"] for item in variable_configs]
    missing_vars = [name for name in source_names if name not in ds]
    if missing_vars:
        raise KeyError(f"Variables not found in dataset: {missing_vars}")

    data_vars = {}
    for item in variable_configs:
        source_name = item["source"]
        output_name = item["name"]
        # 复制原始变量，并按配置中的 name 作为输出字段名。
        data_array = ds[source_name].copy()
        data_array.attrs.update(
            {
                "name": output_name,
                "unit": item["unit"],
                "source_variable": source_name,
            }
        )
        data_vars[output_name] = data_array

    result = xr.Dataset(data_vars=data_vars)

    return result


def read_data_from_file(data_file: str | Path, config: dict) -> xr.Dataset:
    with open_dataset(data_file) as ds:
        return extract_data(ds, config).load()


def read_data(config_path: Path = CONFIG_FILE) -> xr.Dataset:
    """兼容旧配置中直接指定 data_file 的单文件读取方式。"""
    config = load_config(config_path)
    if "data_file" not in config:
        raise KeyError(
            "The current config does not define data_file. "
            "Use read_data_from_file() or the batch_plot_cloud_top.py command."
        )
    return read_data_from_file(config["data_file"], config)


def save_data(ds: xr.Dataset, output_file: str | Path) -> Path:
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    # 将网格数据展开成普通表格，方便直接用 Excel 查看。
    df = ds.to_dataframe().reset_index()
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    return output_file


def main() -> None:
    """解析一个指定的 nc 文件，并将格点数据保存为 CSV。"""
    parser = argparse.ArgumentParser(description="Parse one RY nc file and save its configured variables to CSV.")
    parser.add_argument("data_file", help="Input RY nc file path.")
    parser.add_argument("output_file", help="CSV output file path.")
    parser.add_argument("--config", type=Path, default=CONFIG_FILE, help="Path to config JSON file.")
    args = parser.parse_args()

    config = load_config(args.config)
    cloud_top_ds = read_data_from_file(args.data_file, config)
    output_file = save_data(cloud_top_ds, args.output_file)
    print(f"Saved parsed data to: {output_file}")


if __name__ == "__main__":
    main()

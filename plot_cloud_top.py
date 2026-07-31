"""Plot cloud-top products on the configured map projection."""

import argparse
import re
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader
import matplotlib

matplotlib.use("Agg") # 采用非交互式绘图后端，程序可以在没有图形桌面的服务器或后台任务中运行

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from dataload import load_config, read_data_from_file


MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
LATITUDE_NAME = "g3_lat_0"
LONGITUDE_NAME = "g3_lon_1"
TITLE_PREFIX = "RUC"
MAP_EXTENT = [107.5, 124.5, 35.0, 46.5]
GRID_EXTENT = [105.0, 127.0, 33.0, 48.0]
MAP_PROJECTION = ccrs.LambertConformal(
    central_longitude=105.0,
    central_latitude=40.0,
    standard_parallels=(30.0, 60.0),
)
LON_TICKS = np.arange(108.0, 124.1, 2.0)
LAT_TICKS = np.arange(36.0, 46.1, 2.0)

GRID_LON_LINES = [110.0, 120.0]
GRID_LAT_LINES = [40.0]

HEIGHT_LEVELS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16]
TEMPERATURE_LEVELS = [-70, -60, -50, -40, -30, -25, -20, -15, -10, -5, 0, 5, 10, 20]
HEIGHT_COLORS = [
    "#ffffff",
    "#D0CBF3",
    "#5E9CFB",
    "#71E8A6",
    "#06C80C",
    "#69EB38",
    "#EEF74A",
    "#E3C048",
    "#FCCE32",
    "#FF2828",
    "#FF6464",
    "#FFB4B4",
    "#F803C7",
    "#FF80FF",
    "#D0D0D0",
]
TEMPERATURE_COLORS = [
    # "#ffffff",
    # "#CC92D6",
    # "#CE54BC",
    # "#CEA09B",
    # "#D0614B",
    # "#CD3312",
    # "#DDB42E",
    # "#C29E2B",
    # "#DFCF42",
    # "#85C021",
    # "#4F9A00",
    # "#78C184",
    # "#50A5E1",
    # "#BCB9CC",
    # "#ffffff",
    "#ffffff",
    "#FF80FF",
    "#F803C7",
    "#FFB4B4",
    "#FF6464",
    "#FF2828",
    "#FCCE32",
    "#E3C048",
    "#EEF74A",
    "#69EB38",
    "#06C80C",
    "#71E8A6",
    "#5E9CFB",
    "#D0CBF3",
    "#ffffff"
]
MAP_FEATURE_SCALES = ["10m", "50m", "110m"]


# 从文件名中解析起报时间和预报时效，例如 RY_202607121200f001.grb.nc。
def parse_time_details_from_filename(data_file: str) -> tuple[datetime, int, str]:
    file_name = Path(data_file).name
    match = re.search(r"RY_(\d{14}|\d{12}|\d{10})f(\d{3})", file_name)
    if not match:
        raise ValueError(f"Cannot parse init time and forecast hour from file name: {file_name}")

    init_time_text = match.group(1)
    time_formats = {
        10: "%Y%m%d%H",
        12: "%Y%m%d%H%M",
        14: "%Y%m%d%H%M%S",
    }
    time_format = time_formats[len(init_time_text)]
    init_time = datetime.strptime(init_time_text, time_format)
    forecast_hour = int(match.group(2))
    return init_time, forecast_hour, time_format


def parse_time_from_filename(data_file: str) -> tuple[datetime, int]:
    init_time, forecast_hour, _ = parse_time_details_from_filename(data_file)
    return init_time, forecast_hour


# 根据原始单位和绘图单位自动转换数值。
def convert_values(values: np.ndarray, source_unit: str, target_unit: str) -> np.ndarray:
    source = source_unit.strip().lower()
    target = target_unit.strip().lower()
    celsius_units = {"°c", "℃", "c", "degc", "celsius"}

    if source == target:
        return values
    if source == "k" and target in celsius_units:
        return values - 273.15
    if source == "gpm" and target == "km":
        return values / 1000.0
    raise ValueError(f"Unsupported unit conversion: {source_unit} -> {target_unit}")


# 将无效值处理成 NaN，绘图时显示为白色。
def mask_invalid_values(values: np.ndarray, invalid_value: float) -> np.ndarray:
    result = values.astype(float)
    result[np.isclose(result, invalid_value)] = np.nan
    return result


# 生成离散色标：两端也有颜色档，但标签只显示 levels 里的边界数字。
def build_color_settings(levels: list[int | float], colors: list[str]):
    lower_extra = levels[0] - (levels[1] - levels[0])
    upper_extra = levels[-1] + (levels[-1] - levels[-2])
    boundaries = [lower_extra, *levels, upper_extra]
    cmap = ListedColormap(colors[: len(boundaries) - 1])
    cmap.set_bad("white")
    norm = BoundaryNorm(boundaries, cmap.N, clip=True)
    tick_positions = levels
    tick_labels = [str(value) for value in levels]
    return cmap, norm, tick_positions, tick_labels


# 加载 Natural Earth 地图图层。优先使用高精度，失败时自动降级。
@lru_cache(maxsize=None)
def load_natural_earth_feature(category: str, name: str, scale: str):
    """读取并缓存 Natural Earth 图层，供同一次批处理中的所有图片复用。"""
    shp_file = shapereader.natural_earth(
        resolution=scale,
        category=category,
        name=name,
    )
    geometries = tuple(shapereader.Reader(shp_file).geometries())
    return cfeature.ShapelyFeature(geometries, ccrs.PlateCarree())


def add_natural_earth_feature(ax, category: str, name: str, **style) -> None:
    last_error = None
    for scale in MAP_FEATURE_SCALES:
        try:
            feature = load_natural_earth_feature(category, name, scale)
            ax.add_feature(feature, **style)
            return
        except Exception as exc:
            last_error = exc

    print(f"Skip map feature {name}: {last_error}")


# 在线段中查找指定投影坐标的交点，并对另一坐标做线性插值。
def find_projected_intersection(
    fixed_values: np.ndarray,
    other_values: np.ndarray,
    target_value: float,
) -> float:
    for index in range(len(fixed_values) - 1):
        value1 = fixed_values[index]
        value2 = fixed_values[index + 1]
        if not np.isfinite(value1) or not np.isfinite(value2):
            continue
        if (value1 - target_value) * (value2 - target_value) > 0:
            continue

        ratio = 0.0 if np.isclose(value1, value2) else (target_value - value1) / (value2 - value1)
        return float(other_values[index] + ratio * (other_values[index + 1] - other_values[index]))

    raise ValueError(f"Cannot find projected intersection for target value: {target_value}")


# 根据底边经度交点和左边纬度交点，计算 Lambert 投影中的矩形裁剪范围。
def calculate_projected_map_bounds() -> tuple[float, float, float, float]:
    lon_min, lon_max, lat_min, lat_max = MAP_EXTENT

    # 左下角严格使用最小经度和最小纬度的投影点。
    x_left, y_bottom = MAP_PROJECTION.transform_point(
        lon_min,
        lat_min,
        ccrs.PlateCarree(),
    )

    # 计算最大经度经线与水平底边 y=y_bottom 的交点，确定右边界。
    lat_samples = np.linspace(GRID_EXTENT[2], GRID_EXTENT[3], 2000)
    lon_samples = np.full_like(lat_samples, lon_max)
    right_meridian = MAP_PROJECTION.transform_points(
        ccrs.PlateCarree(),
        lon_samples,
        lat_samples,
    )
    x_right = find_projected_intersection(
        right_meridian[:, 1],
        right_meridian[:, 0],
        y_bottom,
    )

    # 计算最大纬度纬线与垂直左边 x=x_left 的交点，确定上边界。
    lon_samples = np.linspace(GRID_EXTENT[0], GRID_EXTENT[1], 2000)
    lat_samples = np.full_like(lon_samples, lat_max)
    top_parallel = MAP_PROJECTION.transform_points(
        ccrs.PlateCarree(),
        lon_samples,
        lat_samples,
    )
    y_top = find_projected_intersection(
        top_parallel[:, 0],
        top_parallel[:, 1],
        x_left,
    )

    return x_left, x_right, y_bottom, y_top


# 计算经线和图框底边的交点，用于让经度刻度和真实经线对齐。
def get_bottom_tick_position(ax, lon: float) -> float | None:
    y_min = ax.get_ylim()[0]
    x_min, x_max = ax.get_xlim()
    lat_samples = np.linspace(GRID_EXTENT[2], GRID_EXTENT[3], 1000)
    lon_samples = np.full_like(lat_samples, lon)
    points = MAP_PROJECTION.transform_points(ccrs.PlateCarree(), lon_samples, lat_samples)
    x_values = points[:, 0]
    y_values = points[:, 1]

    for index in range(len(y_values) - 1):
        y1 = y_values[index]
        y2 = y_values[index + 1]
        if not np.isfinite(y1) or not np.isfinite(y2):
            continue
        if (y1 - y_min) * (y2 - y_min) > 0:
            continue

        ratio = 0.0 if np.isclose(y1, y2) else (y_min - y1) / (y2 - y1)
        x_cross = x_values[index] + ratio * (x_values[index + 1] - x_values[index])
        if x_min <= x_cross <= x_max:
            return (x_cross - x_min) / (x_max - x_min)

    # 靠近右侧的经线可能不穿过底边，但仍需要在底部显示刻度。
    x_value, _ = MAP_PROJECTION.transform_point(lon, MAP_EXTENT[2], ccrs.PlateCarree())
    position = (x_value - x_min) / (x_max - x_min)
    if 0.0 <= position <= 1.0:
        return position

    return None


# 计算纬线和图框左边的交点，用于让纬度刻度和真实纬线对齐。
def get_left_tick_position(ax, lat: float) -> float | None:
    x_min = ax.get_xlim()[0]
    y_min, y_max = ax.get_ylim()
    lon_samples = np.linspace(GRID_EXTENT[0], GRID_EXTENT[1], 1000)
    lat_samples = np.full_like(lon_samples, lat)
    points = MAP_PROJECTION.transform_points(ccrs.PlateCarree(), lon_samples, lat_samples)
    x_values = points[:, 0]
    y_values = points[:, 1]

    for index in range(len(x_values) - 1):
        x1 = x_values[index]
        x2 = x_values[index + 1]
        if not np.isfinite(x1) or not np.isfinite(x2):
            continue
        if (x1 - x_min) * (x2 - x_min) > 0:
            continue

        ratio = 0.0 if np.isclose(x1, x2) else (x_min - x1) / (x2 - x1)
        y_cross = y_values[index] + ratio * (y_values[index + 1] - y_values[index])
        if y_min <= y_cross <= y_max:
            return (y_cross - y_min) / (y_max - y_min)

    return None


# 手动绘制弧形经纬网和标签，避免 Cartopy Gridliner 在 Lambert 边界上的闭合错误。
def draw_manual_grid(ax) -> None:
    lon_line = np.linspace(GRID_EXTENT[0], GRID_EXTENT[1], 1000)
    lat_line = np.linspace(GRID_EXTENT[2], GRID_EXTENT[3], 1000)

    for lon in GRID_LON_LINES:
        ax.plot(
            np.full_like(lat_line, lon),
            lat_line,
            transform=ccrs.PlateCarree(),
            color="0.55",
            linewidth=0.45,
            linestyle=":",
            alpha=0.9,
            zorder=6,
        )

    for lon in LON_TICKS:
        position = get_bottom_tick_position(ax, lon)
        if position is None or not 0.0 <= position <= 1.0:
            continue
        ax.plot(
            [position] * 2,
            [0.0, -0.018],
            transform=ax.transAxes,
            color="0.25",
            linewidth=0.8,
            clip_on=False,
            zorder=7,
        )
        ax.text(
            position,
            -0.03,
            f"{int(lon)}°E",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9,
            clip_on=False,
        )

    for lat in GRID_LAT_LINES:
        ax.plot(
            lon_line,
            np.full_like(lon_line, lat),
            transform=ccrs.PlateCarree(),
            color="0.55",
            linewidth=0.45,
            linestyle=":",
            alpha=0.9,
            zorder=6,
        )
    for lat in LAT_TICKS:
        position = get_left_tick_position(ax, lat)
        if position is None or not 0.0 <= position <= 1.0:
            continue
        ax.plot(
            [0.0, -0.018],
            [position] * 2,
            transform=ax.transAxes,
            color="0.25",
            linewidth=0.8,
            clip_on=False,
            zorder=7,
        )
        ax.text(
            -0.035,
            position,
            f"{int(lat)}°N",
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=9,
            clip_on=False,
        )


# 设置投影地图范围、边界线和弧形经纬网。
def setup_map_axes(ax) -> None:
    x_left, x_right, y_bottom, y_top = calculate_projected_map_bounds()
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(y_bottom, y_top)
    add_natural_earth_feature(ax, "physical", "coastline", linewidth=0.8, edgecolor="0.2", facecolor="none", zorder=4)
    add_natural_earth_feature(ax, "cultural", "admin_0_boundary_lines_land", linewidth=0.7, edgecolor="black", facecolor="none", zorder=5)
    add_natural_earth_feature(ax, "cultural", "admin_1_states_provinces_lines", linewidth=0.55, edgecolor="0.25", facecolor="none", zorder=5)
    # 湖泊和河流图层暂不绘制，使底图只保留海岸线、国界和省界。
    # add_natural_earth_feature(ax, "physical", "lakes", linewidth=0.45, edgecolor="0.45", facecolor="none", zorder=4)
    # add_natural_earth_feature(ax, "physical", "rivers_lake_centerlines", linewidth=0.35, edgecolor="0.55", facecolor="none", zorder=4)

    draw_manual_grid(ax)

    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
        spine.set_edgecolor("0.2")


# 根据要素选择对应色标等级和颜色。
def get_color_config(variable_name: str) -> tuple[list[int | float], list[str]]:
    if variable_name == "cloudtoph":
        return HEIGHT_LEVELS, HEIGHT_COLORS
    if variable_name == "cloudtopt":
        return TEMPERATURE_LEVELS, TEMPERATURE_COLORS
    raise ValueError(f"No color levels configured for variable: {variable_name}")


# 格式化图像上的 UTC 和北京时间。
def format_time_text(init_time: datetime, forecast_hour: int) -> tuple[str, str]:
    valid_time = init_time + timedelta(hours=forecast_hour)
    valid_bst = valid_time + timedelta(hours=8)

    init_text = f"Init: {init_time:%H%M} UTC {init_time.day:02d} {MONTHS[init_time.month - 1]} {init_time.year}"
    valid_text = f"Valid: {valid_time:%H%M} UTC {valid_time.day:02d} {MONTHS[valid_time.month - 1]} {valid_time.year}"
    bst_text = f"( {valid_bst:%H%M} BST {valid_bst.day:02d} {MONTHS[valid_bst.month - 1]} {valid_bst.year} )"
    return valid_time.strftime("%Y%m%d%H%M"), "\n".join([init_text, valid_text, bst_text])


# 按命名规范生成 png 文件路径。
def build_image_path(
    config: dict,
    variable_config: dict,
    valid_time_text: str,
    init_time: datetime,
    forecast_hour: int,
    output_dir: Path,
) -> Path:
    filename_config = config["filename"]
    forecast_text = f"{forecast_hour:02d}00"
    file_name = (
        f"{filename_config['prefix']}_"
        f"{valid_time_text}_"
        f"{init_time:%Y%m%d%H}_"
        f"{forecast_text}_"
        f"{filename_config['height_code']}_"
        f"{variable_config['name']}_"
        f"{filename_config['region_code']}.png"
    )
    return output_dir / file_name


# 生成当前源文件、当前配置下应输出的全部图像路径。
def expected_image_paths(config: dict, source_file: str | Path, output_dir: Path) -> list[Path]:
    init_time, forecast_hour = parse_time_from_filename(str(source_file))
    valid_time_text, _ = format_time_text(init_time, forecast_hour)
    return [
        build_image_path(
            config,
            variable_config,
            valid_time_text,
            init_time,
            forecast_hour,
            output_dir,
        )
        for variable_config in config["variables"]
    ]


# 绘制单个变量的色斑图。
def plot_one_variable(
    ds,
    config: dict,
    variable_config: dict,
    init_time: datetime,
    forecast_hour: int,
    output_dir: Path,
) -> Path:
    name = variable_config["name"]
    lat = ds[LATITUDE_NAME].values
    lon = ds[LONGITUDE_NAME].values
    values = ds[name].values

    values = mask_invalid_values(values, variable_config["invalid_value"])
    values = convert_values(
        values,
        variable_config["unit"],
        variable_config["plot_unit"],
    )

    levels, colors = get_color_config(name)
    cmap, norm, tick_positions, tick_labels = build_color_settings(levels, colors)
    valid_time_text, time_text = format_time_text(init_time, forecast_hour)
    output_file = build_image_path(
        config,
        variable_config,
        valid_time_text,
        init_time,
        forecast_hour,
        output_dir,
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8, 8), dpi=150)
    ax = plt.axes(projection=MAP_PROJECTION)
    setup_map_axes(ax)

    mesh = ax.pcolormesh(
        lon,
        lat,
        values,
        cmap=cmap,
        norm=norm,
        shading="auto",
        transform=ccrs.PlateCarree(),
        zorder=3,
    )
    colorbar = plt.colorbar(
        mesh,
        ax=ax,
        orientation="vertical",
        pad=0.02,
        shrink=0.78,
        fraction=0.04,
        drawedges=True,
    )
    colorbar.set_ticks(tick_positions)
    colorbar.set_ticklabels(tick_labels)
    colorbar.outline.set_linewidth(0.8)
    colorbar.dividers.set_linewidth(0.5)

    fig.suptitle(
        f"{TITLE_PREFIX} {forecast_hour}h forecast: {variable_config['display_name']}",
        y=0.9,
        fontsize=20,
        fontweight="bold",
        fontfamily="Times New Roman",
    )
    ax.set_title(f"{variable_config['display_name']}(shaded,{variable_config['plot_unit']})", loc="left", fontsize=10, pad=1)
    ax.set_title(time_text, loc="right", fontsize=9, pad=1)

    plt.tight_layout(rect=[0, 0, 1, 0.955])
    plt.savefig(output_file, bbox_inches="tight")
    plt.close(fig)
    return output_file


# 绘制一个源 nc 文件对应的全部配置要素。
def plot_source_file(ds, config: dict, source_file: str | Path, output_dir: Path) -> list[Path]:
    init_time, forecast_hour = parse_time_from_filename(str(source_file))
    return [
        plot_one_variable(
            ds,
            config,
            variable_config,
            init_time,
            forecast_hour,
            output_dir,
        )
        for variable_config in config["variables"]
    ]


# 单文件调试入口：直接绘制指定 nc 文件，不保存 CSV。
def resolve_default_output_directory(config: dict, data_file: str | Path) -> Path:
    """按配置和源文件起报时间生成单文件绘图的默认输出目录。"""
    init_time, _ = parse_time_from_filename(str(data_file))
    tokens = {
        "YYYY": init_time.strftime("%Y"),
        "MM": init_time.strftime("%m"),
        "DD": init_time.strftime("%d"),
        "HH": init_time.strftime("%H"),
        "YYYYMM": init_time.strftime("%Y%m"),
        "YYYYMMDD": init_time.strftime("%Y%m%d"),
        "YYYYMMDDHH": init_time.strftime("%Y%m%d%H"),
        "YYYYMMDDHHMM": init_time.strftime("%Y%m%d%H%M"),
    }
    try:
        relative_dir = config["output"]["directory_template"].format(**tokens)
    except KeyError as exc:
        raise ValueError(f"Unsupported output time placeholder: {exc.args[0]}") from exc

    output_dir = Path(config["output"]["root_dir"]) / relative_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw cloud-top products for one nc file.")
    parser.add_argument("data_file", help="Input RY nc file path.")
    parser.add_argument(
        "--output-dir",
        help="PNG output directory. Defaults to the output template in config.json.",
    )
    args = parser.parse_args()

    config = load_config()
    output_dir = Path(args.output_dir) if args.output_dir else resolve_default_output_directory(config, args.data_file)
    parsed_ds = read_data_from_file(args.data_file, config)

    image_files = plot_source_file(parsed_ds, config, args.data_file, output_dir)

    for image_file in image_files:
        print(f"Saved image to: {image_file}")


if __name__ == "__main__":
    main()

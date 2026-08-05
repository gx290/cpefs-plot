"""绘制 KeyMete 配置要素的完整图和透明图。"""

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from shapely.geometry import box, shape
from shapely.ops import unary_union

from dataload import CONFIG_FILE, load_config, read_data_from_file


HEIGHT_LEVELS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16]
HEIGHT_COLORS = [
    "#b9afc6",
    "#3876d2",
    "#5ad073",
    "#458e39",
    "#66d800",
    "#f2f51b",
    "#c9a94e",
    "#edb800",
    "#ed2213",
    "#f35349",
    "#fbae9c",
    "#af4f6b",
    "#76287b",
    "#b667c7",
]

TEMPERATURE_LEVELS = [-70, -60, -50, -40, -30, -25, -20, -15, -10, -5, 0, 5, 10, 20]
TEMPERATURE_COLORS = [
    "#b667c7",
    "#76287b",
    "#af4f6b",
    "#fbae9c",
    "#f35349",
    "#ed2213",
    "#edb800",
    "#c9a94e",
    "#f2f51b",
    "#66d800",
    "#458e39",
    "#5ad073",
    "#3876d2",
    "#b9afc6",
]

PROJECT_DIR = Path(__file__).resolve().parent
COMPLETE_SIZE = (16.0, 14.5)
SIMPLE_SIZE = (14.1, 11.7)
IMAGE_DPI = 100
MAP_BACKGROUND = "#dcebf1"
FIGURE_BACKGROUND = "#eef4f7"
SIMPLE_ALPHA = 0.9
TIANJIN_URBAN_DISTRICTS = {
    "和平区",
    "河东区",
    "河西区",
    "南开区",
    "河北区",
    "红桥区",
}

@dataclass(frozen=True)
class ProductMetadata:
    """文件名中与绘图有关的产品时间信息。"""

    init_time: datetime
    total_forecast_hours: int
    forecast_hour: int
    forecast_text: str

    @property
    def valid_time(self) -> datetime:
        return self.init_time + timedelta(hours=self.forecast_hour)


def parse_product_filename(data_file: str | Path) -> ProductMetadata:
    """按固定字段位数解析顺序可变的产品文件名。"""
    file_name = Path(data_file).name
    if not file_name.endswith(".nc"):
        raise ValueError(
            f"Cannot parse product file name without .nc extension: {file_name}"
        )

    tokens = file_name[:-3].split("_")
    field_specs = {
        "init time": 10,
        "total forecast hours": 3,
        "forecast hour": 2,
    }
    parsed_tokens = {}
    for field_name, length in field_specs.items():
        candidates = [token for token in tokens if token.isdigit() and len(token) == length]
        if len(candidates) != 1:
            raise ValueError(
                f"Cannot parse {field_name} from {file_name}: expected exactly one "
                f"{length}-digit underscore-separated token, found {candidates}"
            )
        parsed_tokens[field_name] = candidates[0]

    init_time_text = parsed_tokens["init time"]
    total_hours = int(parsed_tokens["total forecast hours"])
    forecast_text = parsed_tokens["forecast hour"]
    forecast_hour = int(forecast_text)
    if forecast_hour > total_hours:
        raise ValueError(
            f"Forecast hour {forecast_hour} exceeds total forecast hours "
            f"{total_hours} in file name: {file_name}"
        )

    return ProductMetadata(
        init_time=datetime.strptime(init_time_text, "%Y%m%d%H"),
        total_forecast_hours=total_hours,
        forecast_hour=forecast_hour,
        forecast_text=forecast_text,
    )


def parse_time_details_from_filename(data_file: str) -> tuple[datetime, int, str]:
    """向批处理返回起报时间、当前时效和文件时间格式。"""
    metadata = parse_product_filename(data_file)
    return metadata.init_time, metadata.forecast_hour, "%Y%m%d%H"


def parse_time_from_filename(data_file: str) -> tuple[datetime, int]:
    """兼容原批处理调用方式。"""
    metadata = parse_product_filename(data_file)
    return metadata.init_time, metadata.forecast_hour


def normalize_unit(unit: str) -> str:
    """将常见的同义单位转换为统一标识。"""
    normalized = unit.strip().lower().replace(" ", "")
    aliases = {
        "°c": "c",
        "℃": "c",
        "degc": "c",
        "degree_celsius": "c",
        "degreescelsius": "c",
        "celsius": "c",
        "kelvin": "k",
        "geopotentialmeter": "gpm",
        "geopotentialmeters": "gpm",
        "metre": "m",
        "meter": "m",
        "kilometer": "km",
        "kilometre": "km",
    }
    return aliases.get(normalized, normalized)


def convert_values(values: np.ndarray, source_unit: str, target_unit: str) -> np.ndarray:
    """只有原单位和绘图单位不同时才执行单位转换。"""
    source = normalize_unit(source_unit)
    target = normalize_unit(target_unit)

    if source == target:
        return values
    if source == "k" and target == "c":
        return values - 273.15
    if source in {"gpm", "m"} and target == "km":
        return values / 1000.0
    if source == "km" and target in {"gpm", "m"}:
        return values * 1000.0
    raise ValueError(f"Unsupported unit conversion: {source_unit} -> {target_unit}")


def prepare_values(data_array, variable_config: dict, levels: list[int | float]) -> np.ma.MaskedArray:
    """处理 NaN、Inf、配置无效值和色标下限，并完成必要的单位转换。"""
    values = np.asarray(data_array.values, dtype=float)
    invalid_mask = ~np.isfinite(values)
    for invalid_value in variable_config.get("invalid_values", []):
        invalid_mask |= np.isclose(values, invalid_value)

    source_unit = data_array.attrs.get("units") or data_array.attrs.get("unit")
    if not source_unit:
        raise ValueError(f"Variable {variable_config['source']} does not define units")

    values = convert_values(values, source_unit, variable_config["plot_unit"])
    invalid_mask |= ~np.isfinite(values)
    invalid_mask |= values < levels[0]
    return np.ma.array(values, mask=invalid_mask)


def get_color_config(variable_name: str) -> tuple[list[int | float], list[str]]:
    """返回产品色标文档中规定的等级和颜色。"""
    if variable_name == "cloudtoph":
        return HEIGHT_LEVELS, HEIGHT_COLORS
    if variable_name == "cloudtopt":
        return TEMPERATURE_LEVELS, TEMPERATURE_COLORS
    raise ValueError(f"No color settings configured for variable: {variable_name}")


def build_data_color_settings(
    levels: list[int | float],
    colors: list[str],
) -> tuple[ListedColormap, BoundaryNorm]:
    """创建用于色斑分级的离散颜色映射。"""
    upper_boundary = levels[-1] + max(abs(levels[-1]), 1) * 10000
    boundaries = [*levels, upper_boundary]
    cmap = ListedColormap(colors)
    cmap.set_bad((0, 0, 0, 0))
    norm = BoundaryNorm(boundaries, cmap.N, clip=True)
    return cmap, norm


def get_chinese_font() -> str:
    """选择当前系统中可用的中文字体。"""
    installed_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
        if font_name in installed_fonts:
            return font_name
    return "DejaVu Sans"


CHINESE_FONT = get_chinese_font()


def resolve_boundary_path(file_name: str) -> Path:
    """将边界文件路径解析为绝对路径，相对路径以程序目录为基准。"""
    path = Path(file_name)
    if not path.is_absolute():
        path = PROJECT_DIR / path
    return path.resolve()


@lru_cache(maxsize=None)
def load_geojson_records(file_name: str) -> tuple[tuple[dict, object], ...]:
    """读取并缓存 GeoJSON 中具有行政区面的属性和几何对象。"""
    boundary_file = resolve_boundary_path(file_name)
    if not boundary_file.is_file():
        raise FileNotFoundError(f"Boundary file does not exist: {boundary_file}")

    with boundary_file.open("r", encoding="utf-8") as file:
        geojson = json.load(file)

    records = []
    for feature in geojson.get("features", []):
        geometry_data = feature.get("geometry")
        if geometry_data is None:
            continue
        geometry = shape(geometry_data)
        # 省、市文件还包含“境界线”，这里只使用行政区面，避免重复描边。
        if geometry.is_empty or geometry.geom_type not in {"Polygon", "MultiPolygon"}:
            continue
        records.append((dict(feature.get("properties") or {}), geometry))
    return tuple(records)


@lru_cache(maxsize=None)
def load_country_outline(file_name: str):
    """合并全部省级面，得到不包含省内边界的中国外轮廓。"""
    return unary_union(
        [geometry for _, geometry in load_geojson_records(file_name)]
    )


def records_in_extent(
    records: tuple[tuple[dict, object], ...],
    extent: list[float],
    predicate=None,
) -> list[tuple[dict, object]]:
    """筛选与当前地图范围相交并满足属性条件的边界记录。"""
    view_box = box(extent[0], extent[2], extent[1], extent[3])
    return [
        (attributes, geometry)
        for attributes, geometry in records
        if (predicate is None or predicate(attributes)) and geometry.intersects(view_box)
    ]


def add_geojson_layer(
    ax,
    file_name: str,
    extent: list[float],
    predicate=None,
    **style,
) -> None:
    """把筛选后的本地 GeoJSON 行政区面叠加到地图上。"""
    records = records_in_extent(
        load_geojson_records(file_name),
        extent,
        predicate,
    )
    if records:
        ax.add_geometries(
            [geometry for _, geometry in records],
            ccrs.PlateCarree(),
            **style,
        )

def draw_grid_and_inner_labels(ax, extent: list[float]) -> None:
    """每隔2度绘制直角经纬线，并把标注放在图框内部。"""
    lon_min, lon_max, lat_min, lat_max = extent
    longitude_ticks = np.arange(np.ceil(lon_min / 2) * 2, lon_max + 0.001, 2)
    latitude_ticks = np.arange(np.ceil(lat_min / 2) * 2, lat_max + 0.001, 2)

    for longitude in longitude_ticks:
        ax.axvline(longitude, color="#77858b", linewidth=0.55, alpha=0.75, zorder=3)
        ax.text(
            longitude + 0.07,
            lat_min + 0.10,
            f"{int(longitude)}°E",
            fontsize=8,
            color="#566168",
            ha="left",
            va="bottom",
            fontfamily=CHINESE_FONT,
            zorder=8,
            clip_on=True,
        )

    for latitude in latitude_ticks:
        ax.axhline(latitude, color="#77858b", linewidth=0.55, alpha=0.75, zorder=3)
        ax.text(
            lon_min + 0.07,
            latitude - 0.08,
            f"{int(latitude)}°N",
            fontsize=8,
            color="#566168",
            ha="left",
            va="top",
            fontfamily=CHINESE_FONT,
            zorder=8,
            clip_on=True,
        )


def add_map_boundaries(ax, config: dict, extent: list[float]) -> None:
    """使用天地图 GeoJSON 绘制国界、省界、市界和天津区界。"""
    boundary_config = config["boundaries"]
    ax.add_geometries(
        [load_country_outline(boundary_config["province_file"])],
        ccrs.PlateCarree(),
        linewidth=boundary_config["country_linewidth"],
        edgecolor="#15191b",
        facecolor="none",
        zorder=6,
    )
    add_geojson_layer(
        ax,
        boundary_config["province_file"],
        extent,
        linewidth=boundary_config["province_linewidth"],
        edgecolor="#15191b",
        facecolor="none",
        zorder=6.1,
    )
    add_geojson_layer(
        ax,
        boundary_config["city_file"],
        extent,
        predicate=lambda attributes: str(attributes.get("gb")) != "156120000",
        linewidth=boundary_config["city_linewidth"],
        edgecolor="#4e5559",
        facecolor="none",
        zorder=6.2,
    )
    add_geojson_layer(
        ax,
        boundary_config["tianjin_district_file"],
        extent,
        linewidth=boundary_config["district_linewidth"],
        edgecolor="#4e5559",
        facecolor="none",
        zorder=6.3,
    )


def add_china_province_labels(ax, config: dict, extent: list[float]) -> None:
    """根据省级 GeoJSON 的 name 字段标注省级行政区名称。"""
    view_box = box(extent[0], extent[2], extent[1], extent[3])
    offsets = {
        "北京市": (-0.35, 0.30),
        "天津市": (0.25, -0.20),
        "河北省": (-0.15, -0.15),
    }
    for attributes, geometry in load_geojson_records(
        config["boundaries"]["province_file"]
    ):
        name = str(attributes.get("name") or "").strip()
        if not name or name == "天津市":
            continue
        visible_geometry = geometry.intersection(view_box)
        if visible_geometry.is_empty:
            continue

        label_point = visible_geometry.representative_point()
        offset_lon, offset_lat = offsets.get(name, (0.0, 0.0))
        text = ax.text(
            label_point.x + offset_lon,
            label_point.y + offset_lat,
            name,
            transform=ccrs.PlateCarree(),
            fontsize=15,
            fontweight="bold",
            fontfamily=CHINESE_FONT,
            color="#52585c",
            ha="center",
            va="center",
            zorder=8,
            clip_on=True,
        )
        text.set_path_effects(
            [
                path_effects.Stroke(linewidth=4.0, foreground="white", alpha=0.9),
                path_effects.Normal(),
            ]
        )


def add_tianjin_district_labels(ax, config: dict, extent: list[float]) -> None:
    """标注天津外围各区，并将中心六区合并标注为天津市区。"""
    view_box = box(extent[0], extent[2], extent[1], extent[3])
    boundary_config = config["boundaries"]
    labels = []
    urban_geometries = []
    for attributes, geometry in load_geojson_records(
        boundary_config["tianjin_district_file"],
    ):
        visible_geometry = geometry.intersection(view_box)
        if visible_geometry.is_empty:
            continue

        name = str(attributes.get("name") or "").strip()
        if not name:
            continue
        if name in TIANJIN_URBAN_DISTRICTS:
            urban_geometries.append(visible_geometry)
        else:
            labels.append((name, visible_geometry))

    if urban_geometries:
        labels.append(("天津市区", unary_union(urban_geometries)))

    for name, geometry in labels:
        label_point = geometry.representative_point()
        text = ax.text(
            label_point.x,
            label_point.y,
            name,
            transform=ccrs.PlateCarree(),
            fontsize=9 if name == "天津市区" else 6.5,
            fontfamily=CHINESE_FONT,
            color="#30373b",
            ha="center",
            va="center",
            zorder=8.2,
            clip_on=True,
        )
        text.set_path_effects(
            [
                path_effects.Stroke(linewidth=2.0, foreground="white", alpha=0.95),
                path_effects.Normal(),
            ]
        )


def draw_discrete_colorbar(
    fig,
    variable_config: dict,
    levels: list[int | float],
    colors: list[str],
) -> None:
    """在完整图左下方绘制等高色块式图例。"""
    colorbar_axis = fig.add_axes([0.026, 0.115, 0.023, 0.365])
    category_boundaries = np.arange(len(colors) + 1)
    category_centers = np.arange(len(colors)) + 0.5
    colorbar = ColorbarBase(
        colorbar_axis,
        cmap=ListedColormap(colors),
        norm=Normalize(0, len(colors)),
        boundaries=category_boundaries,
        ticks=category_centers,
        spacing="uniform",
        orientation="vertical",
        drawedges=True,
    )
    colorbar.ax.set_yticklabels([str(value) for value in levels])
    colorbar.ax.tick_params(length=0, labelsize=10, pad=8)
    colorbar.outline.set_linewidth(0.8)
    colorbar.dividers.set_linewidth(0.5)

    fig.text(
        0.026,
        0.493,
        f"{variable_config['name'].upper()}\n({variable_config['plot_unit']})",
        fontsize=13,
        fontweight="bold",
        fontfamily=CHINESE_FONT,
        color="#30363a",
        ha="left",
        va="bottom",
    )


def build_image_name(
    config: dict,
    metadata: ProductMetadata,
    variable_name: str,
    complete: bool,
) -> str:
    """按照可配置前缀生成二维产品文件名。"""
    prefix = config["output"].get("filename_prefix", "").strip("_")
    parts = [
        part
        for part in (
            prefix,
            metadata.init_time.strftime("%Y%m%d%H"),
            f"F{metadata.forecast_text}",
            variable_name,
            "2D",
        )
        if part
    ]
    if complete:
        parts.append("COMPLETE")
    return "_".join(parts) + ".png"


def build_image_path(
    config: dict,
    output_dir: Path,
    metadata: ProductMetadata,
    variable_name: str,
    complete: bool,
) -> Path:
    """返回 complete 或 simple 子目录中的图像路径。"""
    variant_directory = "complete" if complete else "simple"
    return (
        output_dir
        / "2d"
        / variant_directory
        / build_image_name(config, metadata, variable_name, complete)
    )


def expected_image_paths(config: dict, source_file: str | Path, output_dir: Path) -> list[Path]:
    """返回一个 NC 文件按当前配置应该生成的全部 PNG。"""
    metadata = parse_product_filename(source_file)
    paths = []
    for variable_config in config["variables"]:
        for complete in (True, False):
            paths.append(
                build_image_path(
                    config,
                    output_dir,
                    metadata,
                    variable_config["name"],
                    complete,
                )
            )
    return paths


def draw_complete_image(
    latitude: np.ndarray,
    longitude: np.ndarray,
    values: np.ma.MaskedArray,
    cmap: ListedColormap,
    norm: BoundaryNorm,
    levels: list[int | float],
    colors: list[str],
    variable_config: dict,
    metadata: ProductMetadata,
    config: dict,
    output_file: Path,
) -> None:
    """绘制包含地图、中文行政区名称、标题和色标的完整产品。"""
    extent = config["plot"]["extent"]
    figure = plt.figure(figsize=COMPLETE_SIZE, dpi=IMAGE_DPI, facecolor=FIGURE_BACKGROUND)
    axis = figure.add_axes([0.09, 0.087, 0.881, 0.807], projection=ccrs.PlateCarree())
    axis.set_extent(extent, crs=ccrs.PlateCarree())
    axis.set_aspect("auto")
    axis.set_facecolor(MAP_BACKGROUND)
    axis.set_xticks([])
    axis.set_yticks([])

    axis.pcolormesh(
        longitude,
        latitude,
        values,
        cmap=cmap,
        norm=norm,
        shading="auto",
        transform=ccrs.PlateCarree(),
        antialiased=False,
        zorder=2,
    )
    draw_grid_and_inner_labels(axis, extent)
    add_map_boundaries(axis, config, extent)
    add_china_province_labels(axis, config, extent)
    add_tianjin_district_labels(axis, config, extent)

    for spine in axis.spines.values():
        spine.set_linewidth(3.0)
        spine.set_edgecolor("#111516")

    title = (
        f"{config['plot']['title_prefix']}  {variable_config['display_name']}  "
        f"{metadata.valid_time:%Y年%m月%d日%H时}"
    )
    figure.text(
        0.055,
        0.965,
        title,
        fontsize=26,
        fontweight="bold",
        fontfamily=CHINESE_FONT,
        color="#202427",
        ha="left",
        va="top",
    )
    figure.text(
        0.86,
        0.965,
        "(UTC)",
        fontsize=14,
        fontfamily=CHINESE_FONT,
        color="#30363a",
        ha="center",
        va="top",
    )
    figure.text(
        0.875,
        0.94,
        f"{metadata.init_time:%Y%m%d%H}  F{metadata.forecast_hour}",
        fontsize=14,
        fontfamily=CHINESE_FONT,
        color="#30363a",
        ha="center",
        va="top",
    )
    figure.text(
        0.022,
        0.045,
        config["plot"].get("footer", ""),
        fontsize=17,
        fontweight="bold",
        fontfamily=CHINESE_FONT,
        color="#30363a",
        ha="left",
        va="bottom",
    )
    draw_discrete_colorbar(figure, variable_config, levels, colors)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_file, dpi=IMAGE_DPI, facecolor=FIGURE_BACKGROUND)
    plt.close(figure)


def draw_simple_image(
    latitude: np.ndarray,
    longitude: np.ndarray,
    values: np.ma.MaskedArray,
    cmap: ListedColormap,
    norm: BoundaryNorm,
    extent: list[float],
    output_file: Path,
) -> None:
    """绘制仅包含色块、背景完全透明的 GIS 叠加产品。"""
    figure = plt.figure(figsize=SIMPLE_SIZE, dpi=IMAGE_DPI)
    figure.patch.set_alpha(0)
    axis = figure.add_axes([0, 0, 1, 1])
    axis.patch.set_alpha(0)
    axis.set_xlim(extent[0], extent[1])
    axis.set_ylim(extent[2], extent[3])
    axis.set_aspect("auto")
    axis.set_axis_off()
    axis.pcolormesh(
        longitude,
        latitude,
        values,
        cmap=cmap,
        norm=norm,
        shading="auto",
        antialiased=False,
        alpha=SIMPLE_ALPHA,
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_file, dpi=IMAGE_DPI, transparent=True, pad_inches=0)
    plt.close(figure)


def plot_one_variable(
    dataset,
    config: dict,
    variable_config: dict,
    metadata: ProductMetadata,
    output_dir: Path,
) -> list[Path]:
    """为一个要素分别绘制完整图和透明图。"""
    latitude_name = config["plot"]["latitude"]
    longitude_name = config["plot"]["longitude"]
    variable_name = variable_config["name"]

    latitude = dataset[latitude_name].values
    longitude = dataset[longitude_name].values
    levels, colors = get_color_config(variable_name)
    values = prepare_values(dataset[variable_name], variable_config, levels)
    cmap, norm = build_data_color_settings(levels, colors)

    complete_file = build_image_path(
        config,
        output_dir,
        metadata,
        variable_name,
        complete=True,
    )
    simple_file = build_image_path(
        config,
        output_dir,
        metadata,
        variable_name,
        complete=False,
    )
    draw_complete_image(
        latitude,
        longitude,
        values,
        cmap,
        norm,
        levels,
        colors,
        variable_config,
        metadata,
        config,
        complete_file,
    )
    draw_simple_image(
        latitude,
        longitude,
        values,
        cmap,
        norm,
        config["plot"]["extent"],
        simple_file,
    )
    return [complete_file, simple_file]


def plot_source_file(
    dataset,
    config: dict,
    source_file: str | Path,
    output_dir: Path,
) -> list[Path]:
    """绘制一个源文件中配置的全部二维要素。"""
    metadata = parse_product_filename(source_file)
    image_files = []
    for variable_config in config["variables"]:
        image_files.extend(
            plot_one_variable(
                dataset,
                config,
                variable_config,
                metadata,
                output_dir,
            )
        )
    return image_files


def resolve_default_output_directory(config: dict, data_file: str | Path) -> Path:
    """生成单文件调试模式使用的预报时效根目录。"""
    metadata = parse_product_filename(data_file)
    output_config = config["output"]
    output_dir = (
        Path(output_config["root_dir"])
        / output_config["model_directory"]
        / metadata.init_time.strftime("%Y%m%d%H")
        / f"F{metadata.forecast_text}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main() -> None:
    """单独绘制一个 KeyMete NC 文件。"""
    parser = argparse.ArgumentParser(description="Draw configured products for one KeyMete NC file.")
    parser.add_argument("data_file", help="Input KeyMete NC file path.")
    parser.add_argument(
        "--output-dir",
        help="Forecast-hour output root; defaults to the configured directory structure.",
    )
    parser.add_argument("--config", type=Path, default=CONFIG_FILE, help="Path to config JSON file.")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else resolve_default_output_directory(config, args.data_file)
    )
    parsed_dataset = read_data_from_file(args.data_file, config)
    image_files = plot_source_file(parsed_dataset, config, args.data_file, output_dir)
    for image_file in image_files:
        print(f"Saved image to: {image_file}")


if __name__ == "__main__":
    main()

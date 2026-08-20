"""绘制 KeyMete 配置要素的完整图和透明图。"""

import argparse
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from shapely.geometry import box

from dataload import (
    CONFIG_FILE,
    get_variable_configs,
    load_config,
    read_pressure_levels,
    read_product_data_from_file,
)


PROJECT_DIR = Path(__file__).resolve().parent
COMPLETE_SIZE = (14.1, 11.7)
SIMPLE_SIZE = (14.1, 11.7)
IMAGE_DPI = 100
MAP_BACKGROUND = "#ffffff"
FIGURE_BACKGROUND = "#ffffff"
SIMPLE_ALPHA = 0.9

@dataclass(frozen=True)
class ProductMetadata:
    """文件名中与绘图有关的产品时间信息。"""

    region_code: str
    init_time: datetime
    total_forecast_hours: int
    forecast_hour: int
    forecast_text: str

    @property
    def valid_time(self) -> datetime:
        return self.init_time + timedelta(hours=self.forecast_hour)


def parse_product_filename(data_file: str | Path) -> ProductMetadata:
    """解析文件名第6段区域代码，以及产品时间和预报时效。"""
    file_name = Path(data_file).name
    if not file_name.endswith(".nc"):
        raise ValueError(
            f"Cannot parse product file name without .nc extension: {file_name}"
        )

    tokens = file_name[:-3].split("_")
    if len(tokens) < 6:
        raise ValueError(
            f"Cannot parse REGION from {file_name}: expected at least 6 "
            f"underscore-separated fields, found {len(tokens)}"
        )
    region_code = tokens[5].strip().upper()
    if not re.fullmatch(r"[A-Z0-9]+", region_code):
        raise ValueError(
            f"Invalid REGION in field 6 of {file_name}: {tokens[5]!r}"
        )

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
        region_code=region_code,
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
        "kgkg-1": "kg/kg",
        "kgkg^-1": "kg/kg",
        "gkg-1": "g/kg",
        "gkg^-1": "g/kg",
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
    if source == "kg/kg" and target == "g/kg":
        return values * 1000.0
    if source == "g/kg" and target == "kg/kg":
        return values / 1000.0
    raise ValueError(f"Unsupported unit conversion: {source_unit} -> {target_unit}")


def prepare_values(
    data_array,
    variable_config: dict,
    levels: list[int | float],
    mask_below_first_level: bool = True,
) -> np.ma.MaskedArray:
    """处理 NaN、Inf、配置无效值和色标下限，并完成必要的单位转换。"""
    values = np.asarray(data_array.values, dtype=float)
    invalid_mask = ~np.isfinite(values)
    for invalid_value in variable_config.get("invalid_values", []):
        invalid_mask |= np.isclose(values, invalid_value)

    source_unit = data_array.attrs.get("units") or data_array.attrs.get("unit")
    if not source_unit:
        source_name = variable_config.get("source", variable_config["name"])
        raise ValueError(f"Variable {source_name} does not define units")

    values = convert_values(values, source_unit, variable_config["plot_unit"])
    invalid_mask |= ~np.isfinite(values)
    if mask_below_first_level:
        invalid_mask |= values < levels[0]
    return np.ma.array(values, mask=invalid_mask)


def get_color_config(
    config: dict,
    variable_config: dict,
) -> tuple[list[int | float], list[str], list[str], bool, str]:
    """从 YAML 返回分级值、颜色、标签、低值掩膜方式和色标模式。"""
    template_name = variable_config["color_template"]
    colors = list(config["color_templates"][template_name]["colors"])
    color_mode = str(variable_config["color_mode"]).lower()
    if color_mode == "threshold":
        values = list(variable_config["levels"])
        mask_below_first_level = True
    else:
        values = list(variable_config["boundaries"])
        mask_below_first_level = False
    labels = [str(value) for value in values]
    return values, colors, labels, mask_below_first_level, color_mode


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


def build_interval_color_settings(
    boundaries: list[int | float],
    colors: list[str],
) -> tuple[ListedColormap, BoundaryNorm]:
    """为温度这种包含低于首界限和高于末界限的区间色标创建映射。"""
    outer_limit = 1e10
    data_boundaries = list(boundaries)
    # 文档首档为“≤-50”，将首边界向上移动一个浮点步长以包含恰好 -50。
    data_boundaries[0] = np.nextafter(data_boundaries[0], np.inf)
    cmap = ListedColormap(colors)
    cmap.set_bad((0, 0, 0, 0))
    norm = BoundaryNorm([-outer_limit, *data_boundaries, outer_limit], cmap.N, clip=True)
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
def load_shapefile_records(file_name: str) -> tuple[tuple[dict, object], ...]:
    """读取并缓存 Shapefile 中的行政区属性和几何对象。"""
    boundary_file = resolve_boundary_path(file_name)
    if not boundary_file.is_file():
        raise FileNotFoundError(f"Boundary file does not exist: {boundary_file}")

    records = []
    for record in shpreader.Reader(boundary_file).records():
        geometry = record.geometry
        if geometry is None or geometry.is_empty:
            continue
        records.append((dict(record.attributes), geometry))
    return tuple(records)


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


def add_shapefile_layer(
    ax,
    file_name: str,
    extent: list[float],
    predicate=None,
    **style,
) -> None:
    """把筛选后的本地 Shapefile 行政区几何叠加到地图上。"""
    records = records_in_extent(
        load_shapefile_records(file_name),
        extent,
        predicate,
    )
    if records:
        ax.add_geometries(
            [geometry for _, geometry in records],
            ccrs.PlateCarree(),
            **style,
        )


def add_map_boundaries(ax, config: dict, extent: list[float]) -> None:
    """使用 GADM Shapefile 绘制国界、省界、市界和直辖市区界。"""
    boundary_config = config["boundaries"]
    add_shapefile_layer(
        ax,
        boundary_config["city_file"],
        extent,
        linewidth=boundary_config["city_linewidth"],
        edgecolor="#c3c3c3",
        facecolor="none",
        zorder=5.8,
    )
    add_shapefile_layer(
        ax,
        boundary_config["district_file"],
        extent,
        predicate=lambda attributes: attributes.get("NAME_1") in {"Beijing", "Tianjin"},
        linewidth=boundary_config["district_linewidth"],
        edgecolor="#b8b8b8",
        facecolor="none",
        zorder=5.9,
    )
    add_shapefile_layer(
        ax,
        boundary_config["province_file"],
        extent,
        linewidth=boundary_config["province_linewidth"],
        edgecolor="#111111",
        facecolor="none",
        zorder=6.2,
    )
    add_shapefile_layer(
        ax,
        boundary_config["country_file"],
        extent,
        linewidth=boundary_config["country_linewidth"],
        edgecolor="#111111",
        facecolor="none",
        zorder=6.4,
    )


def get_chinese_province_name(attributes: dict) -> str:
    """从 GADM 属性中提取简体中文省级名称并补充行政区后缀。"""
    native_name = str(attributes.get("NL_NAME_1") or "").strip()
    candidates = [item.strip() for item in native_name.split("|") if item.strip()]
    if not candidates:
        return ""

    name = candidates[-1]
    english_name = str(attributes.get("NAME_1") or "")
    if english_name in {"Beijing", "Tianjin", "Shanghai", "Chongqing"}:
        return name if name.endswith("市") else f"{name}市"
    if name.endswith(("省", "自治区", "特别行政区")):
        return name
    return f"{name}省"


def add_china_province_labels(ax, config: dict, extent: list[float]) -> None:
    """根据 GADM 的中文属性标注省、自治区和直辖市名称。"""
    view_box = box(extent[0], extent[2], extent[1], extent[3])
    offsets = {
        "北京市": (-0.20, 0.18),
        "天津市": (0.28, -0.10),
        "河北省": (-0.18, -0.08),
    }
    for attributes, geometry in load_shapefile_records(
        config["boundaries"]["province_file"]
    ):
        name = get_chinese_province_name(attributes)
        if not name:
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
            fontsize=12,
            fontweight="bold",
            fontfamily=CHINESE_FONT,
            color="#7c7c7c",
            ha="center",
            va="center",
            zorder=8,
            clip_on=True,
        )
        text.set_path_effects(
            [
                path_effects.Stroke(linewidth=2.5, foreground="white", alpha=0.92),
                path_effects.Normal(),
            ]
        )


def draw_discrete_colorbar(
    fig,
    variable_config: dict,
    labels: list[str],
    colors: list[str],
) -> None:
    """按样例图4在地图左下角绘制紧凑的离散色标。"""
    legend_background = fig.add_axes([0.063, 0.047, 0.074, 0.235], zorder=9)
    legend_background.set_facecolor("white")
    legend_background.set_xticks([])
    legend_background.set_yticks([])
    for spine in legend_background.spines.values():
        spine.set_color("#aaaaaa")
        spine.set_linewidth(0.8)

    colorbar_axis = fig.add_axes([0.069, 0.055, 0.018, 0.205])
    colorbar_axis.set_zorder(10)
    category_boundaries = np.arange(len(colors) + 1)
    if len(labels) == len(colors):
        tick_positions = np.arange(len(colors)) + 0.5
    elif len(labels) == len(colors) - 1:
        # 温度色标只在相邻颜色的分界处显示温度值，不显示区间文字。
        tick_positions = np.arange(1, len(colors))
    else:
        raise ValueError(
            "Colorbar labels must match either all color blocks or their boundaries"
        )
    colorbar = ColorbarBase(
        colorbar_axis,
        cmap=ListedColormap(colors),
        norm=Normalize(0, len(colors)),
        boundaries=category_boundaries,
        ticks=tick_positions,
        spacing="uniform",
        orientation="vertical",
        drawedges=True,
    )
    colorbar.ax.set_yticklabels(labels)
    colorbar.ax.tick_params(length=0, labelsize=8, pad=4)
    colorbar.outline.set_linewidth(0.8)
    colorbar.dividers.set_linewidth(0.5)

    fig.text(
        0.069,
        0.267,
        f"({variable_config['plot_unit']})",
        fontsize=9,
        fontfamily=CHINESE_FONT,
        color="#222222",
        ha="left",
        va="bottom",
        zorder=11,
    )


def normalize_code(value: str, field_name: str, allow_underscore: bool = False) -> str:
    """校验并标准化 schema_v3 文件名中的代码字段。"""
    code = str(value).strip().upper()
    pattern = r"[A-Z0-9_]+" if allow_underscore else r"[A-Z0-9]+"
    if not re.fullmatch(pattern, code):
        raise ValueError(f"Invalid {field_name} code for schema_v3: {value!r}")
    return code


def get_publish_codes(
    config: dict,
    metadata: ProductMetadata,
    variable_config: dict,
) -> tuple[str, str, str, str]:
    """读取模式、区域、产品和要素的标准发布代码。"""
    output_config = config["output"]
    model = normalize_code(output_config["model_code"], "MODEL")
    region = normalize_code(metadata.region_code, "REGION")
    product = normalize_code(variable_config["product"], "PRODUCT")
    var_code = normalize_code(variable_config["var_code"], "VAR", allow_underscore=True)
    return model, region, product, var_code


def format_beijing_init(metadata: ProductMetadata) -> str:
    """将源文件中的 UTC 起报时间转换为发布使用的北京时间。"""
    return (metadata.init_time + timedelta(hours=8)).strftime("%Y%m%d%H")


def build_image_name(
    config: dict,
    metadata: ProductMetadata,
    variable_config: dict,
    complete: bool,
) -> str:
    """按照 schema_v3 生成二维产品文件名。"""
    model, region, product, var_code = get_publish_codes(config, metadata, variable_config)
    variant = "COMPLETE" if complete else "SIMPLE"
    parts = [
        model,
        region,
        format_beijing_init(metadata),
        f"{metadata.forecast_hour:03d}",
        product,
        variant,
        var_code,
    ]
    return "_".join(parts) + ".png"


def build_image_path(
    config: dict,
    output_dir: Path,
    metadata: ProductMetadata,
    variable_config: dict,
    complete: bool,
) -> Path:
    """返回 schema_v3 宏观场二维产品的发布路径。"""
    _, _, product, var_code = get_publish_codes(config, metadata, variable_config)
    return (
        output_dir
        / product.lower()
        / var_code.lower()
        / "000hpa"
        / format_beijing_init(metadata)
        / f"F{metadata.forecast_hour:03d}"
        / build_image_name(config, metadata, variable_config, complete)
    )


def format_pressure_level(level: int | float) -> str:
    """将气压层格式化为目录名使用的简洁数字。"""
    level_value = float(level)
    if level_value.is_integer():
        return str(int(level_value))
    return f"{level_value:g}"


def build_3d_image_name(
    config: dict,
    metadata: ProductMetadata,
    variable_config: dict,
    pressure_level: int | float,
    complete: bool,
) -> str:
    """按照 schema_v3 生成三维产品文件名。"""
    model, region, product, var_code = get_publish_codes(config, metadata, variable_config)
    variant = "COMPLETE" if complete else "SIMPLE"
    parts = [
        model,
        region,
        format_beijing_init(metadata),
        f"{metadata.forecast_hour:03d}",
        product,
        variant,
        f"{format_pressure_level(pressure_level)}HPA",
        var_code,
    ]
    return "_".join(parts) + ".png"


def build_3d_image_path(
    config: dict,
    output_dir: Path,
    metadata: ProductMetadata,
    variable_config: dict,
    pressure_level: int | float,
    complete: bool,
) -> Path:
    """返回 schema_v3 指定气压层三维产品的发布路径。"""
    _, _, product, var_code = get_publish_codes(config, metadata, variable_config)
    level_text = format_pressure_level(pressure_level)
    return (
        output_dir
        / product.lower()
        / var_code.lower()
        / f"{level_text}hpa"
        / format_beijing_init(metadata)
        / f"F{metadata.forecast_hour:03d}"
        / build_3d_image_name(
            config,
            metadata,
            variable_config,
            pressure_level,
            complete,
        )
    )


def expected_image_paths(config: dict, source_file: str | Path, output_dir: Path) -> list[Path]:
    """返回一个 NC 文件按当前配置应该生成的全部 PNG。"""
    metadata = parse_product_filename(source_file)
    paths = []
    for variable_config in get_variable_configs(config, "2D"):
        # 完整底图暂时关闭，恢复时将 True 加回变体列表。
        for complete in (False,):
            paths.append(
                build_image_path(
                    config,
                    output_dir,
                    metadata,
                    variable_config,
                    complete,
                )
            )
    for pressure_level in read_pressure_levels(source_file, config):
        for variable_config in get_variable_configs(config, "3D"):
            # 完整底图暂时关闭，恢复时将 True 加回变体列表。
            for complete in (False,):
                paths.append(
                    build_3d_image_path(
                        config,
                        output_dir,
                        metadata,
                        variable_config,
                        pressure_level,
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
    colorbar_labels: list[str],
    colors: list[str],
    variable_config: dict,
    metadata: ProductMetadata,
    config: dict,
    output_file: Path,
    pressure_level: int | float | None = None,
) -> None:
    """按照样例图4绘制白底、顶部标题和内嵌色标的完整产品。"""
    extent = config["plot"]["extent"]
    figure = plt.figure(figsize=COMPLETE_SIZE, dpi=IMAGE_DPI, facecolor=FIGURE_BACKGROUND)
    axis = figure.add_axes([0.06, 0.04, 0.87, 0.91], projection=ccrs.PlateCarree())
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
    add_map_boundaries(axis, config, extent)
    add_china_province_labels(axis, config, extent)

    for spine in axis.spines.values():
        spine.set_linewidth(2.0)
        spine.set_edgecolor("#111111")

    pressure_text = (
        f"  {format_pressure_level(pressure_level)} hPa"
        if pressure_level is not None
        else ""
    )
    title = (
        f"{config['plot']['title_prefix']} {variable_config['display_name']}"
        f"{pressure_text} {metadata.valid_time:%Y年%m月%d日%H时}"
    )
    figure.text(
        0.06,
        0.98,
        title,
        fontsize=18,
        fontweight="bold",
        fontfamily=CHINESE_FONT,
        color="#202427",
        ha="left",
        va="top",
    )
    figure.text(
        0.93,
        0.978,
        f"(UTC) {metadata.init_time:%Y%m%d%H} F{metadata.forecast_hour:03d}",
        fontsize=12,
        fontfamily=CHINESE_FONT,
        color="#222222",
        ha="right",
        va="top",
    )
    draw_discrete_colorbar(figure, variable_config, colorbar_labels, colors)

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
    levels, colors, colorbar_labels, mask_below, color_mode = get_color_config(
        config,
        variable_config,
    )
    values = prepare_values(
        dataset[variable_name],
        variable_config,
        levels,
        mask_below_first_level=mask_below,
    )
    if color_mode == "interval":
        cmap, norm = build_interval_color_settings(levels, colors)
    else:
        cmap, norm = build_data_color_settings(levels, colors)

    complete_file = build_image_path(
        config,
        output_dir,
        metadata,
        variable_config,
        complete=True,
    )
    simple_file = build_image_path(
        config,
        output_dir,
        metadata,
        variable_config,
        complete=False,
    )
    # draw_complete_image(
    #     latitude,
    #     longitude,
    #     values,
    #     cmap,
    #     norm,
    #     colorbar_labels,
    #     colors,
    #     variable_config,
    #     metadata,
    #     config,
    #     complete_file,
    # )
    draw_simple_image(
        latitude,
        longitude,
        values,
        cmap,
        norm,
        config["plot"]["extent"],
        simple_file,
    )
    return [simple_file]


def plot_one_3d_level(
    dataset,
    config: dict,
    variable_config: dict,
    metadata: ProductMetadata,
    pressure_level: int | float,
    output_dir: Path,
) -> list[Path]:
    """为一个三维要素的一个气压层分别绘制完整图和透明图。"""
    plot_config = config["plot"]
    latitude = dataset[plot_config["latitude"]].values
    longitude = dataset[plot_config["longitude"]].values
    variable_name = variable_config["name"]
    level_name = plot_config["level"]
    level_data = dataset[variable_name].sel({level_name: pressure_level})

    levels, colors, colorbar_labels, mask_below, color_mode = get_color_config(
        config,
        variable_config,
    )
    values = prepare_values(
        level_data,
        variable_config,
        levels,
        mask_below_first_level=mask_below,
    )
    if color_mode == "interval":
        cmap, norm = build_interval_color_settings(levels, colors)
    else:
        cmap, norm = build_data_color_settings(levels, colors)

    complete_file = build_3d_image_path(
        config,
        output_dir,
        metadata,
        variable_config,
        pressure_level,
        complete=True,
    )
    simple_file = build_3d_image_path(
        config,
        output_dir,
        metadata,
        variable_config,
        pressure_level,
        complete=False,
    )
    # draw_complete_image(
    #     latitude,
    #     longitude,
    #     values,
    #     cmap,
    #     norm,
    #     colorbar_labels,
    #     colors,
    #     variable_config,
    #     metadata,
    #     config,
    #     complete_file,
    #     pressure_level=pressure_level,
    # )
    draw_simple_image(
        latitude,
        longitude,
        values,
        cmap,
        norm,
        plot_config["extent"],
        simple_file,
    )
    return [simple_file]


def plot_3d_variables(
    dataset,
    config: dict,
    metadata: ProductMetadata,
    output_dir: Path,
) -> list[Path]:
    """逐气压层绘制配置中的全部三维要素。"""
    variable_configs = get_variable_configs(config, "3D")
    if not variable_configs:
        return []

    level_name = config["plot"]["level"]
    pressure_levels = dataset[level_name].values
    image_files = []
    for pressure_level in pressure_levels:
        level_value = float(pressure_level)
        for variable_config in variable_configs:
            image_files.extend(
                plot_one_3d_level(
                    dataset,
                    config,
                    variable_config,
                    metadata,
                    level_value,
                    output_dir,
                )
            )
    return image_files


def plot_source_file(
    dataset_2d,
    dataset_3d,
    config: dict,
    source_file: str | Path,
    output_dir: Path,
) -> list[Path]:
    """绘制一个源文件中配置的全部二维要素和逐层三维要素。"""
    metadata = parse_product_filename(source_file)
    image_files = []
    for variable_config in get_variable_configs(config, "2D"):
        image_files.extend(
            plot_one_variable(
                dataset_2d,
                config,
                variable_config,
                metadata,
                output_dir,
            )
        )
    image_files.extend(plot_3d_variables(dataset_3d, config, metadata, output_dir))
    return image_files


def resolve_default_output_directory(config: dict, data_file: str | Path) -> Path:
    """返回单文件调试模式使用的模式图片根目录。"""
    parse_product_filename(data_file)
    output_config = config["output"]
    output_dir = Path(output_config["root_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main() -> None:
    """单独绘制一个 KeyMete NC 文件。"""
    parser = argparse.ArgumentParser(description="Draw configured products for one KeyMete NC file.")
    parser.add_argument("data_file", help="Input KeyMete NC file path.")
    parser.add_argument(
        "--output-dir",
        help="Model image root; defaults to <output.root_dir>.",
    )
    parser.add_argument("--config", type=Path, default=CONFIG_FILE, help="Path to config YAML file.")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else resolve_default_output_directory(config, args.data_file)
    )
    dataset_2d, dataset_3d = read_product_data_from_file(args.data_file, config)
    image_files = plot_source_file(
        dataset_2d,
        dataset_3d,
        config,
        args.data_file,
        output_dir,
    )
    for image_file in image_files:
        print(f"Saved image to: {image_file}")


if __name__ == "__main__":
    main()

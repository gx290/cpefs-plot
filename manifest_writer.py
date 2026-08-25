"""生成一个起报时间和预报时效对应的产品清单。"""

import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

from PIL import Image


FILE_NAME_CONVENTION = "schema_v3_product_category_level_beijing_time"


def format_beijing_cycle(metadata) -> str:
    """将产品的 UTC 起报时间转换为北京起报时间。"""
    return (metadata.init_time + timedelta(hours=8)).strftime("%Y%m%d%H")


def build_manifest_path(output_dir: Path, metadata) -> Path:
    """返回同一起报时间和预报时效的批次清单路径。"""
    return (
        output_dir
        / format_beijing_cycle(metadata)
        / f"F{metadata.forecast_hour:03d}"
        / "manifest.json"
    )


def format_utc_time(value: datetime) -> str:
    """将无时区的模式时间格式化为带 Z 后缀的 UTC 时间。"""
    return value.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def format_beijing_time(value: datetime) -> str:
    """将无时区的 UTC 模式时间格式化为北京时间。"""
    beijing_timezone = timezone(timedelta(hours=8))
    return value.replace(tzinfo=timezone.utc).astimezone(beijing_timezone).isoformat()


def calculate_file_sha256(file_path: Path) -> str:
    """分块计算图片文件的 SHA-256。"""
    digest = hashlib.sha256()
    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_manifest_variable_lookup(config: dict) -> dict[tuple[str, str], dict]:
    """按产品代码和要素代码建立图片到变量配置的索引。"""
    lookup = {}
    for variable_config in config["variables"]:
        key = (
            str(variable_config["product"]).lower(),
            str(variable_config["var_code"]).lower(),
        )
        if key in lookup:
            raise ValueError(
                "Duplicate manifest product/category mapping: "
                f"{key[0]}/{key[1]}"
            )
        lookup[key] = variable_config
    return lookup


def get_grid_manifest_info(config: dict) -> dict:
    """根据发布范围和固定网格间距生成统一的 GIS 网格元数据。"""
    west, east, south, north = [float(value) for value in config["plot"]["extent"]]
    spacing = float(config["manifest"]["grid_spacing_degrees"])
    longitude_intervals = (east - west) / spacing
    latitude_intervals = (north - south) / spacing
    rounded_longitude_intervals = round(longitude_intervals)
    rounded_latitude_intervals = round(latitude_intervals)

    if not math.isclose(longitude_intervals, rounded_longitude_intervals, abs_tol=1e-8):
        raise ValueError("plot.extent longitude span must be divisible by grid spacing")
    if not math.isclose(latitude_intervals, rounded_latitude_intervals, abs_tol=1e-8):
        raise ValueError("plot.extent latitude span must be divisible by grid spacing")

    half_spacing = spacing / 2
    return {
        "gridWidth": rounded_longitude_intervals + 1,
        "gridHeight": rounded_latitude_intervals + 1,
        "gridBounds": {
            "west": west,
            "south": south,
            "east": east,
            "north": north,
        },
        "imageBounds": {
            "west": round(west - half_spacing, 8),
            "south": round(south - half_spacing, 8),
            "east": round(east + half_spacing, 8),
            "north": round(north + half_spacing, 8),
        },
        "pixelSizeDegrees": {
            "longitude": spacing,
            "latitude": spacing,
        },
    }


def build_manifest_image_entry(
    image_path: Path,
    output_dir: Path,
    variable_lookup: dict[tuple[str, str], dict],
    metadata,
    grid_info: dict,
    default_opacity: float,
) -> dict:
    """读取一张已生成 PNG，并构造 manifest 的图片记录。"""
    try:
        relative_path = image_path.resolve().relative_to(output_dir.resolve())
    except ValueError as exc:
        raise ValueError(f"Image is outside output root: {image_path}") from exc

    path_parts = relative_path.parts
    if len(path_parts) < 6:
        raise ValueError(f"Unexpected schema_v3 image path: {relative_path}")
    product_type = path_parts[0].lower()
    category = path_parts[1].lower()
    variable_config = variable_lookup.get((product_type, category))
    if variable_config is None:
        raise ValueError(
            f"No variable config matches image path: {product_type}/{category}"
        )

    file_name = image_path.name.upper()
    if "_COMPLETE_" in file_name:
        variant = "complete"
    elif "_SIMPLE_" in file_name:
        variant = "simple"
    else:
        raise ValueError(f"Cannot determine image variant from file name: {image_path.name}")

    level_text = path_parts[2].lower()
    if not level_text.endswith("hpa"):
        raise ValueError(f"Cannot determine pressure level from path: {relative_path}")
    level_value = float(level_text[:-3])
    level = int(level_value) if level_value.is_integer() else level_value

    file_hash = calculate_file_sha256(image_path)
    with Image.open(image_path) as image:
        width, height = image.size
        image_mode = image.mode

    relative_text = relative_path.as_posix()
    image_id = hashlib.sha256(
        f"{relative_text}:{file_hash}".encode("utf-8")
    ).hexdigest()[:16]
    entry = {
        "id": image_id,
        "productType": product_type,
        "category": category,
        "variant": variant,
        "variable": str(variable_config["name"]).lower(),
        "relativePath": relative_text,
        "size": image_path.stat().st_size,
        "sha256": file_hash,
        "width": width,
        "height": height,
        "imageMode": image_mode,
        "level": level,
        "region": metadata.region_code,
    }
    if variant == "simple":
        entry["gisOverlay"] = {
            "crs": "EPSG:4326",
            "origin": "northwest",
            "transparent": True,
            "defaultOpacity": default_opacity,
            "imageWidth": width,
            "imageHeight": height,
            **grid_info,
        }
    return entry


def write_manifest(
    config: dict,
    metadata,
    source_file: str | Path,
    image_paths: list[Path],
    output_dir: Path,
    default_opacity: float,
) -> Path:
    """汇总一个起报时次和预报时效的全部图片并原子写入清单。"""
    model = str(config["output"]["model_code"]).strip().upper()
    region = metadata.region_code
    cycle = format_beijing_cycle(metadata)
    manifest_path = build_manifest_path(output_dir, metadata)
    variable_lookup = get_manifest_variable_lookup(config)
    grid_info = get_grid_manifest_info(config)
    image_entries = [
        build_manifest_image_entry(
            Path(image_path),
            output_dir,
            variable_lookup,
            metadata,
            grid_info,
            default_opacity,
        )
        for image_path in sorted(image_paths, key=lambda path: str(path).lower())
    ]

    manifest = {
        "schemaVersion": config["manifest"]["schema_version"],
        "productId": f"{model}:{cycle}:F{metadata.forecast_hour:03d}",
        "model": model,
        "region": region,
        "fileNameConvention": FILE_NAME_CONVENTION,
        "initialTime": format_utc_time(metadata.init_time),
        "initialTimeBeijing": format_beijing_time(metadata.init_time),
        "cycle": cycle,
        "forecastHour": metadata.forecast_hour,
        "validTime": format_utc_time(metadata.valid_time),
        "validTimeBeijing": format_beijing_time(metadata.valid_time),
        "status": "complete",
        "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "rendererVersion": config["manifest"]["renderer_version"],
        "sourceFiles": [Path(source_file).name],
        "imageCount": len(image_entries),
        "images": image_entries,
        "failedProducts": [],
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = manifest_path.with_suffix(".json.tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    temp_path.replace(manifest_path)
    return manifest_path

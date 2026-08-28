import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter

from dataload import (
    CONFIG_FILE,
    get_variable_configs,
    load_config,
    read_product_data_from_file,
)
from manifest_writer import build_manifest_path, write_manifest
from product_plotter import (
    SIMPLE_ALPHA,
    expected_image_paths,
    parse_product_filename,
    plot_source_file,
)


def log_event(event: str, **fields) -> None:
    """以统一的 key=value 格式输出运行节点。"""
    parts = []
    for key, value in fields.items():
        if isinstance(value, bool):
            text = str(value).lower()
        elif isinstance(value, (int, float)):
            text = str(value)
        else:
            text = json.dumps(str(value), ensure_ascii=False)
        parts.append(f"{key}={text}")
    details = " ".join(parts)
    print(f"[{event}] {details}".rstrip(), flush=True)


def parse_input_time(value: str) -> datetime:
    """解析云平台传入的小时级 UTC 时间。"""
    if len(value) != 14 or not value.isdigit():
        raise ValueError("input time must use YYYYMMDDHHMMSS format, for example: 20260724120000")
    return datetime.strptime(value, "%Y%m%d%H%M%S")


def build_time_tokens(value_time: datetime) -> dict[str, str]:
    """从一个时间对象生成目录模板可以使用的字段。"""
    return {
        "YYYY": value_time.strftime("%Y"),
        "MM": value_time.strftime("%m"),
        "DD": value_time.strftime("%d"),
        "HH": value_time.strftime("%H"),
        "YYYYMM": value_time.strftime("%Y%m"),
        "YYYYMMDD": value_time.strftime("%Y%m%d"),
        "YYYYMMDDHH": value_time.strftime("%Y%m%d%H"),
        "YYYYMMDDHHMM": value_time.strftime("%Y%m%d%H%M"),
        "YYYYMMDDHHMMSS": value_time.strftime("%Y%m%d%H%M%S"),
    }


def render_template(template: str, tokens: dict[str, str]) -> str:
    """使用时间字段生成实际目录或文件名。"""
    try:
        return template.format(**tokens)
    except KeyError as exc:
        raise ValueError(f"Unsupported time placeholder in template: {exc.args[0]}") from exc


def get_scan_window(config: dict, input_time: datetime, redraw: bool) -> tuple[datetime, datetime]:
    """计算正常回溯或精确重绘使用的 UTC 时间窗口。"""
    if redraw:
        return input_time, input_time

    lookback_hours = config["batch"]["lookback_hours"]
    if not isinstance(lookback_hours, int) or lookback_hours < 0:
        raise ValueError("batch.lookback_hours must be a non-negative integer")
    return input_time - timedelta(hours=lookback_hours), input_time


def get_source_directories(config: dict, start_time: datetime, end_time: datetime) -> list[Path]:
    """按小时渲染源目录模板，并去除日期模板产生的重复目录。"""
    source_config = config["source"]
    source_root = Path(source_config["root_dir"])
    directories: dict[Path, Path] = {}
    current_time = start_time
    while current_time <= end_time:
        relative_dir = render_template(
            source_config["directory_template"],
            build_time_tokens(current_time),
        )
        source_dir = source_root / relative_dir
        directories[source_dir.resolve()] = source_dir
        current_time += timedelta(hours=1)
    return list(directories.values())


def collect_source_files(
    config: dict,
    start_time: datetime,
    end_time: datetime,
) -> tuple[list[Path], dict[str, int]]:
    """扫描模板生成的目录，并按文件名中的起报时间筛选源文件。"""
    source_files: dict[Path, Path] = {}
    source_directories = get_source_directories(config, start_time, end_time)
    stats = {
        "directories_total": len(source_directories),
        "directories_existing": 0,
        "directories_missing": 0,
        "files_discovered": 0,
        "files_parse_failed": 0,
        "files_outside_window": 0,
        "files_matched": 0,
    }

    for source_dir in source_directories:
        exists = source_dir.is_dir()
        log_event("DIRECTORY", path=source_dir, status="exists" if exists else "missing")
        if not exists:
            stats["directories_missing"] += 1
            continue

        stats["directories_existing"] += 1
        for source_file in source_dir.glob(config["source"]["file_pattern"]):
            if not source_file.is_file():
                continue
            stats["files_discovered"] += 1

            try:
                file_init_time = parse_product_filename(source_file).init_time
            except ValueError as exc:
                stats["files_parse_failed"] += 1
                log_event("FILTER", source=source_file, reason="filename_parse_failed", detail=exc)
                continue

            file_time_format = "%Y%m%d%H"
            file_time_text = file_init_time.strftime(file_time_format)
            start_time_text = start_time.strftime(file_time_format)
            end_time_text = end_time.strftime(file_time_format)
            if not start_time_text <= file_time_text <= end_time_text:
                stats["files_outside_window"] += 1
                log_event(
                    "FILTER",
                    source=source_file,
                    reason="outside_time_window",
                    init_time=file_time_text,
                )
                continue

            source_files[source_file.resolve()] = source_file

    result = sorted(
        source_files.values(),
        key=lambda path: (parse_product_filename(path).init_time, path.name),
    )
    stats["files_matched"] = len(result)
    return result, stats


def load_state(state_file: Path) -> dict:
    """读取一个状态文件；文件不存在时返回空状态。"""
    if not state_file.exists():
        return {"files": {}}

    with state_file.open("r", encoding="utf-8") as file:
        state = json.load(file)
    if not isinstance(state, dict) or not isinstance(state.get("files"), dict):
        raise ValueError(f"Invalid state file format: {state_file}")
    return state


def save_state(state_file: Path, state: dict) -> None:
    """原子写入状态文件，避免任务中断留下不完整 JSON。"""
    state_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = state_file.with_suffix(f"{state_file.suffix}.tmp")
    with temp_file.open("w", encoding="utf-8") as file:
        json.dump(state, file, ensure_ascii=False, indent=2)
    temp_file.replace(state_file)


def get_state_directory(config: dict) -> Path:
    """返回按日状态文件的存储目录。"""
    output_root = Path(config["output"]["root_dir"])
    return output_root / config["state"]["directory"]


def resolve_state_file(config: dict, init_time: datetime) -> Path:
    """根据源文件起报日期生成对应的按日状态文件。"""
    file_name = render_template(
        config["state"]["filename_template"],
        build_time_tokens(init_time),
    )
    return get_state_directory(config) / file_name


def extract_state_file_date(file_name: str, template: str):
    """从按日状态文件名中解析日期。"""
    marker = "{YYYYMMDD}"
    if template.count(marker) != 1:
        raise ValueError("state.filename_template must contain {YYYYMMDD} exactly once")

    prefix, suffix = template.split(marker)
    if not file_name.startswith(prefix) or not file_name.endswith(suffix):
        return None
    end_index = len(file_name) - len(suffix) if suffix else len(file_name)
    date_text = file_name[len(prefix):end_index]
    if len(date_text) != 8 or not date_text.isdigit():
        return None
    return datetime.strptime(date_text, "%Y%m%d").date()


def cleanup_state_files(config: dict) -> dict[str, int]:
    """删除超过保留天数的按日状态文件。"""
    state_config = config["state"]
    retention_days = state_config["retention_days"]
    if not isinstance(retention_days, int) or retention_days <= 0:
        raise ValueError("state.retention_days must be a positive integer")

    state_dir = get_state_directory(config)
    state_dir.mkdir(parents=True, exist_ok=True)
    template = state_config["filename_template"]
    glob_pattern = template.replace("{YYYYMMDD}", "*")
    cutoff_date = datetime.now(timezone.utc).date() - timedelta(days=retention_days - 1)
    stats = {"files_seen": 0, "files_deleted": 0, "files_failed": 0}

    for state_file in state_dir.glob(glob_pattern):
        if not state_file.is_file():
            continue
        file_date = extract_state_file_date(state_file.name, template)
        if file_date is None:
            continue
        stats["files_seen"] += 1
        if file_date >= cutoff_date:
            continue
        try:
            state_file.unlink()
            stats["files_deleted"] += 1
            log_event("CLEANUP_FILE", path=state_file, status="deleted")
        except OSError as exc:
            stats["files_failed"] += 1
            log_event("WARNING", stage="state_cleanup", path=state_file, error=exc)

    log_event(
        "CLEANUP",
        retention_days=retention_days,
        cutoff_date=cutoff_date.strftime("%Y%m%d"),
        state_files_seen=stats["files_seen"],
        deleted=stats["files_deleted"],
        failed=stats["files_failed"],
    )
    return stats


def source_key(source_file: Path, source_root: Path) -> str:
    """生成源文件相对于输入顶层目录的稳定状态键。"""
    try:
        return source_file.relative_to(source_root).as_posix()
    except ValueError:
        return str(source_file.resolve())


def relative_image_paths(image_paths: list[Path], output_root: Path) -> list[str]:
    """将图片路径转换成相对于输出顶层目录的路径。"""
    result = []
    for image_path in image_paths:
        try:
            result.append(image_path.relative_to(output_root).as_posix())
        except ValueError:
            result.append(str(image_path))
    return result


def is_processed(
    state: dict,
    key: str,
    expected_paths: list[Path],
    manifest_path: Path,
) -> bool:
    """状态成功且当前配置要求的全部图片与清单存在时才判定为已完成。"""
    entry = state["files"].get(key, {})
    return (
        entry.get("status") == "success"
        and all(path.is_file() for path in expected_paths)
        and manifest_path.is_file()
    )


def load_relevant_states(
    config: dict,
    source_files: list[Path],
) -> tuple[dict[Path, dict], dict, int]:
    """只加载待检查源文件涉及的按日状态，以及旧版全局状态。"""
    state_cache: dict[Path, dict] = {}
    for source_file in source_files:
        init_time = parse_product_filename(source_file).init_time
        state_file = resolve_state_file(config, init_time)
        if state_file not in state_cache:
            state_cache[state_file] = load_state(state_file)

    output_root = Path(config["output"]["root_dir"])
    legacy_state_file = output_root / "processed_files.json"
    legacy_state = load_state(legacy_state_file)
    daily_records = sum(len(state["files"]) for state in state_cache.values())
    legacy_records = len(legacy_state["files"])
    log_event(
        "STATE",
        daily_state_files=len(state_cache),
        daily_records=daily_records,
        legacy_records=legacy_records,
        legacy_mode="read_only",
    )
    return state_cache, legacy_state, legacy_records


def process_batch(config: dict, input_time: datetime, redraw: bool) -> int:
    """处理窗口内未完成的 NC 文件，或精确重绘指定起报时次。"""
    batch_started = perf_counter()
    start_time, end_time = get_scan_window(config, input_time, redraw)
    source_root = Path(config["source"]["root_dir"])
    output_root = Path(config["output"]["root_dir"])
    output_root.mkdir(parents=True, exist_ok=True)

    log_event(
        "CONFIG",
        source_root=source_root,
        directory_template=config["source"]["directory_template"],
        file_pattern=config["source"]["file_pattern"],
        lookback_hours=config["batch"]["lookback_hours"],
        output_root=output_root,
        output_structure=(
            "{product}/{variable}/{pressure}hpa/"
            "{BEIJING_INIT}/F{forecast_hour:03d}/*.png"
        ),
        model_code=config["output"]["model_code"],
        region_source="source_filename_field_6",
        state_retention_days=config["state"]["retention_days"],
    )
    log_event(
        "WINDOW",
        start=start_time.strftime("%Y%m%d%H%M%S"),
        end=end_time.strftime("%Y%m%d%H%M%S"),
        timezone="UTC",
    )

    cleanup_state_files(config)
    source_files, scan_stats = collect_source_files(config, start_time, end_time)
    log_event(
        "DIRECTORY",
        total=scan_stats["directories_total"],
        existing=scan_stats["directories_existing"],
        missing=scan_stats["directories_missing"],
    )
    log_event(
        "SCAN",
        discovered=scan_stats["files_discovered"],
        parse_failed=scan_stats["files_parse_failed"],
        outside_window=scan_stats["files_outside_window"],
        matched=scan_stats["files_matched"],
    )

    state_cache, legacy_state, _ = load_relevant_states(config, source_files)
    if not source_files:
        log_event(
            "SUMMARY",
            status="no_files",
            drawn=0,
            skipped=0,
            failed=0,
            scanned_files=0,
            elapsed_seconds=round(perf_counter() - batch_started, 2),
        )
        return 1

    processed_count = 0
    skipped_count = 0
    failed_count = 0
    promoted_legacy_count = 0

    for source_file in source_files:
        file_started = perf_counter()
        product_metadata = parse_product_filename(source_file)
        file_init_time = product_metadata.init_time
        forecast_hour = product_metadata.forecast_hour
        output_dir = output_root
        state_file = resolve_state_file(config, file_init_time)
        state = state_cache[state_file]
        key = source_key(source_file, source_root)
        expected_paths = expected_image_paths(config, source_file, output_dir)
        manifest_path = build_manifest_path(output_dir, product_metadata)

        if key not in state["files"] and key in legacy_state["files"]:
            state["files"][key] = legacy_state["files"][key]
            save_state(state_file, state)
            promoted_legacy_count += 1
            log_event("STATE_MIGRATE", source=source_file, target=state_file)

        state_entry = state["files"].get(key, {})
        images_are_complete = all(path.is_file() for path in expected_paths)
        if (
            not redraw
            and state_entry.get("status") == "success"
            and images_are_complete
            and not manifest_path.is_file()
        ):
            try:
                manifest_path = write_manifest(
                    config,
                    product_metadata,
                    source_file,
                    expected_paths,
                    output_dir,
                    SIMPLE_ALPHA,
                )
                state_entry["manifest"] = relative_image_paths(
                    [manifest_path],
                    output_root,
                )[0]
                save_state(state_file, state)
                log_event(
                    "MANIFEST",
                    source=source_file.name,
                    path=manifest_path,
                    images=len(expected_paths),
                    status="backfilled",
                )
            except Exception as exc:
                log_event(
                    "WARNING",
                    stage="manifest_backfill",
                    source=source_file.name,
                    error=exc,
                )

        if not redraw and is_processed(state, key, expected_paths, manifest_path):
            log_event(
                "SKIP",
                source=source_file.name,
                reason="already_processed",
                expected_images=len(expected_paths),
                manifest=manifest_path,
            )
            skipped_count += 1
            continue

        stage = "prepare"
        try:
            log_event(
                "DRAW",
                source=source_file.name,
                region_code=product_metadata.region_code,
                init_time=file_init_time.strftime("%Y%m%d%H"),
                total_forecast_hours=product_metadata.total_forecast_hours,
                forecast_hour=forecast_hour,
                output_dir=output_dir,
            )

            stage = "read_nc"
            log_event("READ", source=source_file.name, path=source_file)
            dataset_2d, dataset_3d = read_product_data_from_file(source_file, config)

            stage = "plot"
            log_event(
                "PLOT",
                source=source_file.name,
                variables_2d=len(get_variable_configs(config, "2D")),
                variables_3d=len(get_variable_configs(config, "3D")),
                pressure_levels=dataset_3d.sizes.get(config["plot"].get("level", ""), 0),
                expected_images=len(expected_paths),
            )
            image_paths = plot_source_file(
                dataset_2d,
                dataset_3d,
                config,
                source_file,
                output_dir,
            )

            stage = "verify_output"
            if not all(path.is_file() for path in image_paths):
                raise RuntimeError("Not all configured image files were created")
            actual_paths = {path.resolve() for path in image_paths}
            required_paths = {path.resolve() for path in expected_paths}
            if actual_paths != required_paths:
                raise RuntimeError(
                    "Generated image set does not match the configured image set: "
                    f"generated={len(actual_paths)}, expected={len(required_paths)}"
                )

            stage = "write_manifest"
            manifest_path = write_manifest(
                config,
                product_metadata,
                source_file,
                image_paths,
                output_dir,
                SIMPLE_ALPHA,
            )
            log_event(
                "MANIFEST",
                source=source_file.name,
                path=manifest_path,
                images=len(image_paths),
                status="complete",
            )

            stage = "save_state"
            state["files"][key] = {
                "status": "success",
                "processed_time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "images": relative_image_paths(image_paths, output_root),
                "manifest": relative_image_paths([manifest_path], output_root)[0],
            }
            save_state(state_file, state)

            for image_path in image_paths:
                log_event("SAVE", source=source_file.name, image=image_path)
            processed_count += 1
            log_event(
                "DONE",
                source=source_file.name,
                images=len(image_paths),
                elapsed_seconds=round(perf_counter() - file_started, 2),
            )
        except Exception as exc:
            state["files"][key] = {
                "status": "failed",
                "last_attempt_time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "failed_stage": stage,
                "last_error": str(exc),
            }
            save_state(state_file, state)
            failed_count += 1
            log_event(
                "FAILED",
                source=source_file.name,
                stage=stage,
                error=exc,
                elapsed_seconds=round(perf_counter() - file_started, 2),
            )

    if promoted_legacy_count:
        log_event("STATE_MIGRATE", promoted_records=promoted_legacy_count)

    log_event(
        "SUMMARY",
        status="success" if failed_count == 0 else "partial_failure",
        drawn=processed_count,
        skipped=skipped_count,
        failed=failed_count,
        scanned_files=len(source_files),
        redraw=redraw,
        elapsed_seconds=round(perf_counter() - batch_started, 2),
    )
    return 1 if failed_count else 0


def main() -> int:
    """解析命令行参数并运行批处理任务。"""
    parser = argparse.ArgumentParser(description="Batch generate configured forecast products.")
    parser.add_argument("--input-time", required=True, help="UTC time in YYYYMMDDHHMMSS format.")
    parser.add_argument(
        "--redraw",
        action="store_true",
        help="Redraw the exact input init time and overwrite existing PNG files.",
    )
    parser.add_argument("--config", type=Path, default=CONFIG_FILE, help="Path to batch config YAML file.")
    args = parser.parse_args()

    mode = "redraw" if args.redraw else "normal"
    log_event("START", input_time=args.input_time, mode=mode, config=args.config)
    try:
        config = load_config(args.config)
        input_time = parse_input_time(args.input_time)
        return process_batch(config, input_time, args.redraw)
    except Exception as exc:
        log_event("ERROR", stage="startup_or_batch", error=exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

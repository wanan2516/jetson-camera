from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
CONFIG_PATH = CONFIG_DIR / "config.json"

ALLOWED_TARGETS = {"all", "high", "low"}
ALLOWED_ROI_TYPES = {"cleaning", "warning", "forbidden"}
ROI_TYPE_TO_CONFIG = {
    "cleaning": "clear_zone",
    "warning": "warning_zone",
    "forbidden": "forbidden_zone",
}
CONFIG_TYPE_TO_ROI = {
    "clear_zone": "cleaning",
    "warning_zone": "warning",
    "forbidden_zone": "forbidden",
}
DEFAULT_ROI_TYPES = {
    1: "cleaning",
    2: "warning",
    3: "forbidden",
}

DEFAULT_CONFIG: Dict[str, Any] = {
    "version": "1.0",
    "camera_id": "cam_demo_001",
    "detect_enable": True,
    "model_name": "240922",
    "backend": "tensorrt",
    "engine_path": "weights/yolo11n_person.engine",
    "model_path": "../weights/yolo11n.pt",
    "imgsz": 640,
    "device": None,
    "person_class_ids": [0],
    "thresholds": {
        "conf_thres": 0.35,
        "iou_thres": 0.45,
    },
    "overlap_threshold": 0.1,
    "match_enable": True,
    "match_threshold": 0.5,
    "match_frequency": 5,
    "target": "all",
    "alarm": {
        "enter_frames": 3,
        "exit_frames": 5,
    },
    "rois": [],
}


class ConfigValidationError(ValueError):
    """Raised when frontend config payloads fail validation."""


def ensure_config_files() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_json_file(CONFIG_PATH, DEFAULT_CONFIG)


def _ensure_json_file(path: Path, defaults: Dict[str, Any]) -> None:
    if not path.exists():
        _write_json(path, defaults)


def _read_json(path: Path, defaults: Dict[str, Any]) -> Dict[str, Any]:
    ensure_config_files()
    try:
        with path.open("r", encoding="utf-8") as config_file:
            data = json.load(config_file)
    except json.JSONDecodeError as exc:
        raise ConfigValidationError(f"{path.name} 不是合法 JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigValidationError(f"{path.name} 顶层结构必须是对象")
    return {**defaults, **data}


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as config_file:
        json.dump(data, config_file, ensure_ascii=False, indent=2)
        config_file.write("\n")


def get_detection_config() -> Dict[str, Any]:
    ensure_config_files()
    return {
        "config": _read_json(CONFIG_PATH, DEFAULT_CONFIG),
    }


def get_roi_config() -> Dict[str, Any]:
    config = _read_json(CONFIG_PATH, DEFAULT_CONFIG)
    return {"rois": config.get("rois", [])}


def save_detection_settings(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ConfigValidationError("请求体必须是 JSON 对象")

    config = _read_json(CONFIG_PATH, DEFAULT_CONFIG)

    if "detectionEnabled" in payload:
        config["detect_enable"] = _to_bool(payload["detectionEnabled"], "detectionEnabled")
    if "detectionModel" in payload:
        config["model_name"] = str(payload["detectionModel"])
    if "enginePath" in payload:
        config["engine_path"] = _non_empty_string(payload["enginePath"], "enginePath")
    if "detectionThreshold" in payload:
        thresholds = _ensure_dict(config, "thresholds")
        thresholds["conf_thres"] = _number_in_range(
            payload["detectionThreshold"], "detectionThreshold"
        )
    if "overlapRate" in payload:
        config["overlap_threshold"] = _number_in_range(payload["overlapRate"], "overlapRate")
    if "matchEnabled" in payload:
        config["match_enable"] = _to_bool(payload["matchEnabled"], "matchEnabled")
    if "matchThreshold" in payload:
        config["match_threshold"] = _number_in_range(payload["matchThreshold"], "matchThreshold")
    if "matchFrequency" in payload:
        config["match_frequency"] = _positive_int(payload["matchFrequency"], "matchFrequency")
    if "target" in payload:
        config["target"] = _validate_target(payload["target"])

    _write_json(CONFIG_PATH, config)
    return config


def save_detection_regions(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    if not isinstance(payload, dict):
        raise ConfigValidationError("请求体必须是 JSON 对象")

    target = _validate_target(payload.get("target", "all"))
    clear = _to_bool(payload.get("clear", False), "clear")

    config = _read_json(CONFIG_PATH, DEFAULT_CONFIG)
    existing_rois = config.get("rois", [])
    if not isinstance(existing_rois, list):
        raise ConfigValidationError("config.json 中 rois 必须是列表")

    preserved_rois = [
        roi for roi in existing_rois
        if isinstance(roi, dict) and _roi_target(roi) != target
    ]

    if clear:
        config["rois"] = preserved_rois
        _write_json(CONFIG_PATH, config)
        return config, True

    regions = payload.get("regions")
    if not isinstance(regions, list):
        raise ConfigValidationError("regions 必须是 list")
    if len(regions) > 3:
        raise ConfigValidationError("最多只能保存 3 个区域")

    config = _read_json(CONFIG_PATH, DEFAULT_CONFIG)
    overlap_threshold = _number_in_range(
        config.get("overlap_threshold", DEFAULT_CONFIG["overlap_threshold"]),
        "overlap_threshold",
    )

    new_rois = [
        _normalize_region(region, target, overlap_threshold)
        for region in regions
    ]
    config["rois"] = preserved_rois + new_rois
    _write_json(CONFIG_PATH, config)
    return config, False


def rect_to_points(rect: Dict[str, float]) -> List[List[float]]:
    x1, y1, x2, y2 = rect["x1"], rect["y1"], rect["x2"], rect["y2"]
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def rect_to_bbox(rect: Dict[str, float]) -> List[float]:
    return [rect["x1"], rect["y1"], rect["x2"] - rect["x1"], rect["y2"] - rect["y1"]]


def _normalize_region(region: Any, target: str, overlap_threshold: float) -> Dict[str, Any]:
    if not isinstance(region, dict):
        raise ConfigValidationError("regions 中每一项都必须是对象")

    region_id = _positive_int(region.get("id"), "id")
    if region_id not in DEFAULT_ROI_TYPES:
        raise ConfigValidationError("id 只允许 1 / 2 / 3")

    roi_type = _normalize_roi_type(region, region_id)
    roi_id = _non_empty_string(region.get("roi_id", f"region_{region_id}"), "roi_id")
    name = _non_empty_string(region.get("name", roi_id), "name")

    if "polygon" in region:
        polygon = _validate_polygon(region["polygon"])
        coordinate_mode = _validate_coordinate_mode(region.get("coordinate_mode", "absolute"))
        rect = _rect_from_points(polygon)
    else:
        rect = _validate_rect(region.get("rect"))
        polygon = rect_to_points(rect)
        coordinate_mode = "absolute"

    return {
        "id": region_id,
        "roi_id": roi_id,
        "name": name,
        "type": roi_type,
        "roi_type": ROI_TYPE_TO_CONFIG[roi_type],
        "target": target,
        "enabled": _to_bool(region.get("enabled", True), "enabled"),
        "rect": rect,
        "bbox": rect_to_bbox(rect),
        "points": polygon,
        "polygon": polygon,
        "coordinate_mode": coordinate_mode,
        "judge_method": _validate_judge_method(region.get("judge_method", "overlap")),
        "overlap_thres": _number_in_range(region.get("overlap_thres", overlap_threshold), "overlap_thres"),
        "overlap_threshold": overlap_threshold,
    }


def _validate_rect(value: Any) -> Dict[str, float]:
    if not isinstance(value, dict):
        raise ConfigValidationError("rect 必须是对象")

    missing = [key for key in ("x1", "y1", "x2", "y2") if key not in value]
    if missing:
        raise ConfigValidationError(f"rect 缺少字段: {', '.join(missing)}")

    rect = {key: _number(value[key], key) for key in ("x1", "y1", "x2", "y2")}
    if rect["x2"] <= rect["x1"]:
        raise ConfigValidationError("x2 必须大于 x1")
    if rect["y2"] <= rect["y1"]:
        raise ConfigValidationError("y2 必须大于 y1")
    return rect


def _validate_polygon(value: Any) -> List[List[float]]:
    if not isinstance(value, list) or len(value) < 3:
        raise ConfigValidationError("polygon 必须是至少 3 个点的列表")

    polygon = []
    for point in value:
        if not isinstance(point, list) or len(point) != 2:
            raise ConfigValidationError("polygon 中每个点必须是 [x, y]")
        polygon.append([_number(point[0], "polygon.x"), _number(point[1], "polygon.y")])
    return polygon


def _rect_from_points(points: List[List[float]]) -> Dict[str, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return {
        "x1": min(xs),
        "y1": min(ys),
        "x2": max(xs),
        "y2": max(ys),
    }


def _normalize_roi_type(region: Dict[str, Any], region_id: int) -> str:
    raw_type = region.get("type", region.get("roi_type", DEFAULT_ROI_TYPES[region_id]))
    roi_type = str(raw_type)
    if roi_type in CONFIG_TYPE_TO_ROI:
        return CONFIG_TYPE_TO_ROI[roi_type]
    if roi_type in ALLOWED_ROI_TYPES:
        return roi_type
    raise ConfigValidationError("type 只允许 cleaning / warning / forbidden，roi_type 只允许 clear_zone / warning_zone / forbidden_zone")


def _validate_coordinate_mode(value: Any) -> str:
    mode = str(value or "absolute")
    if mode not in {"absolute", "normalized"}:
        raise ConfigValidationError("coordinate_mode 只允许 absolute / normalized")
    return mode


def _validate_judge_method(value: Any) -> str:
    method = str(value or "overlap")
    if method not in {"foot_point", "center_point", "overlap"}:
        raise ConfigValidationError("judge_method 只允许 foot_point / center_point / overlap")
    return method


def _validate_target(value: Any) -> str:
    target = str(value or "all").lower()
    if target not in ALLOWED_TARGETS:
        raise ConfigValidationError("target 只允许 all / high / low")
    return target


def _roi_target(roi: Dict[str, Any]) -> str:
    target = str(roi.get("target", "all")).lower()
    return target if target in ALLOWED_TARGETS else "all"


def _ensure_dict(config: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        value = {}
        config[key] = value
    return value


def _non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigValidationError(f"{field_name} 必须是非空字符串")
    return value.strip()


def _to_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    raise ConfigValidationError(f"{field_name} 必须是 bool")


def _number(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ConfigValidationError(f"{field_name} 必须是数字")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigValidationError(f"{field_name} 必须是数字") from exc


def _number_in_range(value: Any, field_name: str) -> float:
    number = _number(value, field_name)
    if not 0 <= number <= 1:
        raise ConfigValidationError(f"{field_name} 必须在 [0, 1] 范围内")
    return number


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ConfigValidationError(f"{field_name} 必须是正整数")

    if isinstance(value, str) and not value.strip().isdigit():
        raise ConfigValidationError(f"{field_name} 必须是正整数")
    if isinstance(value, float) and not value.is_integer():
        raise ConfigValidationError(f"{field_name} 必须是正整数")

    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigValidationError(f"{field_name} 必须是正整数") from exc
    if number <= 0:
        raise ConfigValidationError(f"{field_name} 必须是正整数")
    return number

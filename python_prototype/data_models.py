from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class ROIType(str, Enum):
    CLEAR = "clear_zone"
    WARNING = "warning_zone"
    FORBIDDEN = "forbidden_zone"


class SystemState(str, Enum):
    SAFE = "safe"
    PRESTART_CHECKING = "prestart_checking"
    PRESTART_BLOCKED = "prestart_blocked"
    WARNING = "warning"
    ALARM = "alarm"


@dataclass
class ThresholdConfig:
    conf_thres: float = 0.35
    iou_thres: float = 0.45

    def __post_init__(self) -> None:
        if not 0.0 <= self.conf_thres <= 1.0:
            raise ValueError("conf_thres must be in [0.0, 1.0]")
        if not 0.0 <= self.iou_thres <= 1.0:
            raise ValueError("iou_thres must be in [0.0, 1.0]")


@dataclass
class ROIRule:
    roi_id: str
    name: str
    roi_type: ROIType
    polygon: List[Tuple[float, float]]
    judge_method: str = "foot_point"
    overlap_thres: float = 0.2
    coordinate_mode: str = "absolute"
    enabled: bool = True

    def __post_init__(self) -> None:
        allowed_methods = {"foot_point", "center_point", "overlap"}
        allowed_coordinate_modes = {"absolute", "normalized"}
        if self.judge_method not in allowed_methods:
            raise ValueError(f"judge_method must be one of {sorted(allowed_methods)}")
        if self.coordinate_mode not in allowed_coordinate_modes:
            raise ValueError(f"coordinate_mode must be one of {sorted(allowed_coordinate_modes)}")
        if len(self.polygon) < 3:
            raise ValueError(f"ROI {self.roi_id} polygon must contain at least 3 points")
        if not 0.0 <= self.overlap_thres <= 1.0:
            raise ValueError(f"ROI {self.roi_id} overlap_thres must be in [0.0, 1.0]")
        if self.coordinate_mode == "normalized":
            for x, y in self.polygon:
                if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
                    raise ValueError(f"ROI {self.roi_id} normalized polygon points must be in [0.0, 1.0]")


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]
    center: Tuple[int, int]
    foot_point: Tuple[int, int]
    roi_hits: List[Dict[str, Any]] = field(default_factory=list)
    track_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ZoneStatus:
    roi_id: str
    roi_name: str
    roi_type: str
    person_count: int = 0
    raw_active: bool = False
    stable_active: bool = False
    enter_counter: int = 0
    exit_counter: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FrameResult:
    frame_id: int
    timestamp: float
    detections: List[Dict[str, Any]]
    zone_summary: Dict[str, Dict[str, Any]]
    system_state: str
    allow_start: bool
    warning: bool
    alarm: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

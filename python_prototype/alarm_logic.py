from __future__ import annotations

from typing import Dict, List, Tuple

from data_models import Detection, ROIRule, ROIType, SystemState, ZoneStatus


class AlarmLogic:
    def __init__(self, roi_rules: List[ROIRule], enter_frames: int = 3, exit_frames: int = 5):
        self.enter_frames = enter_frames
        self.exit_frames = exit_frames
        self.zone_status: Dict[str, ZoneStatus] = {
            roi.roi_id: ZoneStatus(roi_id=roi.roi_id, roi_name=roi.name, roi_type=roi.roi_type.value)
            for roi in roi_rules
        }

    def update_zone_counts(self, detections: List[Detection]) -> None:
        current_counts = {roi_id: 0 for roi_id in self.zone_status}
        for detection in detections:
            for hit in detection.roi_hits:
                current_counts[hit["roi_id"]] += 1

        for roi_id, zone in self.zone_status.items():
            zone.person_count = current_counts[roi_id]
            zone.raw_active = current_counts[roi_id] > 0

    def update_state_machine(self) -> None:
        for zone in self.zone_status.values():
            if zone.raw_active:
                zone.enter_counter += 1
                zone.exit_counter = 0
                if zone.enter_counter >= self.enter_frames:
                    zone.stable_active = True
            else:
                zone.exit_counter += 1
                zone.enter_counter = 0
                if zone.exit_counter >= self.exit_frames:
                    zone.stable_active = False

    def evaluate(self, detections: List[Detection], prestart_mode: bool = False) -> Tuple[SystemState, bool, bool, bool]:
        self.update_zone_counts(detections)
        self.update_state_machine()

        clear_zones = [z for z in self.zone_status.values() if z.roi_type == ROIType.CLEAR.value]
        clear_active = any(z.stable_active for z in clear_zones)
        clear_confirmed = bool(clear_zones) and all(
            not z.stable_active and z.exit_counter >= self.exit_frames for z in clear_zones
        )
        warning_active = any(z.stable_active for z in self.zone_status.values() if z.roi_type == ROIType.WARNING.value)
        forbidden_active = any(z.stable_active for z in self.zone_status.values() if z.roi_type == ROIType.FORBIDDEN.value)

        alarm = forbidden_active
        warning = warning_active

        if alarm:
            state = SystemState.ALARM
        elif prestart_mode and clear_active:
            state = SystemState.PRESTART_BLOCKED
        elif warning:
            state = SystemState.WARNING
        elif prestart_mode and not clear_confirmed:
            state = SystemState.PRESTART_CHECKING
        else:
            state = SystemState.SAFE

        allow_start = prestart_mode and clear_confirmed and not warning and not alarm
        return state, allow_start, warning, alarm

    def build_zone_summary(self) -> Dict[str, Dict[str, object]]:
        return {roi_id: zone.to_dict() for roi_id, zone in self.zone_status.items()}

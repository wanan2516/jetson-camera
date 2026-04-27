from __future__ import annotations

from typing import List

import cv2
import numpy as np

from data_models import FrameResult, ROIType, ROIRule
from roi_manager import ROIManager


def draw_rois(frame: np.ndarray, roi_rules: List[ROIRule]) -> np.ndarray:
    canvas = frame.copy()
    overlay = frame.copy()
    color_map = {
        ROIType.CLEAR.value: (255, 255, 0),
        ROIType.WARNING.value: (0, 255, 255),
        ROIType.FORBIDDEN.value: (0, 0, 255),
    }

    for roi in roi_rules:
        pts = np.array(ROIManager.resolve_polygon(roi, frame.shape), dtype=np.int32)
        color = color_map[roi.roi_type.value]
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2)

    return cv2.addWeighted(overlay, 0.14, canvas, 0.86, 0)


def draw_result(frame: np.ndarray, result: FrameResult) -> np.ndarray:
    canvas = frame.copy()

    for detection in result.detections:
        x1, y1, x2, y2 = map(int, detection["bbox"])

        box_color = (52, 235, 88)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 1)

    return canvas

from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np

from data_models import Detection, ROIRule


class ROIManager:
    def __init__(self, roi_rules: List[ROIRule]):
        self.roi_rules = roi_rules
        self._mask_cache: Dict[Tuple[str, int, int], np.ndarray] = {}

    @staticmethod
    def resolve_polygon(roi: ROIRule, image_shape: Tuple[int, ...]) -> List[Tuple[int, int]]:
        height, width = image_shape[:2]
        if roi.coordinate_mode == "normalized":
            return [
                (
                    int(round(max(0.0, min(1.0, x)) * (width - 1))),
                    int(round(max(0.0, min(1.0, y)) * (height - 1))),
                )
                for x, y in roi.polygon
            ]
        return [(int(round(x)), int(round(y))) for x, y in roi.polygon]

    @staticmethod
    def point_in_polygon(point: Tuple[int, int], polygon: List[Tuple[int, int]]) -> bool:
        polygon_np = np.array(polygon, dtype=np.int32)
        return cv2.pointPolygonTest(polygon_np, point, False) >= 0

    def _get_roi_mask(self, roi: ROIRule, image_shape: Tuple[int, ...]) -> np.ndarray:
        height, width = image_shape[:2]
        cache_key = (roi.roi_id, width, height)
        cached = self._mask_cache.get(cache_key)
        if cached is not None:
            return cached

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(self.resolve_polygon(roi, image_shape), dtype=np.int32)], 1)
        self._mask_cache[cache_key] = mask
        return mask

    def bbox_overlap_ratio(self, bbox: List[float], roi: ROIRule, image_shape: Tuple[int, ...]) -> float:
        height, width = image_shape[:2]
        x1, y1, x2, y2 = bbox

        x1 = max(0, min(width, int(np.floor(x1))))
        y1 = max(0, min(height, int(np.floor(y1))))
        x2 = max(0, min(width, int(np.ceil(x2))))
        y2 = max(0, min(height, int(np.ceil(y2))))

        if x2 <= x1 or y2 <= y1:
            return 0.0

        roi_slice = self._get_roi_mask(roi, image_shape)[y1:y2, x1:x2]
        inter_area = int(np.count_nonzero(roi_slice))
        bbox_area = max((x2 - x1) * (y2 - y1), 1)
        return inter_area / bbox_area

    def judge_detection(self, detection: Detection, image_shape: Tuple[int, ...]) -> List[Dict[str, object]]:
        hits: List[Dict[str, object]] = []

        for roi in self.roi_rules:
            polygon = self.resolve_polygon(roi, image_shape)
            inside = False
            if roi.judge_method == "foot_point":
                inside = self.point_in_polygon(detection.foot_point, polygon)
            elif roi.judge_method == "center_point":
                inside = self.point_in_polygon(detection.center, polygon)
            elif roi.judge_method == "overlap":
                inside = self.bbox_overlap_ratio(detection.bbox, roi, image_shape) >= roi.overlap_thres
            else:
                raise ValueError(f"Unsupported judge_method: {roi.judge_method}")

            if inside:
                hits.append(
                    {
                        "roi_id": roi.roi_id,
                        "roi_name": roi.name,
                        "roi_type": roi.roi_type.value,
                        "inside": True,
                        "method": roi.judge_method,
                    }
                )

        return hits

    def apply(self, detections: List[Detection], image_shape: Tuple[int, ...]) -> List[Detection]:
        for detection in detections:
            detection.roi_hits = self.judge_detection(detection, image_shape)
        return detections

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np


class SystemState(str, Enum):
    SAFE = "safe"
    PRESTART_CHECKING = "prestart_checking"
    PRESTART_BLOCKED = "prestart_blocked"
    WARNING = "warning"
    ALARM = "alarm"


@dataclass
class Detection:
    bbox: Tuple[float, float, float, float]
    class_id: int = 0
    class_name: str = ""
    confidence: float = 0.0
    center: Tuple[float, float] = (0.0, 0.0)
    foot_point: Tuple[float, float] = (0.0, 0.0)
    roi_hits: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ROIRule:
    roi_id: str
    name: str
    roi_type: str
    polygon: List[Tuple[float, float]]
    judge_method: str = "foot_point"
    overlap_thres: float = 0.2
    coordinate_mode: str = "absolute"
    enabled: bool = True


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


@dataclass
class FrameResult:
    detections: List[Detection]
    zone_summary: List[ZoneStatus]
    system_state: SystemState = SystemState.SAFE
    allow_start: bool = False
    warning: bool = False
    alarm: bool = False


@dataclass
class InferenceConfig:
    input_size: Tuple[int, int] = (640, 640)
    score_threshold: float = 0.35
    iou_threshold: float = 0.45
    topk: int = 100
    num_labels: int = 0
    class_names: List[str] = field(default_factory=list)

    def get_class_name(self, class_id: int) -> str:
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return f"class_{class_id}"


@dataclass
class CameraInferResult:
    image: np.ndarray
    alarm: bool = False
    warning: bool = False
    system_state: SystemState = SystemState.SAFE


class ROIManager:
    def __init__(self, roi_rules: Sequence[ROIRule]):
        self.roi_rules = list(roi_rules)
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
    def point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[int, int]]) -> bool:
        if len(polygon) < 3:
            return False
        return cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), point, False) >= 0

    def _get_roi_mask(self, roi: ROIRule, image_shape: Tuple[int, ...]) -> np.ndarray:
        height, width = image_shape[:2]
        cache_key = (roi.roi_id, width, height)
        cached = self._mask_cache.get(cache_key)
        if cached is not None:
            return cached

        mask = np.zeros((height, width), dtype=np.uint8)
        polygon = np.array(self.resolve_polygon(roi, image_shape), dtype=np.int32)
        if len(polygon) >= 3:
            cv2.fillPoly(mask, [polygon], 1)
        self._mask_cache[cache_key] = mask
        return mask

    def bbox_overlap_ratio(self, bbox: Tuple[float, float, float, float], roi: ROIRule, image_shape: Tuple[int, ...]) -> float:
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

    def apply(self, detections: List[Detection], image_shape: Tuple[int, ...]) -> List[Detection]:
        for detection in detections:
            hits: List[Dict[str, Any]] = []
            for roi in self.roi_rules:
                polygon = self.resolve_polygon(roi, image_shape)
                inside = False
                if roi.judge_method == "foot_point":
                    inside = self.point_in_polygon(detection.foot_point, polygon)
                elif roi.judge_method == "center_point":
                    inside = self.point_in_polygon(detection.center, polygon)
                elif roi.judge_method == "overlap":
                    inside = self.bbox_overlap_ratio(detection.bbox, roi, image_shape) >= roi.overlap_thres

                if inside:
                    hits.append(
                        {
                            "roi_id": roi.roi_id,
                            "roi_name": roi.name,
                            "roi_type": roi.roi_type,
                            "inside": True,
                            "method": roi.judge_method,
                        }
                    )
            detection.roi_hits = hits
        return detections

    def draw_rois(self, image: np.ndarray) -> None:
        for roi in self.roi_rules:
            polygon = self.resolve_polygon(roi, image.shape)
            if len(polygon) < 3:
                continue
            color = (
                (0, 0, 255)
                if roi.roi_type == "forbidden_zone"
                else (0, 255, 255)
                if roi.roi_type == "warning_zone"
                else (0, 255, 0)
            )
            polygon_np = np.array(polygon, dtype=np.int32)
            cv2.polylines(image, [polygon_np], True, color, 2)
            cv2.putText(image, roi.name, polygon[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


class AlarmLogic:
    def __init__(self, roi_rules: Sequence[ROIRule], enter_frames: int = 3, exit_frames: int = 5):
        self.enter_frames = int(enter_frames)
        self.exit_frames = int(exit_frames)
        self.zone_status = [
            ZoneStatus(roi_id=roi.roi_id, roi_name=roi.name, roi_type=roi.roi_type)
            for roi in roi_rules
        ]

    def update_zone_counts(self, detections: Sequence[Detection]) -> None:
        for zone in self.zone_status:
            zone.person_count = 0

        for detection in detections:
            for hit in detection.roi_hits:
                for zone in self.zone_status:
                    if zone.roi_id == hit["roi_id"]:
                        zone.person_count += 1
                        break

        for zone in self.zone_status:
            zone.raw_active = zone.person_count > 0

    def update_state_machine(self) -> None:
        for zone in self.zone_status:
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

    def evaluate(self, detections: List[Detection], prestart_mode: bool = False) -> FrameResult:
        self.update_zone_counts(detections)
        self.update_state_machine()

        clear_active = False
        has_clear_zone = False
        warning_active = False
        forbidden_active = False

        for zone in self.zone_status:
            if zone.roi_type == "clear_zone":
                has_clear_zone = True
                clear_active = clear_active or zone.stable_active
            elif zone.roi_type == "warning_zone":
                warning_active = warning_active or zone.stable_active
            elif zone.roi_type == "forbidden_zone":
                forbidden_active = forbidden_active or zone.stable_active

        clear_confirmed = has_clear_zone
        for zone in self.zone_status:
            if zone.roi_type == "clear_zone" and (zone.stable_active or zone.exit_counter < self.exit_frames):
                clear_confirmed = False

        result = FrameResult(
            detections=detections,
            zone_summary=list(self.zone_status),
            alarm=forbidden_active,
            warning=warning_active,
        )

        if result.alarm:
            result.system_state = SystemState.ALARM
        elif prestart_mode and clear_active:
            result.system_state = SystemState.PRESTART_BLOCKED
        elif result.warning:
            result.system_state = SystemState.WARNING
        elif prestart_mode and not clear_confirmed:
            result.system_state = SystemState.PRESTART_CHECKING
        else:
            result.system_state = SystemState.SAFE

        result.allow_start = prestart_mode and clear_confirmed and not result.warning and not result.alarm
        return result


class TensorRTDetector:
    def __init__(self, engine_path: Path, config: InferenceConfig):
        self.engine_path = Path(engine_path)
        self.config = config
        self._load_runtime_modules()
        self.logger = self.trt.Logger(self.trt.Logger.ERROR)
        self.runtime = self.trt.Runtime(self.logger)
        with self.engine_path.open("rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"TensorRT engine 反序列化失败: {self.engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("TensorRT execution context 创建失败")

        self.input_names: List[str] = []
        self.output_names: List[str] = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == self.trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
        if not self.input_names or not self.output_names:
            raise RuntimeError("TensorRT engine 必须至少包含一个输入和一个输出 tensor")

        self.device_tensors: Dict[str, Any] = {}
        self._allocate_buffers()
        self._warmup()

    def _load_runtime_modules(self) -> None:
        try:
            import tensorrt as trt
            import torch
        except Exception as exc:
            raise RuntimeError("Python TensorRT 运行需要 tensorrt 和 torch CUDA 环境") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda 不可用，无法运行 TensorRT engine")
        self.trt = trt
        self.torch = torch

    def _torch_dtype(self, trt_dtype: Any) -> Any:
        mapping = {
            self.trt.float32: self.torch.float32,
            self.trt.float16: self.torch.float16,
            self.trt.int32: self.torch.int32,
            self.trt.int8: self.torch.int8,
            self.trt.bool: self.torch.bool,
        }
        return mapping.get(trt_dtype, self.torch.float32)

    def _allocate_buffers(self) -> None:
        height, width = self.config.input_size
        input_name = self.input_names[0]
        input_shape = (1, 3, int(height), int(width))
        self.context.set_input_shape(input_name, input_shape)

        for name in self.input_names + self.output_names:
            shape = tuple(int(dim) for dim in self.context.get_tensor_shape(name))
            if any(dim <= 0 for dim in shape):
                shape = tuple(int(dim) for dim in self.engine.get_tensor_shape(name))
            if any(dim <= 0 for dim in shape):
                raise RuntimeError(f"无法解析 TensorRT tensor shape: {name}={shape}")
            dtype = self._torch_dtype(self.engine.get_tensor_dtype(name))
            tensor = self.torch.empty(shape, device="cuda", dtype=dtype)
            self.device_tensors[name] = tensor
            self.context.set_tensor_address(name, tensor.data_ptr())

    def _warmup(self) -> None:
        input_name = self.input_names[0]
        self.device_tensors[input_name].zero_()
        for _ in range(5):
            self._execute()

    def _execute(self) -> None:
        stream = self.torch.cuda.current_stream()
        ok = self.context.execute_async_v3(stream_handle=stream.cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 执行失败")
        stream.synchronize()

    @staticmethod
    def _letterbox(image: np.ndarray, input_size: Tuple[int, int]) -> Tuple[np.ndarray, float, float, float]:
        input_h, input_w = input_size
        height, width = image.shape[:2]
        ratio = min(input_h / height, input_w / width)
        resized_w = int(round(width * ratio))
        resized_h = int(round(height * ratio))
        resized = cv2.resize(image, (resized_w, resized_h)) if (width, height) != (resized_w, resized_h) else image.copy()

        dw = (input_w - resized_w) / 2.0
        dh = (input_h - resized_h) / 2.0
        top = int(round(dh - 0.1))
        bottom = int(round(dh + 0.1))
        left = int(round(dw - 0.1))
        right = int(round(dw + 0.1))
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        return np.ascontiguousarray(tensor), ratio, dw, dh

    def infer(self, frame: np.ndarray) -> List[Detection]:
        tensor, ratio, dw, dh = self._letterbox(frame, self.config.input_size)
        input_name = self.input_names[0]
        input_tensor = self.torch.as_tensor(tensor, device="cuda")
        self.device_tensors[input_name].copy_(input_tensor)
        self._execute()
        output = self.device_tensors[self.output_names[0]].detach().cpu().numpy()
        return self._postprocess(output, frame.shape, ratio, dw, dh)

    def _postprocess(
        self,
        raw_output: np.ndarray,
        frame_shape: Tuple[int, ...],
        ratio: float,
        dw: float,
        dh: float,
    ) -> List[Detection]:
        predictions = np.squeeze(raw_output)
        if predictions.ndim != 2:
            raise RuntimeError(f"不支持的 TensorRT 输出形状: {raw_output.shape}")
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T
        if predictions.shape[1] < 5:
            raise RuntimeError(f"不支持的 YOLO 输出布局: {predictions.shape}")

        engine_num_classes = max(0, predictions.shape[1] - 4)
        num_labels = self.config.num_labels if self.config.num_labels > 0 else engine_num_classes
        num_labels = min(num_labels, engine_num_classes)
        if num_labels <= 0:
            return []

        class_scores = predictions[:, 4 : 4 + num_labels]
        labels = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(len(class_scores)), labels]
        keep_mask = scores > self.config.score_threshold
        if not np.any(keep_mask):
            return []

        boxes_xywh = predictions[keep_mask, :4].astype(np.float32)
        scores = scores[keep_mask].astype(np.float32)
        labels = labels[keep_mask].astype(np.int32)

        inv_ratio = 1.0 / max(ratio, 1e-6)
        height, width = frame_shape[:2]
        boxes_xyxy: List[Tuple[float, float, float, float]] = []
        nms_boxes: List[List[int]] = []
        score_list: List[float] = []
        label_list: List[int] = []

        for box, score, label in zip(boxes_xywh, scores, labels):
            cx, cy, bw, bh = box.tolist()
            cx -= dw
            cy -= dh
            x0 = float(np.clip((cx - 0.5 * bw) * inv_ratio, 0, width))
            y0 = float(np.clip((cy - 0.5 * bh) * inv_ratio, 0, height))
            x1 = float(np.clip((cx + 0.5 * bw) * inv_ratio, 0, width))
            y1 = float(np.clip((cy + 0.5 * bh) * inv_ratio, 0, height))
            if x1 <= x0 or y1 <= y0:
                continue
            boxes_xyxy.append((x0, y0, x1, y1))
            nms_boxes.append([int(round(x0)), int(round(y0)), int(round(x1 - x0)), int(round(y1 - y0))])
            score_list.append(float(score))
            label_list.append(int(label))

        if not nms_boxes:
            return []

        if hasattr(cv2.dnn, "NMSBoxesBatched"):
            indices = cv2.dnn.NMSBoxesBatched(
                nms_boxes,
                score_list,
                label_list,
                self.config.score_threshold,
                self.config.iou_threshold,
            )
        else:
            indices = cv2.dnn.NMSBoxes(nms_boxes, score_list, self.config.score_threshold, self.config.iou_threshold)
        if len(indices) == 0:
            return []

        detections: List[Detection] = []
        for idx in np.array(indices).reshape(-1)[: self.config.topk]:
            x0, y0, x1, y1 = boxes_xyxy[int(idx)]
            class_id = label_list[int(idx)]
            detection = Detection(
                bbox=(x0, y0, x1, y1),
                class_id=class_id,
                class_name=self.config.get_class_name(class_id),
                confidence=score_list[int(idx)],
                center=((x0 + x1) * 0.5, (y0 + y1) * 0.5),
                foot_point=((x0 + x1) * 0.5, y1),
            )
            detections.append(detection)
        return detections


class CameraTensorRTInfer:
    def __init__(
        self,
        engine_path: Path,
        runtime_config: Dict[str, Any],
        prestart_mode: bool = False,
        settle_single_frame: bool = False,
    ):
        self.prestart_mode = bool(prestart_mode)
        self.settle_single_frame = bool(settle_single_frame)
        self.inference_config = self._build_inference_config(runtime_config)
        self.roi_rules = self._build_roi_rules(runtime_config.get("rois", []))
        if not self.roi_rules:
            raise ValueError("没有找到可用 ROI 配置")
        self.detector = TensorRTDetector(Path(engine_path), self.inference_config)
        self.roi_manager = ROIManager(self.roi_rules)
        self.alarm_logic = AlarmLogic(
            self.roi_rules,
            enter_frames=int(runtime_config.get("enter_frames", 3)),
            exit_frames=int(runtime_config.get("exit_frames", 5)),
        )

    @staticmethod
    def _build_inference_config(config: Dict[str, Any]) -> InferenceConfig:
        thresholds = config.get("thresholds", {}) or {}
        imgsz = int(config.get("imgsz", 640))
        class_names = config.get("class_name", config.get("class_names", [])) or []
        if isinstance(class_names, str):
            class_names = [class_names]
        num_labels = int(config.get("num_labels", len(class_names) if class_names else 0))
        return InferenceConfig(
            input_size=(imgsz, imgsz),
            score_threshold=float(config.get("conf_thres", thresholds.get("conf_thres", 0.35))),
            iou_threshold=float(config.get("iou_thres", thresholds.get("iou_thres", 0.45))),
            topk=int(config.get("topk", 100)),
            num_labels=num_labels,
            class_names=[str(name) for name in class_names],
        )

    @staticmethod
    def _build_roi_rules(rois: Sequence[Dict[str, Any]]) -> List[ROIRule]:
        rules: List[ROIRule] = []
        for idx, roi in enumerate(rois):
            if not bool(roi.get("enabled", True)):
                continue
            polygon = roi.get("polygon", [])
            if len(polygon) < 3:
                raise ValueError(f"rois[{idx}].polygon 至少需要 3 个点")
            rules.append(
                ROIRule(
                    roi_id=str(roi.get("roi_id", f"roi_{idx + 1}")),
                    name=str(roi.get("name", roi.get("roi_id", f"ROI{idx + 1}"))),
                    roi_type=str(roi.get("roi_type", "forbidden_zone")),
                    polygon=[(float(point[0]), float(point[1])) for point in polygon],
                    judge_method=str(roi.get("judge_method", "foot_point")),
                    overlap_thres=float(roi.get("overlap_thres", 0.2)),
                    coordinate_mode=str(roi.get("coordinate_mode", "absolute")),
                    enabled=True,
                )
            )
        return rules

    def infer(self, input_img: np.ndarray) -> CameraInferResult:
        if input_img is None or input_img.size == 0:
            raise ValueError("Input image is empty")

        frame = input_img.copy()
        detections = self.detector.infer(frame)
        detections = self.roi_manager.apply(detections, frame.shape)

        eval_times = 1
        if self.settle_single_frame:
            eval_times = max(self.alarm_logic.enter_frames, self.alarm_logic.exit_frames)

        frame_result = FrameResult(detections=[], zone_summary=[])
        for _ in range(eval_times):
            frame_result = self.alarm_logic.evaluate(detections, self.prestart_mode)

        self._draw_result(frame, frame_result)
        return CameraInferResult(
            image=frame,
            alarm=frame_result.warning or frame_result.alarm,
            warning=frame_result.warning,
            system_state=frame_result.system_state,
        )

    def _draw_result(self, frame: np.ndarray, frame_result: FrameResult) -> None:
        self.roi_manager.draw_rois(frame)
        for detection in frame_result.detections:
            x0, y0, x1, y1 = [int(round(v)) for v in detection.bbox]
            cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 2)
            class_name = detection.class_name or f"class_{detection.class_id}"
            label = f"{class_name} {detection.confidence:.2f}"
            cv2.putText(frame, label, (x0, max(10, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            if any(hit["roi_type"] in ("warning_zone", "forbidden_zone") for hit in detection.roi_hits):
                text_y = max(18, y0 - 28)
                text = "WARNING"
                text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                bg_x1 = min(frame.shape[1] - 1, x0 + text_size[0] + 8)
                bg_y0 = max(0, text_y - text_size[1] - 4)
                bg_y1 = min(frame.shape[0] - 1, text_y + baseline + 4)
                if bg_x1 > x0 and bg_y1 > bg_y0:
                    cv2.rectangle(frame, (x0, bg_y0), (bg_x1, bg_y1), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, text, (x0 + 4, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

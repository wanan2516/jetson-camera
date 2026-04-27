from __future__ import annotations

from typing import List, Sequence

import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

from data_models import Detection, ThresholdConfig
from logger import setup_logger

logger = setup_logger("detector")


class PersonDetector:
    def __init__(
        self,
        model_path: str,
        thresholds: ThresholdConfig,
        imgsz: int | None = None,
        device: str | None = None,
        person_class_ids: Sequence[int] | None = (0,),
    ):
        self.model_path = model_path
        logger.info(f"加载 YOLOv11n 模型: {model_path}")
        self.model = YOLO(model_path)
        self.thresholds = thresholds
        self.imgsz = imgsz
        self.device = device
        self.person_class_ids = list(person_class_ids) if person_class_ids is not None else None
        logger.info(f"模型加载成功 - imgsz={imgsz}, device={device}, conf_thres={thresholds.conf_thres}")

    def infer(self, frame: np.ndarray) -> List[Detection]:
        predict_kwargs = {
            "source": frame,
            "conf": self.thresholds.conf_thres,
            "iou": self.thresholds.iou_thres,
            "verbose": False,
        }
        if self.imgsz is not None:
            predict_kwargs["imgsz"] = self.imgsz
        if self.device is not None:
            predict_kwargs["device"] = self.device
        if self.person_class_ids is not None:
            predict_kwargs["classes"] = self.person_class_ids

        results = self.model.predict(**predict_kwargs)

        detections: List[Detection] = []
        if not results:
            return detections

        result = results[0]
        if result.boxes is None:
            return detections

        names = result.names
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            class_name = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
            if class_name != "person":
                continue

            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(_build_detection(class_id, class_name, confidence, [x1, y1, x2, y2]))

        return detections

    def export_onnx(self, output_path: str | None = None, opset: int = 12, simplify: bool = True) -> str:
        from pathlib import Path
        import shutil

        if output_path is None:
            output_path = str(Path(self.model_path).with_suffix(".onnx"))

        logger.info(f"开始导出 ONNX: {output_path} (opset={opset}, simplify={simplify})")
        exported_path = self.model.export(format="onnx", opset=opset, simplify=simplify)
        if output_path == exported_path:
            logger.info(f"ONNX 导出成功: {exported_path}")
            return str(exported_path)

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(exported_path, output)
        logger.info(f"ONNX 导出成功: {output}")
        return str(output)


class ONNXRuntimePersonDetector:
    def __init__(
        self,
        model_path: str,
        thresholds: ThresholdConfig,
        imgsz: int | None = 640,
        device: str | None = None,
        person_class_ids: Sequence[int] | None = (0,),
    ):
        self.model_path = model_path
        self.thresholds = thresholds
        self.imgsz = imgsz or 640
        self.device = device
        self.person_class_ids = list(person_class_ids) if person_class_ids is not None else None
        logger.info(f"加载 ONNX 模型: {model_path}")
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        logger.info(f"ONNX 模型加载成功 - input={self.input_name}, outputs={len(self.output_names)}")

    def _letterbox(self, frame: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        original_height, original_width = frame.shape[:2]
        scale = min(self.imgsz / original_height, self.imgsz / original_width)
        resized_width = int(round(original_width * scale))
        resized_height = int(round(original_height * scale))

        resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        pad_x = (self.imgsz - resized_width) / 2
        pad_y = (self.imgsz - resized_height) / 2
        left = int(round(pad_x - 0.1))
        top = int(round(pad_y - 0.1))
        canvas[top:top + resized_height, left:left + resized_width] = resized

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)
        return tensor, scale, float(left), float(top)

    @staticmethod
    def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
        converted = np.empty_like(boxes)
        converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        return converted

    @staticmethod
    def _box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        box_area = max((box[2] - box[0]) * (box[3] - box[1]), 0.0)
        boxes_area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
        union = np.maximum(box_area + boxes_area - inter, 1e-6)
        return inter / union

    def _nms(self, boxes: np.ndarray, scores: np.ndarray) -> np.ndarray:
        if len(boxes) == 0:
            return np.empty((0,), dtype=np.int64)
        order = scores.argsort()[::-1]
        keep: list[int] = []
        while order.size > 0:
            index = int(order[0])
            keep.append(index)
            if order.size == 1:
                break
            ious = self._box_iou(boxes[index], boxes[order[1:]])
            order = order[1:][ious < self.thresholds.iou_thres]
        return np.array(keep, dtype=np.int64)

    def _decode_raw_predictions(
        self,
        raw_output: np.ndarray,
        frame_shape: tuple[int, ...],
        scale: float,
        pad_x: float,
        pad_y: float,
    ) -> List[Detection]:
        predictions = np.squeeze(raw_output)
        if predictions.ndim != 2:
            raise ValueError(f"Unsupported ONNX output shape: {raw_output.shape}")
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T

        if predictions.shape[1] < 5:
            raise ValueError(f"Unsupported ONNX output layout: {predictions.shape}")

        boxes = predictions[:, :4]
        class_scores = predictions[:, 4:]
        if class_scores.size == 0:
            return []

        if self.person_class_ids is None:
            class_ids = np.argmax(class_scores, axis=1)
            scores = class_scores[np.arange(len(class_scores)), class_ids]
        else:
            selected_scores = class_scores[:, self.person_class_ids]
            if selected_scores.ndim == 1:
                selected_scores = selected_scores[:, None]
            selected_indices = np.argmax(selected_scores, axis=1)
            scores = selected_scores[np.arange(len(selected_scores)), selected_indices]
            class_ids = np.array(self.person_class_ids, dtype=np.int64)[selected_indices]

        mask = scores >= self.thresholds.conf_thres
        if not np.any(mask):
            return []

        boxes = self._xywh_to_xyxy(boxes[mask]).astype(np.float32)
        scores = scores[mask].astype(np.float32)
        class_ids = class_ids[mask]

        boxes[:, [0, 2]] -= pad_x
        boxes[:, [1, 3]] -= pad_y
        boxes /= max(scale, 1e-6)

        height, width = frame_shape[:2]
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height - 1)

        keep = self._nms(boxes, scores)
        detections: List[Detection] = []
        for index in keep:
            class_id = int(class_ids[index])
            class_name = "person" if class_id == 0 else f"class_{class_id}"
            detections.append(_build_detection(class_id, class_name, float(scores[index]), boxes[index].tolist()))
        return detections

    def infer(self, frame: np.ndarray) -> List[Detection]:
        tensor, scale, pad_x, pad_y = self._letterbox(frame)
        outputs = self.session.run(self.output_names, {self.input_name: tensor})
        if not outputs:
            return []
        return self._decode_raw_predictions(outputs[0], frame.shape, scale, pad_x, pad_y)


def _build_detection(class_id: int, class_name: str, confidence: float, bbox: Sequence[float]) -> Detection:
    x1, y1, x2, y2 = bbox
    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
    foot_point = (int((x1 + x2) / 2), int(y2))
    return Detection(
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        bbox=[float(x1), float(y1), float(x2), float(y2)],
        center=center,
        foot_point=foot_point,
    )


def build_detector(
    model_path: str,
    thresholds: ThresholdConfig,
    imgsz: int | None = None,
    device: str | None = None,
    person_class_ids: Sequence[int] | None = (0,),
):
    if model_path.lower().endswith(".onnx"):
        return ONNXRuntimePersonDetector(
            model_path=model_path,
            thresholds=thresholds,
            imgsz=imgsz,
            device=device,
            person_class_ids=person_class_ids,
        )
    return PersonDetector(
        model_path=model_path,
        thresholds=thresholds,
        imgsz=imgsz,
        device=device,
        person_class_ids=person_class_ids,
    )

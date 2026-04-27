from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import onnx
import onnxruntime as ort


@dataclass
class ONNXValidationResult:
    model_path: str
    image_path: str | None
    providers: List[str]
    input_name: str
    input_shape: List[Any]
    output_names: List[str]
    output_shapes: List[List[Any]]
    image_shape: List[int] | None
    output_summaries: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ONNXValidator:
    def __init__(self, model_path: str, imgsz: int = 640):
        self.model_path = str(model_path)
        self.imgsz = imgsz

    def _preprocess(self, image_path: str) -> Tuple[np.ndarray, List[int]]:
        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")

        height, width = frame.shape[:2]
        resized = cv2.resize(frame, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)
        return tensor, [height, width]

    def validate(self, image_path: str | None = None) -> ONNXValidationResult:
        model = onnx.load(self.model_path)
        onnx.checker.check_model(model)

        session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if len(inputs) != 1:
            raise ValueError(f"Expected 1 ONNX input, got {len(inputs)}")

        input_info = inputs[0]
        input_name = input_info.name
        input_shape = list(input_info.shape)
        output_names = [output.name for output in outputs]
        output_shapes = [list(output.shape) for output in outputs]

        output_summaries: List[Dict[str, Any]] = []
        image_shape = None
        if image_path is not None:
            input_tensor, image_shape = self._preprocess(image_path)
            inference_outputs = session.run(output_names, {input_name: input_tensor})
            for name, value in zip(output_names, inference_outputs):
                value_np = np.asarray(value)
                output_summaries.append(
                    {
                        "name": name,
                        "dtype": str(value_np.dtype),
                        "shape": list(value_np.shape),
                        "min": float(value_np.min()) if value_np.size else None,
                        "max": float(value_np.max()) if value_np.size else None,
                        "mean": float(value_np.mean()) if value_np.size else None,
                    }
                )

        return ONNXValidationResult(
            model_path=str(Path(self.model_path).resolve()),
            image_path=str(Path(image_path).resolve()) if image_path is not None else None,
            providers=session.get_providers(),
            input_name=input_name,
            input_shape=input_shape,
            output_names=output_names,
            output_shapes=output_shapes,
            image_shape=image_shape,
            output_summaries=output_summaries,
        )

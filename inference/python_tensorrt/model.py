from __future__ import annotations

import atexit
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import yaml

from .runtime import CameraTensorRTInfer

try:
    import Jetson.GPIO as GPIO
except Exception:
    GPIO = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "app" / "AIConfig.yaml"


class Model:
    def __init__(self, config: str | os.PathLike[str] = DEFAULT_CONFIG):
        self.config_path = Path(config).resolve()
        self.config_base = self.config_path.parent

        self.cfg: Dict[str, Any] = {}
        self.engine_path: Path | None = None
        self.det_config_path: Path | None = None
        self.roi_config_path: Path | None = None
        self.runtime_config: Dict[str, Any] = {}

        self.prestart_mode = False
        self.settle_single_frame = False
        self.jpeg_quality = 95
        self.infer_runtime: CameraTensorRTInfer | None = None
        self.lock = threading.RLock()

        self.alarm_gpios: List[int] = []
        self.alarm_level = 1
        self.alarm_idle_level = 0
        self.alarm_duration = 0.0
        self.alarm_active = False
        self.alarm_end_time: float | None = None
        self.alarm_led_on = False
        self.last_alarm_flag = False

        self._load_config_and_prepare_runtime()
        self._init_alarm_gpio()
        self._check_files()
        self._start_python_tensorrt()
        atexit.register(self.close)

    def inference(self, frame: np.ndarray) -> np.ndarray:
        frame = self._check_frame(frame)
        with self.lock:
            if self.infer_runtime is None:
                self._start_python_tensorrt()
            if self.infer_runtime is None:
                raise RuntimeError("Python TensorRT 推理器未启动")
            infer_result = self.infer_runtime.infer(frame)
            self.last_alarm_flag = bool(infer_result.alarm)
            self._handle_alarm_gpio(self.last_alarm_flag)
            return infer_result.image

    def reload_config(self) -> None:
        with self.lock:
            self._release_runtime()
            self._load_config_and_prepare_runtime()
            self._init_alarm_gpio()
            self._check_files()
            self._start_python_tensorrt()

    def update_config(self, new_cfg: Dict[str, Any]) -> None:
        if not isinstance(new_cfg, dict):
            raise TypeError("new_cfg 必须是 dict")
        with self.lock:
            self._write_yaml_atomic(self.config_path, new_cfg)
            self.reload_config()

    def get_config(self) -> Dict[str, Any]:
        return self.cfg

    def close(self) -> None:
        with self.lock:
            self._release_runtime()
            self._release_alarm_gpio()

    def _release_runtime(self) -> None:
        self.infer_runtime = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _load_config_and_prepare_runtime(self) -> None:
        self.cfg = self._load_yaml(self.config_path)
        model_cfg = self.cfg.get("model", self.cfg)

        self.prestart_mode = bool(model_cfg.get("prestart_mode", False))
        self.settle_single_frame = bool(model_cfg.get("settle_single_frame", False))
        self.jpeg_quality = int(model_cfg.get("jpeg_quality", 95))

        model_path = model_cfg.get("model_path", "../inference/models")
        model_name = str(model_cfg.get("model_name", "yolo11n_person"))
        model_root = self._resolve_path(model_path)
        self.engine_path = self._resolve_engine_path(model_root, model_name)
        class_names = self._resolve_class_names(model_cfg, model_root, model_name)

        runtime_dir = self._resolve_path(model_cfg.get("runtime_config_dir", "../inference/configs/runtime"))
        runtime_dir.mkdir(parents=True, exist_ok=True)
        self.det_config_path = runtime_dir / "config_runtime.json"
        self.roi_config_path = runtime_dir / "roi_config_runtime.json"
        self.runtime_config = self._build_runtime_config(model_cfg, model_name, class_names)
        self._write_json_atomic(self.det_config_path, self.runtime_config)
        self._write_json_atomic(
            self.roi_config_path,
            {"version": self.runtime_config["version"], "rois": self.runtime_config["rois"]},
        )

    def _resolve_engine_path(self, model_root: Path, model_name: str) -> Path:
        candidates = [
            model_root / model_name / f"{model_name}.engine",
            model_root / f"{model_name}.engine",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _resolve_class_names(self, model_cfg: Dict[str, Any], model_root: Path, model_name: str) -> List[str]:
        candidates = [
            model_root / model_name / f"{model_name}.txt",
            model_root / f"{model_name}.txt",
        ]
        for candidate in candidates:
            if candidate.exists():
                with candidate.open("r", encoding="utf-8") as f:
                    names = [line.strip() for line in f if line.strip()]
                if names:
                    return names

        class_names = model_cfg.get("class_name", model_cfg.get("class_names", ["person"]))
        if isinstance(class_names, str):
            return [class_names]
        if isinstance(class_names, list) and class_names:
            return [str(name) for name in class_names]
        return ["person"]

    def _build_runtime_config(self, model_cfg: Dict[str, Any], model_name: str, class_names: List[str]) -> Dict[str, Any]:
        thresholds_cfg = model_cfg.get("thresholds", {}) or {}
        alarm_cfg = model_cfg.get("alarm", {}) or {}
        imgsz = int(model_cfg.get("imgsz", 640))
        conf_thres = float(model_cfg.get("conf_thres", thresholds_cfg.get("conf_thres", 0.35)))
        iou_thres = float(model_cfg.get("iou_thres", thresholds_cfg.get("iou_thres", 0.45)))
        enter_frames = int(model_cfg.get("enter_frames", alarm_cfg.get("enter_frames", 3)))
        exit_frames = int(model_cfg.get("exit_frames", alarm_cfg.get("exit_frames", 5)))
        rois = self._get_rois_from_config(model_cfg)

        return {
            "version": str(model_cfg.get("version", "1.0")),
            "camera_id": model_cfg.get("camera_id", "cam_default"),
            "detect_enable": bool(model_cfg.get("detect_enable", True)),
            "model_name": model_name,
            "class_name": class_names,
            "backend": "python_tensorrt",
            "engine_path": str(self.engine_path),
            "imgsz": imgsz,
            "device": model_cfg.get("device", "cuda"),
            "person_class_ids": model_cfg.get("person_class_ids", [0]),
            "thresholds": {"conf_thres": conf_thres, "iou_thres": iou_thres},
            "conf_thres": conf_thres,
            "iou_thres": iou_thres,
            "alarm": {"enter_frames": enter_frames, "exit_frames": exit_frames},
            "enter_frames": enter_frames,
            "exit_frames": exit_frames,
            "rois": rois,
        }

    def _get_rois_from_config(self, model_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        rois = model_cfg.get("rois", self.cfg.get("rois"))
        if rois:
            self._validate_rois(rois)
            return rois

        legacy_roi_config_path = model_cfg.get("roi_config_path")
        if legacy_roi_config_path:
            legacy_path = self._resolve_path(legacy_roi_config_path)
            if legacy_path.exists():
                legacy_cfg = self._load_json(legacy_path)
                rois = legacy_cfg.get("rois", [])
                if rois:
                    self._validate_rois(rois)
                    return rois

        raise ValueError("没有找到 ROI 配置。请在 AIConfig.yaml 中配置 model.rois，或者提供旧版 roi_config_path。")

    @staticmethod
    def _validate_rois(rois: Any) -> None:
        if not isinstance(rois, list):
            raise TypeError("rois 必须是 list")
        if not rois:
            raise ValueError("rois 不能为空")
        required_keys = ["roi_id", "name", "enabled", "roi_type", "judge_method", "coordinate_mode", "polygon"]
        for idx, roi in enumerate(rois):
            if not isinstance(roi, dict):
                raise TypeError(f"rois[{idx}] 必须是 dict")
            for key in required_keys:
                if key not in roi:
                    raise ValueError(f"rois[{idx}] 缺少字段: {key}")
            polygon = roi.get("polygon", [])
            if not isinstance(polygon, list) or len(polygon) < 3:
                raise ValueError(f"rois[{idx}].polygon 至少需要 3 个点")
            for point_idx, point in enumerate(polygon):
                if not isinstance(point, (list, tuple)) or len(point) != 2:
                    raise ValueError(f"rois[{idx}].polygon[{point_idx}] 必须是 [x, y]")

    def _start_python_tensorrt(self) -> None:
        if self.engine_path is None:
            raise RuntimeError("engine_path 尚未初始化")
        self.infer_runtime = CameraTensorRTInfer(
            engine_path=self.engine_path,
            runtime_config=self.runtime_config,
            prestart_mode=self.prestart_mode,
            settle_single_frame=self.settle_single_frame,
        )

    def _init_alarm_gpio(self) -> None:
        self._release_alarm_gpio()
        cfg = self.cfg.get("exception_output", {}) or {}
        gpio = cfg.get("gpio")
        if not gpio:
            return
        if GPIO is None:
            print("[alarm_gpio] Jetson.GPIO 未安装，跳过 GPIO 报警输出", file=sys.stderr)
            return

        pins = gpio if isinstance(gpio, list) else [gpio]
        normalized_pins = []
        for pin in pins:
            try:
                pin_number = int(pin)
            except Exception:
                continue
            if pin_number > 0:
                normalized_pins.append(pin_number)
        pins = sorted(set(normalized_pins))
        if not pins:
            return

        self.alarm_gpios = pins
        self.alarm_level = 1 if int(cfg.get("output_level", 1)) else 0
        self.alarm_idle_level = 0 if self.alarm_level == 1 else 1
        self.alarm_duration = float(cfg.get("duration", 0))

        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BOARD)
        for pin in self.alarm_gpios:
            GPIO.setup(pin, GPIO.OUT, initial=self.alarm_idle_level)

    def _handle_alarm_gpio(self, alarm_flag: bool) -> None:
        if not self.alarm_gpios or GPIO is None:
            return

        now = time.monotonic()
        if not alarm_flag:
            if self.alarm_led_on:
                for pin in self.alarm_gpios:
                    GPIO.output(pin, self.alarm_idle_level)
            self.alarm_active = False
            self.alarm_led_on = False
            self.alarm_end_time = None
            return

        if not self.alarm_active:
            self.alarm_active = True
            self.alarm_led_on = True
            for pin in self.alarm_gpios:
                GPIO.output(pin, self.alarm_level)
            self.alarm_end_time = now + self.alarm_duration if self.alarm_duration > 0 else None
            return

        if self.alarm_duration > 0 and self.alarm_led_on and self.alarm_end_time is not None and now >= self.alarm_end_time:
            for pin in self.alarm_gpios:
                GPIO.output(pin, self.alarm_idle_level)
            self.alarm_led_on = False

    def _release_alarm_gpio(self) -> None:
        if self.alarm_gpios and GPIO is not None:
            try:
                for pin in self.alarm_gpios:
                    GPIO.output(pin, self.alarm_idle_level)
                GPIO.cleanup(self.alarm_gpios)
            except Exception:
                pass
        self.alarm_gpios = []

    def _check_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame is None:
            raise ValueError("输入 frame 为 None")
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"输入必须是 np.ndarray，但收到: {type(frame)}")
        if frame.size == 0:
            raise ValueError("输入 frame 为空图像")
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"期望输入 HxWx3 BGR 图像，但收到 shape={frame.shape}")
        return np.ascontiguousarray(frame)

    def _check_files(self) -> None:
        if self.engine_path is None or not self.engine_path.exists():
            raise FileNotFoundError(f"TensorRT engine 不存在: {self.engine_path}")
        if self.det_config_path is None or not self.det_config_path.exists():
            raise FileNotFoundError(f"runtime 检测配置不存在: {self.det_config_path}")
        if self.roi_config_path is None or not self.roi_config_path.exists():
            raise FileNotFoundError(f"runtime ROI 配置不存在: {self.roi_config_path}")

    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"配置文件不存在: {path}")
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def _load_json(path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _write_json_atomic(path: Path, data: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)

    @staticmethod
    def _write_yaml_atomic(path: Path, data: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
        os.replace(tmp_path, path)

    def _resolve_path(self, path: str | os.PathLike[str]) -> Path:
        path_obj = Path(path)
        if path_obj.is_absolute():
            return path_obj
        return (self.config_base / path_obj).resolve()

    def __enter__(self) -> "Model":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

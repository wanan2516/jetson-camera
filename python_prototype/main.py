from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

import cv2

from alarm_logic import AlarmLogic
from data_models import FrameResult, ROIRule, ROIType, ThresholdConfig
from detector import build_detector
from onnx_validator import ONNXValidator
from roi_manager import ROIManager
from visualizer import draw_result, draw_rois
from logger import setup_logger

logger = setup_logger("main")


class SafetyVisionSystem:
    def __init__(self, config: dict):
        logger.info(f"初始化安全视觉系统 - version={config.get('version', '1.0')}, camera_id={config.get('camera_id', 'default_camera')}")
        thresholds = ThresholdConfig(**config["thresholds"])
        roi_rules = self._build_roi_rules(config.get("rois", []))
        alarm_cfg = config["alarm"]
        self.config_version = config.get("version", "1.0")
        self.camera_id = config.get("camera_id", "default_camera")

        self.roi_rules = roi_rules
        self.detector = build_detector(
            config["_model_path"],
            thresholds,
            imgsz=config.get("imgsz"),
            device=config.get("device"),
            person_class_ids=config.get("person_class_ids", [0]),
        )
        self.roi_manager = ROIManager(roi_rules)
        self.alarm_logic = AlarmLogic(
            roi_rules=roi_rules,
            enter_frames=alarm_cfg.get("enter_frames", 3),
            exit_frames=alarm_cfg.get("exit_frames", 5),
        )
        self.frame_id = 0
        logger.info(f"系统初始化完成 - ROI数量={len(roi_rules)}, enter_frames={alarm_cfg.get('enter_frames', 3)}")

    @staticmethod
    def _build_roi_rules(roi_configs: List[dict]) -> List[ROIRule]:
        rules: List[ROIRule] = []
        for roi in roi_configs:
            if not roi.get("enabled", True):
                continue
            rules.append(
                ROIRule(
                    roi_id=roi["roi_id"],
                    name=roi["name"],
                    roi_type=ROIType(roi["roi_type"]),
                    polygon=[tuple(point) for point in roi["polygon"]],
                    judge_method=roi.get("judge_method", "foot_point"),
                    overlap_thres=roi.get("overlap_thres", 0.2),
                    coordinate_mode=roi.get("coordinate_mode", "absolute"),
                    enabled=roi.get("enabled", True),
                )
            )
        return rules

    def process_frame(self, frame, prestart_mode: bool = False) -> FrameResult:
        self.frame_id += 1
        detections = self.detector.infer(frame)
        detections = self.roi_manager.apply(detections, frame.shape)
        state, allow_start, warning, alarm = self.alarm_logic.evaluate(detections, prestart_mode=prestart_mode)

        return FrameResult(
            frame_id=self.frame_id,
            timestamp=time.time(),
            detections=[detection.to_dict() for detection in detections],
            zone_summary=self.alarm_logic.build_zone_summary(),
            system_state=state.value,
            allow_start=allow_start,
            warning=warning,
            alarm=alarm,
        )


def resolve_path(path: str, base_dir: Path) -> str:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return str(path_obj)
    return str((base_dir / path_obj).resolve())


def resolve_cli_path(path: str, config: dict) -> str:
    path_obj = Path(path).expanduser()
    if path_obj.is_absolute():
        return str(path_obj)

    cwd_path = Path.cwd() / path_obj
    if cwd_path.exists():
        return str(cwd_path.resolve())

    return resolve_path(path, Path(config["_config_dir"]))


def load_config(config_path: str) -> dict:
    config_file = Path(config_path).expanduser().resolve()
    if not config_file.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_file}")

    logger.info(f"加载配置文件: {config_file}")
    with open(config_file, "r", encoding="utf-8") as file:
        config = json.load(file)
    config["_config_path"] = str(config_file)
    config["_config_dir"] = str(config_file.parent)
    config["_model_path"] = resolve_path(config["model_path"], config_file.parent)
    logger.info(f"配置加载成功 - model_path={config['_model_path']}")
    return config


def process_image(system: SafetyVisionSystem, image_path: str, output_path: str | None, prestart_mode: bool) -> None:
    logger.info(f"处理图像: {image_path}")
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    settle_frames = max(system.alarm_logic.enter_frames, system.alarm_logic.exit_frames)
    result = None
    for _ in range(settle_frames):
        result = system.process_frame(frame, prestart_mode=prestart_mode)
    if result is None:
        raise RuntimeError("Failed to process image")
    vis = draw_result(draw_rois(frame, system.roi_rules), result)

    logger.info(f"检测结果: 检测到 {len(result.detections)} 个目标, 状态={result.system_state}")
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))

    if output_path:
        cv2.imwrite(output_path, vis)
        logger.info(f"结果已保存: {output_path}")
    else:
        cv2.imshow("Safety Vision", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_stream(
    system: SafetyVisionSystem,
    source: int | str,
    prestart_mode: bool,
    output_path: str | None,
    display: bool,
) -> None:
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open source: {source}")

    writer = None
    if output_path:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        result = system.process_frame(frame, prestart_mode=prestart_mode)
        vis = draw_result(draw_rois(frame, system.roi_rules), result)

        if writer is not None:
            writer.write(vis)

        if display:
            cv2.imshow("Safety Vision", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    capture.release()
    if writer is not None:
        writer.release()
    if display:
        cv2.destroyAllWindows()


def export_onnx(system: SafetyVisionSystem, output_path: str | None) -> None:
    onnx_path = system.detector.export_onnx(output_path=output_path)
    print(json.dumps({"onnx_path": onnx_path}, ensure_ascii=False, indent=2))


def validate_onnx(system: SafetyVisionSystem, image_path: str | None, onnx_path: str | None) -> None:
    if onnx_path is not None:
        model_path = onnx_path
    elif str(system.detector.model_path).lower().endswith(".onnx"):
        model_path = str(system.detector.model_path)
    else:
        model_path = str(system.detector.export_onnx())
    validator = ONNXValidator(model_path=model_path, imgsz=getattr(system.detector, "imgsz", 640) or 640)
    result = validator.validate(image_path=image_path)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


def validate_onnx_from_config(config: dict, image_path: str | None, onnx_path: str | None) -> None:
    system = SafetyVisionSystem(config)
    resolved_onnx_path = resolve_cli_path(onnx_path, config) if onnx_path is not None else None
    validate_onnx(system, image_path, resolved_onnx_path)


def infer_onnx(system: SafetyVisionSystem, image_path: str, output_path: str | None) -> None:
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    settle_frames = max(system.alarm_logic.enter_frames, system.alarm_logic.exit_frames)
    result = None
    for _ in range(settle_frames):
        result = system.process_frame(frame, prestart_mode=False)
    if result is None:
        raise RuntimeError("Failed to process image")
    vis = draw_result(draw_rois(frame, system.roi_rules), result)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))

    if output_path:
        cv2.imwrite(output_path, vis)



def export_then_infer_onnx(config: dict, image_path: str, output_path: str | None) -> None:
    thresholds = ThresholdConfig(**config["thresholds"])
    pytorch_detector = build_detector(
        config["_model_path"],
        thresholds,
        imgsz=config.get("imgsz"),
        device=config.get("device"),
        person_class_ids=config.get("person_class_ids", [0]),
    )
    onnx_path = str(pytorch_detector.export_onnx())
    config_for_onnx = dict(config)
    config_for_onnx["model_path"] = onnx_path
    config_for_onnx["_model_path"] = onnx_path
    system = SafetyVisionSystem(config_for_onnx)
    infer_onnx(system, image_path, output_path)



def run_decode_onnx(args: argparse.Namespace, config: dict) -> None:
    if not args.input:
        raise ValueError("decode_onnx mode requires --input <image_path>")
    if args.onnx is not None:
        config = dict(config)
        config["model_path"] = args.onnx
        config["_model_path"] = resolve_cli_path(args.onnx, config)
        system = SafetyVisionSystem(config)
        infer_onnx(system, args.input, args.output)
    else:
        export_then_infer_onnx(config, args.input, args.output)



def compare_backends(config: dict, image_path: str) -> None:
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    pytorch_system = SafetyVisionSystem(config)
    pytorch_result = pytorch_system.process_frame(frame, prestart_mode=False)

    thresholds = ThresholdConfig(**config["thresholds"])
    onnx_path = str(
        build_detector(
            config["_model_path"],
            thresholds,
            imgsz=config.get("imgsz"),
            device=config.get("device"),
            person_class_ids=config.get("person_class_ids", [0]),
        ).export_onnx()
    )
    config_for_onnx = dict(config)
    config_for_onnx["model_path"] = onnx_path
    config_for_onnx["_model_path"] = onnx_path
    onnx_system = SafetyVisionSystem(config_for_onnx)
    onnx_result = onnx_system.process_frame(frame, prestart_mode=False)

    comparison = {
        "pytorch_detection_count": len(pytorch_result.detections),
        "onnx_detection_count": len(onnx_result.detections),
        "pytorch_detections": pytorch_result.detections,
        "onnx_detections": onnx_result.detections,
    }
    print(json.dumps(comparison, ensure_ascii=False, indent=2))




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv11n industrial safety vision prototype")
    default_config = Path(__file__).with_name("config.json")
    parser.add_argument("--config", type=str, default=str(default_config), help="Path to config.json")
    parser.add_argument("--mode", type=str, choices=["image", "video", "camera", "export_onnx", "validate_onnx", "decode_onnx", "compare_backends"], default="image")
    parser.add_argument("--input", type=str, help="Image path, video path, camera index, or validation image")
    parser.add_argument("--output", type=str, default=None, help="Optional output image/video/onnx path")
    parser.add_argument("--prestart", action="store_true", help="Enable prestart clear-zone logic")
    parser.add_argument("--no-display", action="store_true", help="Disable GUI display for video/camera mode")
    parser.add_argument("--onnx", type=str, default=None, help="Existing ONNX path for validation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    system = SafetyVisionSystem(config)

    if args.mode == "export_onnx":
        export_onnx(system, args.output)
    elif args.mode == "validate_onnx":
        validate_onnx_from_config(config, args.input, args.onnx or args.output)
    elif args.mode == "decode_onnx":
        run_decode_onnx(args, config)
    elif args.mode == "compare_backends":
        if not args.input:
            raise ValueError("compare_backends mode requires --input <image_path>")
        compare_backends(config, args.input)
    elif args.mode == "image":
        if not args.input:
            raise ValueError("Image mode requires --input <image_path>")
        process_image(system, args.input, args.output, args.prestart)
    elif args.mode == "video":
        if not args.input:
            raise ValueError("Video mode requires --input <video_path>")
        process_stream(system, args.input, args.prestart, args.output, display=not args.no_display)
    else:
        camera_index = int(args.input) if args.input is not None else 0
        process_stream(system, camera_index, args.prestart, args.output, display=not args.no_display)


if __name__ == "__main__":
    main()

# YOLOv11n Industrial Safety Person Detection Plan

## 1. System Scope

This prototype implements:

- Person detection with YOLOv11n pretrained weights.
- Configurable confidence and NMS IoU thresholds.
- Multiple polygon ROI zones for clear-zone, warning-zone, and forbidden-zone logic.
- Frame smoothing with enter and exit counters to reduce false alarms.
- ONNX export path for later Jetson TensorRT deployment.

The first deployable class is `person` only. Algorithm innovation is intentionally kept low; the goal is a stable engineering baseline that can be tuned on site.

## 2. Model Workflow

Use pretrained YOLOv11n first, then fine-tune on site data.

Recommended steps:

1. Collect images from target cameras in normal, low-light, backlight, occlusion, PPE, and far-distance cases.
2. Label only one class: `person`.
3. Split data by scene, not by adjacent frames, to avoid validation leakage.
4. Start fine-tuning from `yolo11n.pt`, not from scratch.
5. Validate with precision, recall, mAP50, false alarm samples, missed person samples, and per-zone business replay.

Example training commands:

```bash
yolo detect train model=yolo11n.pt data=person.yaml imgsz=640 epochs=80 batch=16 workers=4
yolo detect val model=runs/detect/train/weights/best.pt data=person.yaml imgsz=640 conf=0.35 iou=0.45
```

Example inference command in this project:

```bash
python python_prototype/main.py --config python_prototype/config.json --mode image --input assets/test.jpg --output assets/result.jpg --prestart --no-display
```

## 3. ONNX and TensorRT Path

This project provides an ONNX export mode:

```bash
python python_prototype/main.py --config python_prototype/config.json --mode export_onnx --output weights/yolo11n_person.onnx
```

Ultralytics CLI equivalent:

```bash
yolo export model=runs/detect/train/weights/best.pt format=onnx imgsz=640 opset=12 simplify=True
```

Jetson TensorRT conversion example:

```bash
trtexec --onnx=weights/yolo11n_person.onnx --saveEngine=weights/yolo11n_person_fp16.engine --fp16
```

Deployment notes:

- Keep `imgsz` fixed during export and runtime unless dynamic shape is explicitly required.
- Prefer FP16 on Jetson for real-time throughput.
- Calibrate INT8 only after representative site data is available.
- Keep post-processing thresholds configurable outside the engine.

## 4. ROI Judgment Methods

Supported methods:

- `foot_point`: checks whether the bottom-center point of the person box is inside the ROI.
- `center_point`: checks whether the bbox center is inside the ROI.
- `overlap`: checks whether bbox area overlap with ROI exceeds `overlap_thres`.

Comparison:

- `foot_point` is best when the ROI represents a ground-plane danger area. It is stable for standing people and avoids triggering when only the upper body visually overlaps a region.
- `center_point` is simple and fast, but can miss cases where a person's feet have entered the region while the bbox center is outside.
- `overlap` is more conservative for irregular zones and partial body intrusion, but costs more computation and needs threshold tuning.

Default recommendation:

- Use `foot_point` for industrial safety zones if the camera view is fixed and ROIs are drawn on the floor plane.
- Use `overlap` for overhead cameras, non-floor ROIs, or cases where any body part entering the zone should trigger.

## 5. Alarm Logic

Current logic:

- A zone becomes active only after `enter_frames` consecutive active frames.
- A zone becomes inactive only after `exit_frames` consecutive inactive frames.
- Prestart clear-zone mode allows equipment startup only after clear zones are continuously empty for `exit_frames` frames.

Business outputs:

- Clear-zone prestart: `allow_start=True` only when configured clear zones are confirmed empty.
- Warning zone: stable person presence returns `warning=True`.
- Forbidden zone: stable person presence returns `alarm=True`.

Recommended site defaults:

- Start with `conf_thres=0.35`, `iou_thres=0.45`.
- Use `enter_frames=3` and `exit_frames=5` at 25 FPS.
- Increase `enter_frames` for vibration or lighting flicker.
- Increase `exit_frames` if alarms should latch longer after a person exits.

## 6. Current Code Entry Points

- `python_prototype/main.py`: CLI and system orchestration.
- `python_prototype/detector.py`: YOLOv11n person inference and ONNX export.
- `python_prototype/roi_manager.py`: ROI point and overlap judgment.
- `python_prototype/alarm_logic.py`: frame smoothing and business state evaluation.
- `python_prototype/config.json`: model path, thresholds, device, image size, alarm counters, and ROI definitions.

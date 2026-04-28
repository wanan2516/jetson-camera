# C++ TensorRT Module

This module runs the YOLOv11 person detector through TensorRT C++ on Jetson and applies the same minimal ROI/alarm business logic as the Python prototype.

## Target Jetson environment

Verified target:

- Jetson Orin Nano
- JetPack/L4T R36.5.0
- CUDA 12.6 from JetPack
- TensorRT from JetPack
- OpenCV 4.8.0 CUDA/GStreamer build installed at `/opt/opencv-4.8.0-cuda`
- Project root: `/home/wanan/camera`

Do not install PyPI `opencv-python` for this deployment. The C++ build must use the custom OpenCV CMake package:

```bash
-D OpenCV_DIR=/opt/opencv-4.8.0-cuda/lib/cmake/opencv4
```

## Current responsibility

- Load a TensorRT FP16 engine.
- Run YOLOv11-style detection inference in C++.
- Decode detections with letterbox restore and NMS.
- Read ROI/alarm settings from `configs/config.json`.
- Fallback to `configs/roi_config.json` if the main config has no ROI list.
- Apply ROI hit logic compatible with `python_prototype/roi_manager.py`.
- Apply alarm state logic compatible with `python_prototype/alarm_logic.py`.
- Write an annotated output image and JSON result without requiring a camera.

## Current layout

```text
cpp_tensorrt/
├── CMakeLists.txt
├── include/
│   ├── common.hpp
│   ├── data_types.hpp
│   ├── roi_alarm.hpp
│   └── trt_detector.hpp
├── src/
│   ├── main.cpp
│   ├── roi_alarm.cpp
│   └── trt_detector.cpp
└── README.md
```

Model artifacts stay in the project-level `weights/` directory:

```text
weights/*.pt
weights/*.onnx
weights/*.engine
```

## Model IO

The verified ONNX models use:

```text
input:  images  [1, 3, 640, 640]
output: output0 [1, 84, 8400]
```

The output format is YOLO-style:

```text
[cx, cy, w, h, class0_score, ..., class79_score]
```

For this project, C++ keeps only person detections through `num_labels=1` and class `0`.

## Build TensorRT engine

TensorRT engines must be built on the same Jetson/TensorRT version where they run.

From project root:

```bash
cd /home/wanan/camera

/usr/src/tensorrt/bin/trtexec \
  --onnx=weights/yolo11n_person.onnx \
  --saveEngine=weights/yolo11n_person.engine \
  --fp16
```

Expected artifact:

```text
weights/yolo11n_person.engine
```

## Build C++ program

From project root:

```bash
cd /home/wanan/camera

CUDACXX=/usr/local/cuda-12.6/bin/nvcc cmake -S cpp_tensorrt -B cpp_tensorrt/build \
  -D CMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc \
  -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.6 \
  -D OpenCV_DIR=/opt/opencv-4.8.0-cuda/lib/cmake/opencv4

cmake --build cpp_tensorrt/build -j$(nproc)
```

Expected binary:

```text
cpp_tensorrt/build/camera_tensorrt
```

## Run offline image inference with ROI/alarm

No camera is required.

```bash
cd /home/wanan/camera

./cpp_tensorrt/build/camera_tensorrt \
  weights/yolo11n_person.engine \
  assets/test.jpg \
  outputs/cpp_roi_alarm.jpg \
  configs/config.json \
  outputs/cpp_roi_alarm.json
```

Arguments:

```text
1. engine_path
2. input: image path, video path, camera index, RTSP URL, or GStreamer pipeline
3. output_image_path
4. config_json
5. json_output_path
6. optional --prestart
```

Input examples:

```bash
# USB camera index
./cpp_tensorrt/build/camera_tensorrt weights/yolo11n_person.engine 0 outputs/camera.jpg configs/config.json outputs/camera.json

# RTSP stream
./cpp_tensorrt/build/camera_tensorrt weights/yolo11n_person.engine rtsp://user:pass@host/stream outputs/rtsp.jpg configs/config.json outputs/rtsp.json

# CSI/GStreamer pipeline, quote the whole pipeline string
./cpp_tensorrt/build/camera_tensorrt weights/yolo11n_person.engine "nvarguscamerasrc ! video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink" outputs/csi.jpg configs/config.json outputs/csi.json

# Prestart clear-zone check
./cpp_tensorrt/build/camera_tensorrt weights/yolo11n_person.engine assets/test.jpg outputs/prestart.jpg configs/config.json outputs/prestart.json --prestart
```

Expected console output for the current `assets/test.jpg` regression case:

```text
config: configs/config.json
rois: 1
enter_frames: 3 exit_frames: 5
prestart_mode: false
detections: 2
roi_hits: 2
system_state: alarm
allow_start: false
warning: false
alarm: true
Saved result: outputs/cpp_roi_alarm.jpg
Saved JSON: outputs/cpp_roi_alarm.json
```

Expected files:

```text
outputs/cpp_roi_alarm.jpg
outputs/cpp_roi_alarm.json
```

## One-command regression

Use the regression script so future runs do not depend on remembered commands:

```bash
cd /home/wanan/camera
bash scripts/run_cpp_tensorrt_regression.sh
```

The script will:

1. Build `weights/yolo11n_person.engine` from ONNX if missing.
2. Configure CMake with `/opt/opencv-4.8.0-cuda`.
3. Build `cpp_tensorrt/build/camera_tensorrt`.
4. Run offline inference on `assets/test.jpg`.
5. Write:
   - `outputs/cpp_roi_alarm.jpg`
   - `outputs/cpp_roi_alarm.json`
6. Check the regression expectations:
   - detections = `2`
   - ROI hits = `2`
   - `system_state = alarm`
   - `alarm = true`

Environment overrides are supported:

```bash
ONNX_MODEL=weights/yolo11n_person.onnx \
ENGINE_MODEL=weights/yolo11n_person.engine \
INPUT_IMAGE=assets/test.jpg \
CONFIG_JSON=configs/config.json \
OUTPUT_IMAGE=outputs/cpp_roi_alarm.jpg \
OUTPUT_JSON=outputs/cpp_roi_alarm.json \
PRESTART_MODE=false \
bash scripts/run_cpp_tensorrt_regression.sh
```

## Python comparison command

To compare C++ output with the Python ONNX pipeline:

```bash
cd /home/wanan/camera
source .venv/bin/activate

python - <<'PY'
import json
from pathlib import Path
import sys
sys.path.insert(0, "python_prototype")
import cv2
from main import SafetyVisionSystem, load_config

config = load_config("configs/config.json")
config["model_path"] = "weights/yolo11n_person.onnx"
config["_model_path"] = str(Path("weights/yolo11n_person.onnx").resolve())
system = SafetyVisionSystem(config)
frame = cv2.imread("assets/test.jpg")
result = None
for _ in range(max(system.alarm_logic.enter_frames, system.alarm_logic.exit_frames)):
    result = system.process_frame(frame, prestart_mode=False)

python_result = result.to_dict()
cpp_result = json.loads(Path("outputs/cpp_roi_alarm.json").read_text())

print("python detections:", len(python_result["detections"]))
print("python roi_hits:", sum(len(d["roi_hits"]) for d in python_result["detections"]))
print("python system_state:", python_result["system_state"])
print("python alarm:", python_result["alarm"])
print("cpp detections:", len(cpp_result["detections"]))
print("cpp roi_hits:", sum(len(d["roi_hits"]) for d in cpp_result["detections"]))
print("cpp system_state:", cpp_result["system_state"])
print("cpp alarm:", cpp_result["alarm"])
PY
```

Expected match:

```text
detections: 2
roi_hits: 2
system_state: alarm
alarm: true
```

Small bbox and confidence differences are expected because C++ runs TensorRT FP16 while Python ONNX uses ONNXRuntime CPU.

## Notes

- This module is ready for offline image regression.
- Do not use camera input for this regression step.
- Do not add systemd/autostart here.
- The current JSON/config parser is intentionally minimal and targets the existing `configs/config.json` / `configs/roi_config.json` formats.

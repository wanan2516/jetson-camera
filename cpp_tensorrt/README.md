# C++ TensorRT Module

This directory now contains the initial C++ + TensorRT inference skeleton for later Jetson deployment.

## Current responsibility

- Load a TensorRT engine.
- Run YOLOv11-style detection inference in C++.
- Decode detections into the same core structure used by the Python prototype.
- Keep ROI and alarm logic as separate modules instead of mixing them into the detector.

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

## What is implemented now

- `TrtDetector`:
  - TensorRT engine loading
  - letterbox preprocessing
  - inference
  - YOLO-style decode + NMS
  - detection drawing
- `ROIManager` and `AlarmLogic`:
  - placeholder interface only
  - no complex business logic yet

## What still needs to be done later

- Replace placeholder ROI and alarm logic with logic aligned to `python_prototype/`
- Add config loading instead of keeping C++ inference constants in code
- Add JSON or protocol output for integration
- Validate on Jetson with the final TensorRT engine

## Model artifact location

Model artifacts stay in the project-level `weights/` directory:
- `../weights/*.pt`
- `../weights/*.onnx`
- `../weights/*.engine`

## Intended usage

Build on Jetson after preparing a matching TensorRT engine:

```bash
mkdir -p build
cd build
cmake ../cpp_tensorrt
make -j4
./camera_tensorrt ../weights/yolo11n.engine ../assets/test.jpg
```

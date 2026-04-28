#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OPENCV_DIR="${OpenCV_DIR:-/opt/opencv-4.8.0-cuda/lib/cmake/opencv4}"
CUDA_NVCC="${CUDA_NVCC:-/usr/local/cuda-12.6/bin/nvcc}"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"

ONNX_MODEL="${ONNX_MODEL:-weights/yolo11n_person.onnx}"
ENGINE_MODEL="${ENGINE_MODEL:-weights/yolo11n_person.engine}"
INPUT_IMAGE="${INPUT_IMAGE:-assets/test.jpg}"
CONFIG_JSON="${CONFIG_JSON:-configs/config.json}"
OUTPUT_IMAGE="${OUTPUT_IMAGE:-outputs/cpp_roi_alarm.jpg}"
OUTPUT_JSON="${OUTPUT_JSON:-outputs/cpp_roi_alarm.json}"
EXPECT_DETECTIONS="${EXPECT_DETECTIONS:-2}"
EXPECT_ROI_HITS="${EXPECT_ROI_HITS:-2}"
EXPECT_ALARM="${EXPECT_ALARM:-true}"
EXPECT_SYSTEM_STATE="${EXPECT_SYSTEM_STATE:-alarm}"
PRESTART_MODE="${PRESTART_MODE:-false}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [ -x "$ROOT_DIR/.venv/bin/python" ]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
fi

if [ ! -f "$ONNX_MODEL" ]; then
  echo "Missing ONNX model: $ONNX_MODEL" >&2
  exit 1
fi

if [ ! -f "$INPUT_IMAGE" ]; then
  echo "Missing input image: $INPUT_IMAGE" >&2
  exit 1
fi

if [ ! -f "$CONFIG_JSON" ]; then
  echo "Missing config: $CONFIG_JSON" >&2
  exit 1
fi

if [ ! -x "$TRTEXEC" ]; then
  echo "Missing trtexec: $TRTEXEC" >&2
  exit 1
fi

if [ ! -s "$ENGINE_MODEL" ]; then
  echo "Building TensorRT FP16 engine: $ENGINE_MODEL"
  "$TRTEXEC" --onnx="$ONNX_MODEL" --saveEngine="$ENGINE_MODEL" --fp16
fi

echo "Configuring C++ TensorRT build"
rm -rf cpp_tensorrt/build
CUDACXX="$CUDA_NVCC" cmake -S cpp_tensorrt -B cpp_tensorrt/build \
  -D CMAKE_CUDA_COMPILER="$CUDA_NVCC" \
  -D CUDA_TOOLKIT_ROOT_DIR="$(dirname "$(dirname "$CUDA_NVCC")")" \
  -D OpenCV_DIR="$OPENCV_DIR"

echo "Building C++ TensorRT binary"
cmake --build cpp_tensorrt/build -j"$(nproc)"

EXTRA_ARGS=()
if [ "$PRESTART_MODE" = "true" ]; then
  EXTRA_ARGS+=("--prestart")
fi

echo "Running C++ TensorRT ROI/alarm regression"
./cpp_tensorrt/build/camera_tensorrt \
  "$ENGINE_MODEL" \
  "$INPUT_IMAGE" \
  "$OUTPUT_IMAGE" \
  "$CONFIG_JSON" \
  "$OUTPUT_JSON" \
  "${EXTRA_ARGS[@]}"

if [ ! -s "$OUTPUT_IMAGE" ]; then
  echo "Missing output image: $OUTPUT_IMAGE" >&2
  exit 1
fi

if [ ! -s "$OUTPUT_JSON" ]; then
  echo "Missing output JSON: $OUTPUT_JSON" >&2
  exit 1
fi

export OUTPUT_JSON EXPECT_DETECTIONS EXPECT_ROI_HITS EXPECT_ALARM EXPECT_SYSTEM_STATE

"$PYTHON_BIN" - <<'PY'
import json
import os
import sys
from pathlib import Path

output_json = Path(os.environ["OUTPUT_JSON"])
result = json.loads(output_json.read_text())

detections = result.get("detections", [])
roi_hits = sum(len(d.get("roi_hits", [])) for d in detections)
alarm = result.get("alarm")
system_state = result.get("system_state")

expected_detections = int(os.environ["EXPECT_DETECTIONS"])
expected_roi_hits = int(os.environ["EXPECT_ROI_HITS"])
expected_alarm = os.environ["EXPECT_ALARM"].lower() == "true"
expected_system_state = os.environ["EXPECT_SYSTEM_STATE"]

print("Regression summary:")
print(f"  detections: {len(detections)}")
print(f"  roi_hits: {roi_hits}")
print(f"  system_state: {system_state}")
print(f"  alarm: {alarm}")

errors = []
if len(detections) != expected_detections:
    errors.append(f"detections expected {expected_detections}, got {len(detections)}")
if roi_hits != expected_roi_hits:
    errors.append(f"roi_hits expected {expected_roi_hits}, got {roi_hits}")
if alarm is not expected_alarm:
    errors.append(f"alarm expected {expected_alarm}, got {alarm}")
if system_state != expected_system_state:
    errors.append(f"system_state expected {expected_system_state}, got {system_state}")

if errors:
    for error in errors:
        print(f"  FAIL: {error}", file=sys.stderr)
    sys.exit(1)

print("  PASS")
PY

echo "C++ TensorRT ROI/alarm regression passed"
echo "Output image: $OUTPUT_IMAGE"
echo "Output JSON: $OUTPUT_JSON"

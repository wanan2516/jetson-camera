# Jetson 部署环境说明

本文档记录已经验证过的 Jetson 部署环境，方便后续复现和排查问题。

## 硬件和系统

- 设备：Jetson Orin Nano
- JetPack/L4T：R36.5.0
- Python：3.10.12
- Jetson 上的项目路径：`/home/wanan/camera`
- Python 虚拟环境：`/home/wanan/camera/.venv`

## CUDA / TensorRT / cuDNN

CUDA、TensorRT 和 cuDNN 使用 JetPack 自带版本，不要额外从 PyPI 安装单独的 CUDA toolkit。

已验证的 CUDA/PyTorch 运行信息：

```text
Torch: 2.10.0
Torch CUDA version: 12.6
Torch CUDA available: True
Torch device count: 1
Torch device name: Orin
```

TensorRT 使用 JetPack 系统版本。Python 包快照中记录的版本如下：

```text
tensorrt==10.3.0
tensorrt_dispatch==10.3.0
tensorrt_lean==10.3.0
```

TensorRT engine 必须在最终运行的 Jetson 设备上生成，不能直接跨机器复用：

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=weights/yolo11n_person.onnx \
  --saveEngine=weights/yolo11n_person.engine \
  --fp16
```

## PyTorch / torchvision

PyTorch 和 torchvision 需要安装 Jetson 兼容 wheel，来源为 Jetson AI Lab 的 JP6 CUDA 12.6 软件源。

`requirements-jetson.txt` 中记录的已验证包如下：

```text
torch @ https://pypi.jetson-ai-lab.io/jp6/cu126/+f/37d/7e156cfb4a646/torch-2.10.0-cp310-cp310-linux_aarch64.whl#sha256=37d7e156cfb4a646c4d7347597727db1529d184108f703324dfff1842cec094e
torchvision @ https://pypi.jetson-ai-lab.io/jp6/cu126/+f/1b6/357c5532db61e/torchvision-0.25.0-cp310-cp310-linux_aarch64.whl#sha256=1b6357c5532db61e9bfe7ad69f73ba73e8214010de021da703d360d2cc16c3d7
nvidia-cudss-cu12==0.7.1.6
```

注意事项：

- 不要安装 CUDA 13 或 `cu13` 版本的 PyTorch 包。
- 不要为本项目安装 `triton`。
- `nvidia-cudss-cu12` 只是为了满足 PyTorch 对 `libcudss.so.0` 的运行时加载需求。

## OpenCV

Jetson 上不要使用 PyPI 的 `opencv-python` 或 `opencv-contrib-python`。

已验证的 OpenCV 部署信息：

```text
OpenCV: 4.8.0
cv2 path: /home/wanan/camera/.venv/lib/python3.10/site-packages/cv2/__init__.py
OpenCV CUDA devices: 1
GStreamer: True
DNN CUDA: True
GAPI exists: False
```

OpenCV 安装路径：

```text
/opt/opencv-4.8.0-cuda
```

CMake 配置文件路径：

```text
/opt/opencv-4.8.0-cuda/lib/cmake/opencv4/OpenCVConfig.cmake
```

C++ 编译时需要指定：

```bash
-D OpenCV_DIR=/opt/opencv-4.8.0-cuda/lib/cmake/opencv4
```

这个 OpenCV 构建禁用了 `gapi`，目的是避免 JetPack 6.2.2 / CUDA 12.6 / cuDNN 9 组合下 Python 绑定加载失败。本项目不依赖 G-API。

## Python 包快照

Jetson 环境包快照文件为：

```text
requirements-jetson.txt
```

关键已验证包：

```text
numpy==1.26.1
onnx==1.21.0
onnxruntime==1.23.2
onnxsim==0.6.2
protobuf==5.29.6
ultralytics==8.4.41
ultralytics-thop==2.0.19
pytest==9.0.3
```

`requirements-jetson.txt` 可能包含系统 Python 包，因为 Jetson 虚拟环境可能能看到系统级 site-packages。它更适合作为环境快照，不建议作为普通电脑上的全新安装文件。

## 暂不处理的包检查提示

`pip check` 可能报告以下元数据提示：

```text
nvidia-cudss-cu12 0.7.1.6 requires cuda-toolkit, which is not installed.
ultralytics 8.4.41 requires opencv-python, which is not installed.
```

不要为了消除这些提示去安装对应包：

- `cuda-toolkit` 应来自 JetPack，而不是 PyPI。
- `opencv-python` 可能覆盖或影响当前支持 CUDA/GStreamer 的自定义 OpenCV。

## 环境验证命令

在 Jetson 上执行：

```bash
cd /home/wanan/camera
source .venv/bin/activate

python - <<'PY'
import cv2, torch
print("OpenCV:", cv2.__version__)
print("cv2 path:", cv2.__file__)
print("OpenCV CUDA devices:", cv2.cuda.getCudaEnabledDeviceCount())
print("GStreamer:", "GStreamer:                   YES" in cv2.getBuildInformation())
print("DNN CUDA:", "NVIDIA CUDA:                   YES" in cv2.getBuildInformation())
print("GAPI exists:", hasattr(cv2, "gapi"))
print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("Torch CUDA available:", torch.cuda.is_available())
print("Torch device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Torch device name:", torch.cuda.get_device_name(0))
PY
```

预期结果：

```text
OpenCV: 4.8.0
OpenCV CUDA devices: 1
GStreamer: True
DNN CUDA: True
GAPI exists: False
Torch: 2.10.0
Torch CUDA: 12.6
Torch CUDA available: True
Torch device count: 1
Torch device name: Orin
```

## C++ TensorRT 回归验证

生成 TensorRT engine 后执行：

```bash
cd /home/wanan/camera
bash scripts/run_cpp_tensorrt_regression.sh
```

预期摘要：

```text
detections: 2
roi_hits: 2
system_state: alarm
alarm: true
PASS
```

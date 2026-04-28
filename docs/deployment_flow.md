# 部署流程文档

## 概述

本文档描述从 Python 原型到 Jetson Orin Nano Super 部署的完整流程。

## 部署链路

```
YOLOv11n 权重 (.pt)
    ↓
ONNX 导出 (.onnx)
    ↓
TensorRT 转换 (.engine)
    ↓
C++ 推理程序
    ↓
阈值过滤
    ↓
ROI 区域判断
    ↓
报警逻辑输出
```

## 阶段 1: Python 原型验证（当前阶段）

### 1.1 模型训练与验证

```bash
# 使用预训练模型
python python_prototype/main.py \
  --config configs/config.json \
  --mode image \
  --input assets/test.jpg \
  --output assets/result.jpg

# 或使用独立脚本
python scripts/run_demo.py \
  --config configs/config.json \
  --mode image \
  --input assets/test.jpg \
  --output assets/result.jpg
```

### 1.2 ONNX 导出

```bash
# 方法 1: 使用独立脚本（推荐）
python scripts/export_onnx.py \
  --model weights/yolo11n.pt \
  --output weights/yolo11n_person.onnx \
  --imgsz 640 \
  --opset 12

# 方法 2: 使用 main.py
python python_prototype/main.py \
  --config configs/config.json \
  --mode export_onnx \
  --output weights/yolo11n_person.onnx
```

### 1.3 ONNX 验证

```bash
# 方法 1: 使用独立脚本（推荐）
python scripts/validate_onnx.py \
  --model weights/yolo11n_person.onnx \
  --image assets/test.jpg \
  --imgsz 640

# 方法 2: 使用 main.py
python python_prototype/main.py \
  --config configs/config.json \
  --mode validate_onnx \
  --input assets/test.jpg \
  --onnx weights/yolo11n_person.onnx
```

### 1.4 ONNX Runtime 推理验证

```bash
# 使用 ONNX Runtime 进行推理
python python_prototype/main.py \
  --config configs/config.json \
  --mode decode_onnx \
  --input assets/test.jpg \
  --output assets/onnx_result.jpg \
  --onnx weights/yolo11n_person.onnx
```

## 阶段 2: Jetson 部署准备

### 2.1 环境准备

在 Jetson Orin Nano Super 上安装以下依赖：

```bash
# JetPack 6.0+ (包含 CUDA, cuDNN, TensorRT)
# OpenCV 4.8+
# CMake 3.18+
```

### 2.2 TensorRT Engine 生成

**重要**: TensorRT engine 必须在目标设备上生成，不能跨平台使用。

```bash
# 在 Jetson 上执行
trtexec \
  --onnx=weights/yolo11n_person.onnx \
  --saveEngine=weights/yolo11n_person.engine \
  --fp16 \
  --workspace=4096

# 参数说明：
# --fp16: 使用 FP16 精度（推荐，速度快）
# --workspace: 工作空间大小（MB）
# --int8: INT8 量化（需要校准数据）
```

### 2.3 C++ 推理程序编译

```bash
CUDACXX=/usr/local/cuda-12.6/bin/nvcc cmake -S cpp_tensorrt -B cpp_tensorrt/build \
  -D CMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc \
  -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.6 \
  -D OpenCV_DIR=/opt/opencv-4.8.0-cuda/lib/cmake/opencv4

cmake --build cpp_tensorrt/build -j$(nproc)
```

### 2.4 运行 C++ 推理

```bash
./cpp_tensorrt/build/camera_tensorrt \
  weights/yolo11n_person.engine \
  assets/test.jpg \
  outputs/cpp_roi_alarm.jpg \
  configs/config.json \
  outputs/cpp_roi_alarm.json
```

输入可以是图片、视频文件、摄像头索引、RTSP URL 或 GStreamer pipeline。启动前清扫区检查追加 `--prestart`：

```bash
./cpp_tensorrt/build/camera_tensorrt \
  weights/yolo11n_person.engine \
  0 \
  outputs/camera.jpg \
  configs/config.json \
  outputs/camera.json \
  --prestart
```

## 阶段 3: 性能优化

### 3.1 FPS 测试

```bash
# Python 版本
python python_prototype/main.py \
  --config configs/config.json \
  --mode video \
  --input test_video.mp4 \
  --no-display

# C++ 版本
./cpp_tensorrt/build/camera_tensorrt \
  weights/yolo11n_person.engine \
  test_video.mp4 \
  outputs/video_last_frame.jpg \
  configs/config.json \
  outputs/video_last_frame.json
```

### 3.2 延迟测试

测试从图像输入到报警输出的端到端延迟：

```bash
# 记录每帧处理时间
python python_prototype/main.py \
  --config configs/config.json \
  --mode camera \
  --input 0 \
  --no-display
```

### 3.3 优化建议

1. **模型优化**
   - 使用 FP16 精度（速度提升 2-3x）
   - 考虑 INT8 量化（需要校准数据）
   - 固定输入尺寸（避免动态 shape）

2. **推理优化**
   - 使用 CUDA stream 异步推理
   - 批处理多帧（如果延迟允许）
   - 预分配内存，避免动态分配

3. **后处理优化**
   - NMS 使用 GPU 实现
   - ROI 判断使用 CUDA kernel
   - 减少 CPU-GPU 数据传输

## 阶段 4: 集成与测试

### 4.1 配置文件迁移

Python 和 C++ 当前统一以 `configs/config.json` 作为运行主配置，里面包含模型路径、阈值、ROI 和报警防抖参数。

以下文件仅作为参考模板或说明材料：
- `configs/model_config.yaml`: 模型配置示例
- `configs/roi_config.json`: ROI 配置示例，也可作为 C++ 备用配置
- `configs/alarm_config.yaml`: 报警配置示例

C++ 已读取 `configs/config.json`；正式部署前建议把当前轻量 JSON 解析替换为严格 JSON 解析器。

### 4.2 接口对齐

确保 Python 和 C++ 版本输出格式一致：

```json
{
  "frame_id": 1,
  "timestamp": 1234567890.123,
  "detections": [...],
  "zone_summary": {...},
  "system_state": "safe",
  "allow_start": true,
  "warning": false,
  "alarm": false
}
```

### 4.3 回归测试

使用相同的测试图像/视频，对比 Python 和 C++ 版本的输出：

```bash
# Python 版本
python python_prototype/main.py \
  --config configs/config.json \
  --mode image \
  --input assets/test.jpg \
  --output assets/python_result.jpg

# C++ 版本
./camera_tensorrt \
  ../weights/yolo11n_person.engine \
  ../assets/test.jpg \
  --output ../assets/cpp_result.jpg

# 对比结果
python scripts/compare_results.py \
  assets/python_result.jpg \
  assets/cpp_result.jpg
```

## 常见问题

### Q1: ONNX 导出失败

**原因**: Ultralytics 版本不兼容或 ONNX 版本过低

**解决**:
```bash
pip install --upgrade ultralytics onnx onnxruntime onnxsim
```

### Q2: TensorRT 转换失败

**原因**: ONNX 模型包含不支持的算子

**解决**:
```bash
# 使用 onnxsim 简化模型
python -m onnxsim weights/yolo11n_person.onnx weights/yolo11n_simplified.onnx

# 或降低 opset 版本
python scripts/export_onnx.py --model weights/yolo11n.pt --opset 11
```

### Q3: C++ 推理结果与 Python 不一致

**原因**: 预处理或后处理逻辑不一致

**解决**:
1. 检查 letterbox 实现是否一致
2. 检查归一化方式（/255.0）
3. 检查 NMS 阈值是否一致
4. 检查坐标还原逻辑

### Q4: FPS 不达标

**原因**: 推理或后处理耗时过长

**解决**:
1. 使用 FP16 或 INT8 精度
2. 减小输入尺寸（640 -> 480）
3. 优化 NMS 和 ROI 判断
4. 使用多线程或异步推理

## 性能指标参考

### Jetson Orin Nano Super (8GB)

| 配置 | 输入尺寸 | 精度 | FPS | 延迟 |
|------|---------|------|-----|------|
| YOLOv11n | 640x640 | FP32 | ~15 | ~65ms |
| YOLOv11n | 640x640 | FP16 | ~30 | ~33ms |
| YOLOv11n | 480x480 | FP16 | ~45 | ~22ms |
| YOLOv11n | 640x640 | INT8 | ~50 | ~20ms |

注：以上数据仅供参考，实际性能取决于具体场景和优化程度。

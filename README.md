# Camera Safety Vision

工业安全视觉人员检测系统，基于 YOLOv11n 实现人员检测、区域清扫、危险靠近、入侵检测等功能。

## 项目概述

本项目是一个工业安全视觉系统，用于检测人员并根据 ROI 区域触发相应的报警逻辑。核心功能包括：

- **人员检测**: 使用 YOLOv11n 模型检测画面中的人员
- **区域清扫**: 启动前确保清扫区域无人
- **危险靠近**: 人员接近危险区域时发出警告
- **入侵检测**: 人员进入禁止区域时触发报警

## 项目结构

```
camera/
├── README.md                   # 项目说明
├── requirements.txt            # Python 依赖
├── requirements-jetson.txt     # Jetson 环境包快照，不建议直接用于全新安装
├── configs/                    # 配置文件目录
│   ├── config.json            # 唯一运行主配置
│   ├── model_config.yaml      # 参考模板，不作为默认运行入口
│   ├── roi_config.json        # ROI 参考模板 / C++ fallback
│   └── alarm_config.yaml      # 报警参数参考模板
├── python_prototype/           # Python 原型代码
│   ├── main.py                # 主程序入口
│   ├── detector.py            # 检测器（PyTorch + ONNX Runtime）
│   ├── data_models.py         # 数据模型定义
│   ├── roi_manager.py         # ROI 区域判断
│   ├── alarm_logic.py         # 报警逻辑
│   ├── visualizer.py          # 可视化
│   ├── onnx_validator.py      # ONNX 验证
│   └── logger.py              # 日志工具
├── scripts/                    # 独立脚本
│   ├── export_onnx.py         # ONNX 导出脚本
│   ├── validate_onnx.py       # ONNX 验证脚本
│   ├── run_cpp_tensorrt_regression.sh # C++ TensorRT 回归脚本
│   └── run_demo.py            # 演示脚本
├── tests/                      # 测试用例
│   ├── test_detector.py       # 检测器测试
│   ├── test_roi.py            # ROI 测试
│   └── test_alarm.py          # 报警逻辑测试
├── cpp_tensorrt/               # C++ TensorRT 部署模块
│   ├── CMakeLists.txt
│   ├── include/
│   ├── src/
│   └── README.md
├── weights/                    # 模型权重文件
│   ├── yolo11n.pt             # PyTorch 权重
│   ├── yolo11n_person.onnx           # ONNX 模型
│   └── *.engine               # TensorRT engine（Jetson 上生成）
├── assets/                     # 测试图像和结果
│   └── test.jpg
└── docs/                       # 文档
    ├── SAFETY_VISION_PLAN.md  # 系统设计文档
    ├── ROI_INTERFACE_BRIEF.md # ROI 接口说明
    ├── deployment_flow.md     # 部署流程
    ├── jetson_environment.md  # Jetson CUDA/PyTorch/OpenCV 环境说明
    └── model_io.md            # 模型输入输出规范
```

## 快速开始

### 1. 环境准备

```bash
# 创建 Conda 环境（推荐）
conda create -n camera-yolo python=3.10
conda activate camera-yolo

# 安装依赖
pip install -r requirements.txt
```

### 2. 下载模型

```bash
# 下载 YOLOv11n 预训练模型
# 方法 1: 自动下载（首次运行时）
python python_prototype/main.py --config configs/config.json --mode image --input assets/test.jpg

# 方法 2: 手动下载
# 访问 https://github.com/ultralytics/assets/releases
# 下载 yolo11n.pt 并放到 weights/ 目录
```

说明：运行时统一使用 `configs/config.json`。其它拆分配置文件只作为调参说明或模板，不是默认运行入口。

### 3. 运行演示

#### 图像检测

```bash
# 使用 main.py
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

#### 视频检测

```bash
python python_prototype/main.py \
  --config configs/config.json \
  --mode video \
  --input test_video.mp4 \
  --output result_video.mp4
```

#### 摄像头实时检测

```bash
python python_prototype/main.py \
  --config configs/config.json \
  --mode camera \
  --input 0
```

#### 启动前清扫区检查

```bash
python python_prototype/main.py \
  --config configs/config.json \
  --mode image \
  --input assets/test.jpg \
  --prestart
```

## ONNX 导出与验证

### 导出 ONNX

```bash
# 使用独立脚本（推荐）
python scripts/export_onnx.py \
  --model weights/yolo11n.pt \
  --output weights/yolo11n_person.onnx \
  --imgsz 640 \
  --opset 12

# 或使用 main.py
python python_prototype/main.py \
  --config configs/config.json \
  --mode export_onnx \
  --output weights/yolo11n_person.onnx
```

### 验证 ONNX

```bash
# 使用独立脚本（推荐）
python scripts/validate_onnx.py \
  --model weights/yolo11n_person.onnx \
  --image assets/test.jpg \
  --imgsz 640

# 或使用 main.py
python python_prototype/main.py \
  --config configs/config.json \
  --mode validate_onnx \
  --input assets/test.jpg \
  --onnx weights/yolo11n_person.onnx
```

### ONNX Runtime 推理

```bash
python python_prototype/main.py \
  --config configs/config.json \
  --mode decode_onnx \
  --input assets/test.jpg \
  --output assets/onnx_result.jpg \
  --onnx weights/yolo11n_person.onnx
```

## 测试

```bash
# 运行所有测试
python tests/test_detector.py
python tests/test_roi.py
python tests/test_alarm.py
```

## 配置说明

### 主配置 (configs/config.json)

Python 和 C++ 默认都读取 `configs/config.json`。检测阈值、ROI 和报警防抖参数应优先在这个文件里修改。

参考模板：
- `configs/model_config.yaml`: 模型参数说明模板
- `configs/roi_config.json`: ROI 结构说明模板，也可作为 C++ 在主配置缺少 `rois` 时的 fallback
- `configs/alarm_config.yaml`: 报警参数说明模板

### 模型配置示例

```yaml
model_path: "weights/yolo11n.pt"
imgsz: 640
device: null  # null=自动选择, "cpu", "cuda", "mps"
person_class_ids: [0]
thresholds:
  conf_thres: 0.35
  iou_thres: 0.45
```

### ROI 配置示例

```json
{
  "rois": [
    {
      "roi_id": "zone_1",
      "name": "危险区域",
      "enabled": true,
      "roi_type": "forbidden_zone",
      "judge_method": "foot_point",
      "coordinate_mode": "normalized",
      "polygon": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
      "overlap_thres": 0.2
    }
  ]
}
```

**ROI 类型**:
- `clear_zone`: 清扫区，启动前必须无人
- `warning_zone`: 预警区，有人时发出警告
- `forbidden_zone`: 禁止区，有人时触发报警

**判断方法**:
- `foot_point`: 检测框底部中心点（推荐用于地面区域）
- `center_point`: 检测框中心点（简单快速）
- `overlap`: 检测框与 ROI 重叠率（适合俯视或不规则区域）

### 报警配置示例

```yaml
alarm:
  enter_frames: 3  # 连续 N 帧检测到目标才触发报警
  exit_frames: 5   # 连续 M 帧未检测到目标才解除报警
```

## Jetson 部署

详细部署流程请参考 [docs/deployment_flow.md](docs/deployment_flow.md)。

`requirements-jetson.txt` 是当前 Jetson 环境包快照，主要用于记录环境；新设备安装时优先参考 [docs/jetson_environment.md](docs/jetson_environment.md)。

### 简要步骤

1. **导出 ONNX** (在 PC 上)
   ```bash
   python scripts/export_onnx.py --model weights/yolo11n.pt --output weights/yolo11n_person.onnx
   ```

2. **生成 TensorRT Engine** (在 Jetson 上)
   ```bash
   trtexec --onnx=weights/yolo11n_person.onnx --saveEngine=weights/yolo11n_person.engine --fp16
   ```

3. **编译 C++ 程序** (在 Jetson 上)
   ```bash
   CUDACXX=/usr/local/cuda-12.6/bin/nvcc cmake -S cpp_tensorrt -B cpp_tensorrt/build \
     -D CMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc \
     -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.6 \
     -D OpenCV_DIR=/opt/opencv-4.8.0-cuda/lib/cmake/opencv4

   cmake --build cpp_tensorrt/build -j$(nproc)
   ```

4. **运行推理**
   ```bash
   ./cpp_tensorrt/build/camera_tensorrt \
     weights/yolo11n_person.engine \
     assets/test.jpg \
     outputs/cpp_roi_alarm.jpg \
     configs/config.json \
     outputs/cpp_roi_alarm.json
   ```

   C++ 输入参数支持图片、视频文件、摄像头索引、RTSP URL 和 GStreamer pipeline；启动前清扫区检查可追加 `--prestart`：

   ```bash
   ./cpp_tensorrt/build/camera_tensorrt weights/yolo11n_person.engine 0 outputs/camera.jpg configs/config.json outputs/camera.json --prestart
   ```

## 性能指标

### PC (Apple M3)

| 后端 | 输入尺寸 | FPS | 延迟 |
|------|---------|-----|------|
| PyTorch | 640x640 | ~30 | ~33ms |
| ONNX Runtime | 640x640 | ~40 | ~25ms |

### Jetson Orin Nano Super (预估)

| 精度 | 输入尺寸 | FPS | 延迟 |
|------|---------|-----|------|
| FP32 | 640x640 | ~15 | ~65ms |
| FP16 | 640x640 | ~30 | ~33ms |
| INT8 | 640x640 | ~50 | ~20ms |

## 文档

- [系统设计文档](docs/SAFETY_VISION_PLAN.md)
- [ROI 接口说明](docs/ROI_INTERFACE_BRIEF.md)
- [部署流程](docs/deployment_flow.md)
- [模型输入输出规范](docs/model_io.md)


## 常见问题

### Q1: 如何修改检测阈值？

编辑 `configs/config.json`：

```yaml
thresholds:
  conf_thres: 0.35  # 提高此值减少误检，降低此值减少漏检
  iou_thres: 0.45   # NMS 阈值
```

### Q2: 如何添加新的 ROI 区域？

编辑 `configs/config.json` 里的 `rois`，添加新的 ROI 配置：

```json
{
  "roi_id": "zone_2",
  "name": "新区域",
  "enabled": true,
  "roi_type": "warning_zone",
  "judge_method": "foot_point",
  "coordinate_mode": "normalized",
  "polygon": [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],
  "overlap_thres": 0.2
}
```

### Q3: 如何调整防抖参数？

编辑 `configs/config.json` 里的 `alarm`：

```yaml
alarm:
  enter_frames: 3  # 增加此值减少误报
  exit_frames: 5   # 增加此值让报警持续更久
```

### Q4: 如何使用自己训练的模型？

1. 训练模型：
   ```bash
   yolo detect train model=yolo11n.pt data=person.yaml epochs=80
   ```

2. 更新配置文件中的 `model_path`：
   ```yaml
   model_path: "runs/detect/train/weights/best.pt"
   ```

## 许可证

本项目仅供学习和研究使用。

## 联系方式

如有问题，请提交 Issue。

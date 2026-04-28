# YOLOv11n 工业安全人员检测方案

## 1. 系统范围

本原型实现以下能力：

- 基于 YOLOv11n 预训练权重进行人员检测。
- 支持可配置的置信度阈值和 NMS IoU 阈值。
- 支持多个多边形 ROI 区域，用于清扫区、预警区和禁止区判断。
- 使用进入帧数和退出帧数做时间平滑，降低误报警。
- 支持 ONNX 导出，为后续 Jetson TensorRT 部署做准备。

当前可部署类别只保留 `person`。本项目不追求复杂算法创新，目标是建立一个稳定、可解释、可调参、可迁移到 Jetson 的工程基线。

## 2. 模型流程

模型策略是先使用 YOLOv11n 预训练权重，再根据现场数据微调。

推荐流程：

1. 采集目标摄像头下的真实图像，覆盖正常光照、低照度、逆光、遮挡、防护服、远距离等情况。
2. 标注类别只保留一个：`person`。
3. 训练集和验证集按场景划分，不要用相邻视频帧随机拆分，避免验证集泄漏。
4. 从 `yolo11n.pt` 开始微调，不从零训练。
5. 验证时同时关注 precision、recall、mAP50、误检样本、漏检样本，以及按 ROI 业务逻辑回放的报警结果。

训练和验证命令示例：

```bash
yolo detect train model=yolo11n.pt data=person.yaml imgsz=640 epochs=80 batch=16 workers=4
yolo detect val model=runs/detect/train/weights/best.pt data=person.yaml imgsz=640 conf=0.35 iou=0.45
```

本项目图像推理命令示例：

```bash
python python_prototype/main.py --config configs/config.json --mode image --input assets/test.jpg --output assets/result.jpg --prestart --no-display
```

## 3. ONNX 与 TensorRT 路径

本项目提供 ONNX 导出模式：

```bash
python python_prototype/main.py --config configs/config.json --mode export_onnx --output weights/yolo11n_person.onnx
```

等价的 Ultralytics 命令示例：

```bash
yolo export model=runs/detect/train/weights/best.pt format=onnx imgsz=640 opset=12 simplify=True
```

Jetson 上生成 TensorRT engine 的命令示例：

```bash
trtexec --onnx=weights/yolo11n_person.onnx --saveEngine=weights/yolo11n_person_fp16.engine --fp16
```

部署注意事项：

- 导出和运行时尽量保持固定 `imgsz`，除非明确需要动态 shape。
- Jetson 上优先使用 FP16，通常能在精度影响较小的情况下提升实时性。
- INT8 量化需要代表性现场数据做校准，不建议一开始就做。
- 后处理阈值应保留在配置文件中，不能写死进 engine。

## 4. ROI 判断方式

当前支持三种判断方式：

- `foot_point`：判断人框底部中心点是否落入 ROI。
- `center_point`：判断人框中心点是否落入 ROI。
- `overlap`：判断人框与 ROI 的重叠比例是否超过 `overlap_thres`。

方法对比：

- `foot_point` 适合地面危险区域。摄像头固定且 ROI 画在地面上时，它最符合“人是否站进区域”的定义。
- `center_point` 最简单，计算快，但可能漏掉脚已经进入区域而身体中心仍在区域外的情况。
- `overlap` 适合俯视、遮挡、非地面区域或只要身体一部分进入就要报警的场景，但需要现场调重叠率阈值。

默认建议：

- 如果能看到脚，且 ROI 表示地面危险区域，默认使用 `foot_point`。
- 如果看不到脚、遮挡多、俯视明显，或需要“身体任意部分进入即报警”，使用 `overlap`。

## 5. 报警逻辑

当前逻辑如下：

- 区域连续 `enter_frames` 帧检测到人员后，才进入稳定触发状态。
- 区域连续 `exit_frames` 帧未检测到人员后，才解除稳定触发状态。
- 启动前清扫模式下，只有清扫区连续确认无人后，才允许启动。

业务输出：

- 清扫区：`allow_start=True` 表示允许启动；清扫区有人时禁止启动。
- 预警区：稳定检测到人员后输出 `warning=True`。
- 禁止区：稳定检测到人员后输出 `alarm=True`。

现场初始参数建议：

- `conf_thres=0.35`，`iou_thres=0.45`。
- 25 FPS 左右时可先用 `enter_frames=3`、`exit_frames=5`。
- 光照闪烁、设备振动、误检较多时，适当增大 `enter_frames`。
- 希望报警保持更久时，适当增大 `exit_frames`。

## 6. 当前代码入口

- `python_prototype/main.py`：命令行入口和系统调度。
- `python_prototype/detector.py`：YOLOv11n 推理、ONNX Runtime 推理和 ONNX 导出。
- `python_prototype/roi_manager.py`：ROI 点判定和重叠率判定。
- `python_prototype/alarm_logic.py`：报警防抖和业务状态机。
- `configs/config.json`：模型路径、阈值、设备、输入尺寸、报警参数和 ROI 定义。

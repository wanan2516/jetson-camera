# 模型输入输出规范

## 概述

本文档详细说明 YOLOv11n 模型的输入输出格式，确保 Python 和 C++ 实现的一致性。

## 模型信息

- **模型**: YOLOv11n
- **任务**: 目标检测（Object Detection）
- **类别**: person (COCO class_id=0)
- **框架**: Ultralytics YOLO

## 输入规范

### 输入尺寸

- **固定尺寸**: 640x640 (默认)
- **可选尺寸**: 480x480, 320x320 (需重新导出 ONNX)
- **格式**: NCHW (Batch, Channel, Height, Width)

### 预处理流程

#### 1. Letterbox Resize

保持宽高比的缩放，不足部分用灰色填充：

```python
def letterbox(image, target_size=640):
    """
    Letterbox resize with aspect ratio preservation
    
    Args:
        image: BGR image (H, W, C)
        target_size: target size (default 640)
    
    Returns:
        resized: letterboxed image (target_size, target_size, 3)
        scale: resize scale factor
        pad_x: left padding
        pad_y: top padding
    """
    h, w = image.shape[:2]
    scale = min(target_size / h, target_size / w)
    
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    
    # Resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create canvas with gray padding (114, 114, 114)
    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    
    # Calculate padding
    pad_x = (target_size - new_w) / 2
    pad_y = (target_size - new_h) / 2
    left = int(round(pad_x - 0.1))
    top = int(round(pad_y - 0.1))
    
    # Paste resized image
    canvas[top:top+new_h, left:left+new_w] = resized
    
    return canvas, scale, float(left), float(top)
```

#### 2. 颜色空间转换

```python
# BGR -> RGB
rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
```

#### 3. 归一化

```python
# [0, 255] -> [0.0, 1.0]
normalized = rgb.astype(np.float32) / 255.0
```

#### 4. 转置与扩维

```python
# HWC -> CHW
transposed = np.transpose(normalized, (2, 0, 1))

# CHW -> NCHW
batched = np.expand_dims(transposed, axis=0)
```

### 输入张量

- **名称**: `images` (可能因导出参数而异)
- **形状**: `[1, 3, 640, 640]`
- **数据类型**: `float32`
- **数值范围**: `[0.0, 1.0]`
- **通道顺序**: RGB

## 输出规范

### 输出张量

- **名称**: `output0` (可能因导出参数而异)
- **形状**: `[1, 84, 8400]` 或 `[1, 8400, 84]` (取决于导出参数)
- **数据类型**: `float32`

### 输出格式说明

#### 维度解释

- **Batch**: 1 (单张图像)
- **Anchors**: 8400 (检测框数量，来自 80x80 + 40x40 + 20x20 特征图)
- **Attributes**: 84 (4 bbox + 80 classes)

#### Bbox 格式

前 4 个值为边界框坐标（相对于 640x640 输入）：

```
[cx, cy, w, h]
```

- `cx`: 中心点 x 坐标
- `cy`: 中心点 y 坐标
- `w`: 宽度
- `h`: 高度

#### Class Scores

后 80 个值为 COCO 80 类的置信度分数：

```
[class_0_score, class_1_score, ..., class_79_score]
```

对于 person 检测，只需要 `class_0_score`。

### 后处理流程

#### 1. 转置（如果需要）

```python
# 如果输出形状是 [1, 84, 8400]，转置为 [1, 8400, 84]
if predictions.shape[1] < predictions.shape[2]:
    predictions = predictions.transpose(0, 2, 1)

# 去除 batch 维度
predictions = predictions.squeeze(0)  # [8400, 84]
```

#### 2. 提取 bbox 和 scores

```python
boxes = predictions[:, :4]  # [8400, 4] - cx, cy, w, h
class_scores = predictions[:, 4:]  # [8400, 80]
```

#### 3. 过滤 person 类别

```python
person_class_id = 0
person_scores = class_scores[:, person_class_id]  # [8400]

# 置信度过滤
conf_mask = person_scores >= conf_thres  # 例如 0.35
boxes = boxes[conf_mask]
scores = person_scores[conf_mask]
```

#### 4. 坐标转换

```python
# cx, cy, w, h -> x1, y1, x2, y2
x1 = boxes[:, 0] - boxes[:, 2] / 2
y1 = boxes[:, 1] - boxes[:, 3] / 2
x2 = boxes[:, 0] + boxes[:, 2] / 2
y2 = boxes[:, 1] + boxes[:, 3] / 2
boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
```

#### 5. 坐标还原

```python
# 去除 padding
boxes_xyxy[:, [0, 2]] -= pad_x
boxes_xyxy[:, [1, 3]] -= pad_y

# 缩放到原图尺寸
boxes_xyxy /= scale

# 裁剪到图像边界
boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, original_width)
boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, original_height)
```

#### 6. NMS (Non-Maximum Suppression)

```python
def nms(boxes, scores, iou_thres=0.45):
    """
    Non-Maximum Suppression
    
    Args:
        boxes: [N, 4] - x1, y1, x2, y2
        scores: [N] - confidence scores
        iou_thres: IoU threshold (default 0.45)
    
    Returns:
        keep: indices of boxes to keep
    """
    order = scores.argsort()[::-1]
    keep = []
    
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        if order.size == 1:
            break
        
        # Calculate IoU
        ious = box_iou(boxes[i], boxes[order[1:]])
        
        # Keep boxes with IoU < threshold
        order = order[1:][ious < iou_thres]
    
    return np.array(keep)
```

## 检测结果格式

### Detection 对象

```python
@dataclass
class Detection:
    class_id: int           # 类别 ID (0 for person)
    class_name: str         # 类别名称 ("person")
    confidence: float       # 置信度 [0.0, 1.0]
    bbox: List[float]       # [x1, y1, x2, y2] 原图坐标
    center: Tuple[int, int] # 中心点 (cx, cy)
    foot_point: Tuple[int, int]  # 底部中心点 (cx, y2)
    roi_hits: List[Dict]    # ROI 命中信息
    track_id: Optional[int] # 跟踪 ID (可选)
```

### JSON 输出格式

```json
{
  "frame_id": 1,
  "timestamp": 1234567890.123,
  "detections": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.85,
      "bbox": [100.5, 200.3, 300.7, 450.2],
      "center": [200, 325],
      "foot_point": [200, 450],
      "roi_hits": [
        {
          "roi_id": "zone_1",
          "roi_name": "危险区域",
          "roi_type": "forbidden_zone",
          "inside": true,
          "method": "foot_point"
        }
      ],
      "track_id": null
    }
  ],
  "zone_summary": {
    "zone_1": {
      "roi_id": "zone_1",
      "roi_name": "危险区域",
      "roi_type": "forbidden_zone",
      "person_count": 1,
      "raw_active": true,
      "stable_active": true,
      "enter_counter": 3,
      "exit_counter": 0
    }
  },
  "system_state": "alarm",
  "allow_start": false,
  "warning": false,
  "alarm": true
}
```

## 配置参数

### 检测阈值

```yaml
thresholds:
  conf_thres: 0.35  # 置信度阈值，低于此值的检测框被过滤
  iou_thres: 0.45   # NMS IoU 阈值，重叠度高于此值的框被抑制
```

### 推荐配置

| 场景 | conf_thres | iou_thres | 说明 |
|------|-----------|-----------|------|
| 高精度 | 0.50 | 0.45 | 减少误检，可能漏检 |
| 平衡 | 0.35 | 0.45 | 默认配置 |
| 高召回 | 0.25 | 0.50 | 减少漏检，可能误检 |

## C++ 实现注意事项

### 1. 数据类型

- 使用 `float` 而非 `double`
- 使用 `std::vector<float>` 存储张量数据

### 2. 内存管理

- 预分配内存，避免动态分配
- 使用 CUDA 统一内存或 pinned memory

### 3. 坐标精度

- 保持与 Python 一致的四舍五入规则
- 使用 `std::round()` 而非 `(int)`

### 4. NMS 实现

- 推荐使用 TensorRT 内置 NMS plugin
- 或使用 CUDA kernel 实现

## 验证方法

### 1. 输入验证

```python
# 保存预处理后的输入
np.save("input_tensor.npy", input_tensor)

# C++ 中加载并对比
# input_cpp = load_npy("input_tensor.npy")
# assert(input_cpp == input_python)
```

### 2. 输出验证

```python
# 保存原始输出
np.save("raw_output.npy", raw_output)

# 对比 Python 和 C++ 的原始输出
# 允许小的浮点误差（< 1e-5）
```

### 3. 端到端验证

```bash
# 使用相同图像测试
python python_prototype/main.py --mode image --input test.jpg
./camera_tensorrt test.jpg

# 对比检测框数量和坐标
# 允许小的坐标差异（< 2 pixels）
```

## 常见问题

### Q1: 输出形状不一致

**原因**: ONNX 导出时的 transpose 参数不同

**解决**: 在后处理中检查并转置

### Q2: 坐标还原不准确

**原因**: letterbox padding 计算不一致

**解决**: 使用相同的 `round(x - 0.1)` 规则

### Q3: NMS 结果不同

**原因**: IoU 计算或排序逻辑不同

**解决**: 使用相同的 IoU 计算公式和稳定排序

### Q4: 置信度分数不同

**原因**: 浮点精度或 softmax 实现不同

**解决**: 允许小的浮点误差（< 1e-4）

## 参考资料

- [Ultralytics YOLOv11 文档](https://docs.ultralytics.com/)
- [ONNX Runtime 文档](https://onnxruntime.ai/docs/)
- [TensorRT 文档](https://docs.nvidia.com/deeplearning/tensorrt/)

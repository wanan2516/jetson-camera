# 代码审查与优化总结

## 一、当前仓库主要问题

### 1. 目录结构问题
- 缺少独立的 `scripts/` 目录存放可执行脚本
- 缺少 `configs/` 目录统一管理配置文件
- 缺少 `tests/` 目录存放测试用例
- 配置文件混在 `python_prototype/` 目录中

### 2. 代码质量问题
- `main.py` 功能过多，包含 ONNX 导出、验证等功能
- 缺少日志输出，不便于调试和问题追踪
- 缺少测试用例，不便于验证功能正确性
- 缺少异常处理的统一机制

### 3. 文档问题
- 缺少详细的部署流程文档
- 缺少模型输入输出规范文档
- README 不够详细，缺少快速开始指南

### 4. 配置管理问题
- 只有一个 `config.json`，不够模块化
- 缺少配置文件的说明和示例

## 二、优化方案

### 1. 目录结构重构

**新增目录**:
- `configs/`: 统一管理配置文件
- `scripts/`: 存放独立的可执行脚本
- `tests/`: 存放测试用例

**保留目录**:
- `python_prototype/`: 保持原有代码结构（用户已习惯）
- `cpp_tensorrt/`: C++ TensorRT 部署模块
- `weights/`: 模型权重文件
- `assets/`: 测试图像和结果
- `docs/`: 文档

### 2. 配置文件优化

**新增配置文件**:
- `configs/model_config.yaml`: 模型配置（模型路径、输入尺寸、阈值等）
- `configs/roi_config.json`: ROI 配置（区域定义、判断方法等）
- `configs/alarm_config.yaml`: 报警配置（防抖参数、报警逻辑说明）
- `configs/config.json`: 主配置文件（保持兼容性）

### 3. 独立脚本创建

**新增脚本**:
- `scripts/export_onnx.py`: ONNX 导出脚本
- `scripts/validate_onnx.py`: ONNX 验证脚本
- `scripts/run_demo.py`: 演示脚本

**优点**:
- 功能独立，便于单独运行
- 减少 `main.py` 的复杂度
- 便于集成到 CI/CD 流程

### 4. 日志系统添加

**新增模块**:
- `python_prototype/logger.py`: 统一的日志工具

**改进点**:
- 在关键模块添加日志输出（detector.py, main.py）
- 支持控制台和文件输出
- 便于调试和问题追踪

### 5. 测试用例创建

**新增测试**:
- `tests/test_detector.py`: 检测器测试
- `tests/test_roi.py`: ROI 管理器测试
- `tests/test_alarm.py`: 报警逻辑测试

**覆盖范围**:
- 检测结果构建
- ROI 点判断、重叠率计算
- 报警防抖逻辑、清扫区逻辑、预警区逻辑

### 6. 文档完善

**新增文档**:
- `docs/deployment_flow.md`: 详细的部署流程文档
- `docs/model_io.md`: 模型输入输出规范文档

**更新文档**:
- `README.md`: 完善快速开始、配置说明、常见问题等

## 三、变更文件清单

### 新增文件

#### 配置文件
- `configs/model_config.yaml`
- `configs/roi_config.json`
- `configs/alarm_config.yaml`
- `configs/config.json`

#### 脚本文件
- `scripts/export_onnx.py`
- `scripts/validate_onnx.py`
- `scripts/run_demo.py`

#### 测试文件
- `tests/test_detector.py`
- `tests/test_roi.py`
- `tests/test_alarm.py`

#### 文档文件
- `docs/deployment_flow.md`
- `docs/model_io.md`

#### 其他文件
- `python_prototype/logger.py`
- `requirements.txt` (项目根目录)

### 修改文件

- `python_prototype/detector.py`: 添加日志输出
- `python_prototype/main.py`: 添加日志输出和异常处理
- `README.md`: 完善文档

### 保持不变的文件

- `python_prototype/data_models.py`
- `python_prototype/roi_manager.py`
- `python_prototype/alarm_logic.py`
- `python_prototype/visualizer.py`
- `python_prototype/onnx_validator.py`
- `python_prototype/requirements.txt`
- `cpp_tensorrt/*` (所有 C++ 代码)

## 四、运行命令

### 1. 安装依赖

```bash
# 创建 Conda 环境
conda create -n camera-yolo python=3.10
conda activate camera-yolo

# 安装依赖
pip install -r requirements.txt
```

### 2. 图像检测

```bash
# 使用 main.py
python python_prototype/main.py \
  --config configs/config.json \
  --mode image \
  --input assets/test.jpg \
  --output assets/result.jpg

# 使用独立脚本
python scripts/run_demo.py \
  --config configs/config.json \
  --mode image \
  --input assets/test.jpg \
  --output assets/result.jpg
```

### 3. 视频检测

```bash
python python_prototype/main.py \
  --config configs/config.json \
  --mode video \
  --input test_video.mp4 \
  --output result_video.mp4
```

### 4. 摄像头检测

```bash
python python_prototype/main.py \
  --config configs/config.json \
  --mode camera \
  --input 0
```

### 5. 导出 ONNX

```bash
# 使用独立脚本（推荐）
python scripts/export_onnx.py \
  --model weights/yolo11n.pt \
  --output weights/yolo11n.onnx \
  --imgsz 640 \
  --opset 12

# 使用 main.py
python python_prototype/main.py \
  --config configs/config.json \
  --mode export_onnx \
  --output weights/yolo11n.onnx
```

### 6. 验证 ONNX

```bash
# 使用独立脚本（推荐）
python scripts/validate_onnx.py \
  --model weights/yolo11n.onnx \
  --image assets/test.jpg \
  --imgsz 640

# 使用 main.py
python python_prototype/main.py \
  --config configs/config.json \
  --mode validate_onnx \
  --input assets/test.jpg \
  --onnx weights/yolo11n.onnx
```

### 7. 运行 ROI/报警 Demo

```bash
# 启动前清扫区检查
python python_prototype/main.py \
  --config configs/config.json \
  --mode image \
  --input assets/test.jpg \
  --prestart
```

### 8. 运行测试

```bash
# 运行所有测试
python tests/test_detector.py
python tests/test_roi.py
python tests/test_alarm.py
```

## 五、后续 TODO

### Jetson 到手后

#### 1. 生成 TensorRT Engine

```bash
# 在 Jetson 上执行
trtexec \
  --onnx=weights/yolo11n.onnx \
  --saveEngine=weights/yolo11n_fp16.engine \
  --fp16 \
  --workspace=4096
```

**注意事项**:
- TensorRT engine 必须在目标设备上生成
- 不能跨平台使用（PC 生成的 engine 不能在 Jetson 上用）
- 推荐使用 FP16 精度（速度快，精度损失小）

#### 2. 迁移到 C++

**需要完成的工作**:

1. **配置文件加载**
   - 实现 JSON/YAML 配置文件解析
   - 支持与 Python 版本相同的配置格式

2. **预处理逻辑**
   - 实现 letterbox resize
   - 实现 BGR->RGB 转换
   - 实现归一化和 NCHW 转换

3. **后处理逻辑**
   - 实现坐标转换（cx,cy,w,h -> x1,y1,x2,y2）
   - 实现坐标还原（去除 padding，缩放到原图）
   - 实现 NMS（推荐使用 TensorRT plugin）

4. **ROI 判断逻辑**
   - 实现点在多边形内判断
   - 实现重叠率计算
   - 实现归一化坐标转换

5. **报警逻辑**
   - 实现防抖状态机
   - 实现清扫区、预警区、禁止区逻辑
   - 实现输出格式（JSON 或 Protocol Buffer）

**参考代码**:
- `python_prototype/detector.py`: 预处理和后处理逻辑
- `python_prototype/roi_manager.py`: ROI 判断逻辑
- `python_prototype/alarm_logic.py`: 报警逻辑

#### 3. 测试 FPS 和延迟

**测试方法**:

1. **单帧延迟测试**
   ```cpp
   auto start = std::chrono::high_resolution_clock::now();
   // 推理 + 后处理 + ROI + 报警
   auto end = std::chrono::high_resolution_clock::now();
   auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
   ```

2. **FPS 测试**
   ```cpp
   int frame_count = 0;
   auto start = std::chrono::high_resolution_clock::now();
   while (frame_count < 100) {
       // 处理一帧
       frame_count++;
   }
   auto end = std::chrono::high_resolution_clock::now();
   auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
   float fps = frame_count / duration;
   ```

3. **性能分析**
   - 使用 `nvprof` 或 `nsys` 分析性能瓶颈
   - 分别测试推理、后处理、ROI、报警各部分耗时
   - 优化最耗时的部分

**性能目标**:
- FP16 精度: ≥30 FPS (640x640)
- 端到端延迟: ≤50ms
- CPU 占用: ≤50%
- GPU 占用: ≤80%

## 六、优化亮点

### 1. 保持简单

- 没有引入复杂框架
- 没有过度设计
- 保持代码清晰易懂

### 2. 模块化

- 配置文件分离，便于管理
- 脚本独立，便于单独运行
- 测试用例完善，便于验证

### 3. 工程化

- 添加日志输出，便于调试
- 添加异常处理，提高健壮性
- 添加测试用例，保证质量

### 4. 文档完善

- 详细的部署流程文档
- 清晰的模型输入输出规范
- 完善的 README 和快速开始指南

### 5. 便于迁移

- Python 代码结构清晰，便于迁移到 C++
- 预处理、后处理、ROI、报警逻辑独立
- 配置文件格式清晰，C++ 可直接使用

## 七、注意事项

### 1. 兼容性

- 保持与原有代码的兼容性
- `python_prototype/config.json` 仍然可用
- `python_prototype/main.py` 所有功能保持不变

### 2. 最小化改动

- 只修改必要的部分
- 不改变核心逻辑
- 不引入新的依赖（除了 pyyaml）

### 3. 代码风格

- 保持与原有代码一致的风格
- 遵循 PEP 8 规范
- 添加必要的注释和文档字符串

### 4. 测试验证

- 所有新增功能都有测试用例
- 确保原有功能不受影响
- 验证 ONNX 导出和推理的正确性

## 八、总结

本次代码审查和优化主要聚焦于：

1. **目录结构重构**: 添加 `configs/`, `scripts/`, `tests/` 目录
2. **配置文件优化**: 拆分为 `model_config.yaml`, `roi_config.json`, `alarm_config.yaml`
3. **独立脚本创建**: `export_onnx.py`, `validate_onnx.py`, `run_demo.py`
4. **日志系统添加**: 统一的日志工具和关键模块日志输出
5. **测试用例创建**: 检测器、ROI、报警逻辑的测试用例
6. **文档完善**: 部署流程、模型规范、README 更新

**优化原则**:
- 简单优先，不过度设计
- 保持兼容，最小化改动
- 工程化，便于维护和迁移
- 文档完善，便于理解和使用

**后续工作**:
- Jetson 到手后生成 TensorRT engine
- 完善 C++ 推理程序
- 性能测试与优化
- 现场部署与调试

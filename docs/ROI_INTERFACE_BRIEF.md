# ROI 对接说明

## 1. 文档目的

这份文档用于统一前端、后端和检测服务之间的 ROI 数据格式。

核心原则如下：

- 前端负责在画面上框选和编辑 ROI。
- 后端负责保存和下发 ROI 配置。
- 检测服务负责读取 ROI，并用于区域判断和报警逻辑。



## 2. 推荐 JSON 格式

```json
{
  "version": "1.0",
  "camera_id": "cam_demo_001",
  "rois": [
    {
      "roi_id": "hazard_1",
      "name": "全图危险区域",
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

## 3. 字段说明

### 外层字段

- `version`
  - ROI 协议版本号
  - 当前固定为 `1.0`

- `camera_id`
  - 摄像头或视频流唯一标识
  - 用于区分不同设备、不同产线或不同场景

- `rois`
  - ROI 列表
  - 一个摄像头可以对应一个或多个 ROI

### 单个 ROI 字段

- `roi_id`
  - ROI 唯一标识
  - 建议全局或单 camera 内唯一
  - 例如：`hazard_1`、`warning_1`

- `name`
  - ROI 显示名称
  - 主要用于前端展示、日志和调试

- `enabled`
  - 是否启用
  - `true` 表示该 ROI 参与检测逻辑
  - `false` 表示保留配置但暂不生效

- `roi_type`
  - ROI 业务类型
  - 当前建议支持：
    - `clear_zone`：清扫区
    - `warning_zone`：预警区
    - `forbidden_zone`：禁止区

- `judge_method`
  - 区域判断方式
  - 当前支持：
    - `foot_point`：脚底点判定
    - `center_point`：中心点判定
    - `overlap`：重叠面积判定

- `coordinate_mode`
  - 坐标类型
  - 当前支持：
    - `normalized`：归一化坐标，范围 `0.0 ~ 1.0`
    - `absolute`：像素坐标

- `polygon`
  - ROI 多边形顶点数组
  - 至少 3 个点
  - 点顺序可以顺时针或逆时针

- `overlap_thres`
  - 重叠比例阈值
  - 仅当 `judge_method = overlap` 时使用
  - 例如 `0.2` 表示重叠比例超过 20% 才算进入区域

## 4. 为什么推荐使用 normalized 坐标

推荐前后端统一使用 `normalized` 坐标。

原因如下：

- 前端显示的图片尺寸和算法处理的图片尺寸可能不一致
- 页面可能缩放，视频流分辨率也可能变化
- 如果直接保存像素坐标，ROI 很容易错位

使用归一化坐标后：

- 前端画框时可以先按当前页面尺寸计算
- 提交给后端前统一转成 `0.0 ~ 1.0`
- 检测端运行时再按当前帧宽高恢复成像素点

这样同一份 ROI 配置就能适配不同分辨率。

## 5. 前端实现建议

前端主要负责“画出来”和“传出去”。

建议实现内容：

- 在视频画面或抓拍图上支持鼠标点击多边形框选
- 支持拖动顶点调整 ROI
- 支持新增、删除、启用、禁用 ROI
- 默认提交 `normalized` 坐标
- 不需要关心算法内部实现，只需要保证字段格式符合协议

前端提交给后端的内容，本质上就是这份 ROI JSON。

## 6. 后端实现建议

后端主要负责“存起来”和“发出去”。

建议实现内容：

- 提供 ROI 保存接口
- 提供 ROI 查询接口
- 按 `camera_id` 或 `scene_id` 维度管理 ROI
- 将 ROI 持久化到数据库、配置中心或文件
- 原样返回给检测服务，尽量不要修改字段结构

如果某个 ROI 暂时不用，建议设置 `enabled=false`，不建议直接删除。

## 7. 检测服务使用规则

检测服务主要负责“读配置”和“做判断”。

当前规则如下：

- `enabled=false` 的 ROI 不参与运行时判断
- `normalized` 坐标会先转换成当前帧尺寸下的像素坐标
- 人员检测完成后，再拿检测框与 ROI 做区域判定
- 判定结果再交给报警逻辑处理

也就是说，检测服务不关心 ROI 是人工写配置、后端接口下发，还是前端框选出来的，只关心最终数据格式是否符合协议。

## 8. 推荐接口形式

### 保存 ROI

接口：

`POST /api/v1/roi/save`

请求示例：

```json
{
  "version": "1.0",
  "camera_id": "cam_demo_001",
  "rois": [
    {
      "roi_id": "hazard_1",
      "name": "危险区域1",
      "enabled": true,
      "roi_type": "forbidden_zone",
      "judge_method": "foot_point",
      "coordinate_mode": "normalized",
      "polygon": [[0.33, 0.18], [0.78, 0.18], [0.78, 0.98], [0.33, 0.98]],
      "overlap_thres": 0.2
    }
  ]
}
```

### 查询 ROI

接口：

`GET /api/v1/roi/list?camera_id=cam_demo_001`

返回示例：

```json
{
  "version": "1.0",
  "camera_id": "cam_demo_001",
  "rois": [
    {
      "roi_id": "hazard_1",
      "name": "危险区域1",
      "enabled": true,
      "roi_type": "forbidden_zone",
      "judge_method": "foot_point",
      "coordinate_mode": "normalized",
      "polygon": [[0.33, 0.18], [0.78, 0.18], [0.78, 0.98], [0.33, 0.98]],
      "overlap_thres": 0.2
    }
  ]
}
```


## 10. 当前项目中的对应位置

当前项目里，相关文件如下：

- ROI 运行配置：
  - [config.json](/Users/wanan/PycharmProjects/camera/python_prototype/config.json)

- ROI 配置解析：
  - [main.py](/Users/wanan/PycharmProjects/camera/python_prototype/main.py)

- ROI 数据结构定义：
  - [data_models.py](/Users/wanan/PycharmProjects/camera/python_prototype/data_models.py)

- ROI 区域判断实现：
  - [roi_manager.py](/Users/wanan/PycharmProjects/camera/python_prototype/roi_manager.py)

## 11. 当前推荐结论

如果项目还在原型阶段，建议先按下面的方式推进：

- ROI 协议按本文档固定下来
- 当前先用本地 `config.json` 读取 ROI
- 后续 Web 端直接按同样结构提交 ROI
- 检测服务不需要改算法逻辑，只切换 ROI 来源即可

这样后续从“本地配置版”升级到“远程框选版”时，改动会最小。

"""
ROI 管理器测试
"""
import sys
from pathlib import Path

# 添加 python_prototype 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "python_prototype"))

from data_models import Detection, ROIRule, ROIType
from roi_manager import ROIManager


def test_point_in_polygon():
    """测试点是否在多边形内"""
    roi = ROIRule(
        roi_id="test_1",
        name="测试区域",
        roi_type=ROIType.FORBIDDEN,
        polygon=[(100, 100), (200, 100), (200, 200), (100, 200)],
        judge_method="center_point",
        coordinate_mode="absolute",
    )

    manager = ROIManager([roi])

    # 测试点在多边形内
    assert manager.point_in_polygon((150, 150), [(100, 100), (200, 100), (200, 200), (100, 200)])
    print("✓ 点在多边形内测试通过")

    # 测试点在多边形外
    assert not manager.point_in_polygon((50, 50), [(100, 100), (200, 100), (200, 200), (100, 200)])
    print("✓ 点在多边形外测试通过")


def test_normalized_coordinates():
    """测试归一化坐标"""
    roi = ROIRule(
        roi_id="test_2",
        name="归一化区域",
        roi_type=ROIType.WARNING,
        polygon=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        judge_method="center_point",
        coordinate_mode="normalized",
    )

    manager = ROIManager([roi])
    image_shape = (480, 640, 3)  # H, W, C

    resolved = manager.resolve_polygon(roi, image_shape)
    assert resolved[0] == (0, 0)
    assert resolved[1] == (639, 0)
    assert resolved[2] == (639, 479)
    assert resolved[3] == (0, 479)
    print("✓ 归一化坐标转换测试通过")


def test_detection_judgment():
    """测试检测目标判断"""
    roi = ROIRule(
        roi_id="test_3",
        name="中心区域",
        roi_type=ROIType.CLEAR,
        polygon=[(200, 200), (400, 200), (400, 400), (200, 400)],
        judge_method="center_point",
        coordinate_mode="absolute",
    )

    manager = ROIManager([roi])
    image_shape = (480, 640, 3)

    # 创建一个在 ROI 内的检测
    detection_inside = Detection(
        class_id=0,
        class_name="person",
        confidence=0.9,
        bbox=[250, 250, 350, 350],
        center=(300, 300),
        foot_point=(300, 350),
    )

    hits = manager.judge_detection(detection_inside, image_shape)
    assert len(hits) == 1
    assert hits[0]["roi_id"] == "test_3"
    print("✓ 检测目标在 ROI 内测试通过")

    # 创建一个在 ROI 外的检测
    detection_outside = Detection(
        class_id=0,
        class_name="person",
        confidence=0.9,
        bbox=[50, 50, 150, 150],
        center=(100, 100),
        foot_point=(100, 150),
    )

    hits = manager.judge_detection(detection_outside, image_shape)
    assert len(hits) == 0
    print("✓ 检测目标在 ROI 外测试通过")


def test_overlap_judgment():
    """测试重叠率判断"""
    roi = ROIRule(
        roi_id="test_4",
        name="重叠区域",
        roi_type=ROIType.FORBIDDEN,
        polygon=[(200, 200), (400, 200), (400, 400), (200, 400)],
        judge_method="overlap",
        overlap_thres=0.5,
        coordinate_mode="absolute",
    )

    manager = ROIManager([roi])
    image_shape = (480, 640, 3)

    # 完全在 ROI 内
    bbox_inside = [250.0, 250.0, 350.0, 350.0]
    ratio = manager.bbox_overlap_ratio(bbox_inside, roi, image_shape)
    assert ratio == 1.0
    print(f"✓ 完全重叠测试通过 (ratio={ratio})")

    # 部分重叠
    bbox_partial = [150.0, 150.0, 250.0, 250.0]
    ratio = manager.bbox_overlap_ratio(bbox_partial, roi, image_shape)
    assert 0 < ratio < 1.0
    print(f"✓ 部分重叠测试通过 (ratio={ratio:.2f})")

    # 完全不重叠
    bbox_outside = [50.0, 50.0, 150.0, 150.0]
    ratio = manager.bbox_overlap_ratio(bbox_outside, roi, image_shape)
    assert ratio == 0.0
    print(f"✓ 不重叠测试通过 (ratio={ratio})")


if __name__ == "__main__":
    print("=" * 60)
    print("ROI 管理器测试")
    print("=" * 60)

    test_point_in_polygon()
    test_normalized_coordinates()
    test_detection_judgment()
    test_overlap_judgment()

    print("=" * 60)
    print("✓ 所有测试通过")
    print("=" * 60)

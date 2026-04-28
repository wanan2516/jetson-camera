"""
检测器测试
"""
import sys
from pathlib import Path

# 添加 python_prototype 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "python_prototype"))

from data_models import ThresholdConfig, build_detection


def test_build_detection():
    """测试检测结果构建"""
    detection = build_detection(
        class_id=0,
        class_name="person",
        confidence=0.85,
        bbox=[100.0, 200.0, 300.0, 400.0],
    )

    assert detection.class_id == 0
    assert detection.class_name == "person"
    assert detection.confidence == 0.85
    assert detection.bbox == [100.0, 200.0, 300.0, 400.0]
    assert detection.center == (200, 300)
    assert detection.foot_point == (200, 400)
    print("✓ 检测结果构建测试通过")


def test_threshold_config():
    """测试阈值配置"""
    # 正常配置
    config = ThresholdConfig(conf_thres=0.35, iou_thres=0.45)
    assert config.conf_thres == 0.35
    assert config.iou_thres == 0.45
    print("✓ 正常阈值配置测试通过")

    # 边界值
    config = ThresholdConfig(conf_thres=0.0, iou_thres=1.0)
    assert config.conf_thres == 0.0
    assert config.iou_thres == 1.0
    print("✓ 边界值阈值配置测试通过")

    # 非法值
    try:
        config = ThresholdConfig(conf_thres=1.5, iou_thres=0.45)
        assert False, "应该抛出 ValueError"
    except ValueError:
        print("✓ 非法阈值配置测试通过")


def test_detection_center_calculation():
    """测试检测框中心点计算"""
    # 正方形框
    detection = build_detection(0, "person", 0.9, [100.0, 100.0, 200.0, 200.0])
    assert detection.center == (150, 150)
    assert detection.foot_point == (150, 200)
    print("✓ 正方形框中心点计算测试通过")

    # 长方形框
    detection = build_detection(0, "person", 0.9, [100.0, 100.0, 300.0, 400.0])
    assert detection.center == (200, 250)
    assert detection.foot_point == (200, 400)
    print("✓ 长方形框中心点计算测试通过")

    # 小数坐标
    detection = build_detection(0, "person", 0.9, [100.5, 100.5, 200.5, 200.5])
    assert detection.center == (150, 150)
    assert detection.foot_point == (150, 200)
    print("✓ 小数坐标中心点计算测试通过")


if __name__ == "__main__":
    print("=" * 60)
    print("检测器测试")
    print("=" * 60)

    test_build_detection()
    test_threshold_config()
    test_detection_center_calculation()

    print("=" * 60)
    print("✓ 所有测试通过")
    print("=" * 60)

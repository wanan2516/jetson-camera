"""
报警逻辑测试
"""
import sys
from pathlib import Path

# 添加 python_prototype 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "python_prototype"))

from data_models import Detection, ROIRule, ROIType, SystemState
from alarm_logic import AlarmLogic


def test_debounce_logic():
    """测试防抖逻辑"""
    roi = ROIRule(
        roi_id="test_zone",
        name="测试区域",
        roi_type=ROIType.FORBIDDEN,
        polygon=[(0, 0), (100, 0), (100, 100), (0, 100)],
    )

    alarm = AlarmLogic(roi_rules=[roi], enter_frames=3, exit_frames=5)

    # 创建一个在 ROI 内的检测
    detection = Detection(
        class_id=0,
        class_name="person",
        confidence=0.9,
        bbox=[10, 10, 50, 50],
        center=(30, 30),
        foot_point=(30, 50),
        roi_hits=[{"roi_id": "test_zone", "roi_name": "测试区域", "roi_type": "forbidden_zone", "inside": True, "method": "foot_point"}],
    )

    # 第 1 帧：检测到目标，但未达到 enter_frames
    state, allow_start, warning, alarm_flag = alarm.evaluate([detection])
    assert not alarm_flag
    assert alarm.zone_status["test_zone"].enter_counter == 1
    print("✓ 第 1 帧：未触发报警")

    # 第 2 帧：检测到目标，但未达到 enter_frames
    state, allow_start, warning, alarm_flag = alarm.evaluate([detection])
    assert not alarm_flag
    assert alarm.zone_status["test_zone"].enter_counter == 2
    print("✓ 第 2 帧：未触发报警")

    # 第 3 帧：检测到目标，达到 enter_frames，触发报警
    state, allow_start, warning, alarm_flag = alarm.evaluate([detection])
    assert alarm_flag
    assert alarm.zone_status["test_zone"].stable_active
    print("✓ 第 3 帧：触发报警")

    # 第 4 帧：目标消失，但未达到 exit_frames
    state, allow_start, warning, alarm_flag = alarm.evaluate([])
    assert alarm_flag
    assert alarm.zone_status["test_zone"].exit_counter == 1
    print("✓ 第 4 帧：报警持续")

    # 第 5-8 帧：目标持续消失
    for i in range(4):
        state, allow_start, warning, alarm_flag = alarm.evaluate([])
        if i < 3:
            assert alarm_flag
        else:
            assert not alarm_flag
    print("✓ 第 5-8 帧：报警解除")


def test_clear_zone_logic():
    """测试清扫区逻辑"""
    roi = ROIRule(
        roi_id="clear_zone",
        name="清扫区",
        roi_type=ROIType.CLEAR,
        polygon=[(0, 0), (100, 0), (100, 100), (0, 100)],
    )

    alarm = AlarmLogic(roi_rules=[roi], enter_frames=3, exit_frames=5)

    detection = Detection(
        class_id=0,
        class_name="person",
        confidence=0.9,
        bbox=[10, 10, 50, 50],
        center=(30, 30),
        foot_point=(30, 50),
        roi_hits=[{"roi_id": "clear_zone", "roi_name": "清扫区", "roi_type": "clear_zone", "inside": True, "method": "foot_point"}],
    )

    # 启动前模式：清扫区有人
    for _ in range(3):
        state, allow_start, warning, alarm_flag = alarm.evaluate([detection], prestart_mode=True)
    assert state == SystemState.PRESTART_BLOCKED
    assert not allow_start
    print("✓ 清扫区有人：禁止启动")

    # 清扫区无人
    for _ in range(5):
        state, allow_start, warning, alarm_flag = alarm.evaluate([], prestart_mode=True)
    assert state == SystemState.SAFE
    assert allow_start
    print("✓ 清扫区无人：允许启动")


def test_warning_zone_logic():
    """测试预警区逻辑"""
    roi = ROIRule(
        roi_id="warning_zone",
        name="预警区",
        roi_type=ROIType.WARNING,
        polygon=[(0, 0), (100, 0), (100, 100), (0, 100)],
    )

    alarm = AlarmLogic(roi_rules=[roi], enter_frames=3, exit_frames=5)

    detection = Detection(
        class_id=0,
        class_name="person",
        confidence=0.9,
        bbox=[10, 10, 50, 50],
        center=(30, 30),
        foot_point=(30, 50),
        roi_hits=[{"roi_id": "warning_zone", "roi_name": "预警区", "roi_type": "warning_zone", "inside": True, "method": "foot_point"}],
    )

    # 预警区有人
    for _ in range(3):
        state, allow_start, warning, alarm_flag = alarm.evaluate([detection])
    assert state == SystemState.WARNING
    assert warning
    assert not alarm_flag
    print("✓ 预警区有人：触发警告")


if __name__ == "__main__":
    print("=" * 60)
    print("报警逻辑测试")
    print("=" * 60)

    test_debounce_logic()
    test_clear_zone_logic()
    test_warning_zone_logic()

    print("=" * 60)
    print("✓ 所有测试通过")
    print("=" * 60)

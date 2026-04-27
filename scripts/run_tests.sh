#!/bin/bash
# 测试运行脚本

echo "================================"
echo "运行测试用例"
echo "================================"

# 检查是否在 conda 环境中
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "警告: 未检测到 Conda 环境"
    echo "请先激活环境: conda activate camera-yolo"
    echo ""
fi

echo "1. 运行检测器测试..."
python tests/test_detector.py
if [ $? -eq 0 ]; then
    echo "✓ 检测器测试通过"
else
    echo "✗ 检测器测试失败"
    exit 1
fi

echo ""
echo "2. 运行 ROI 测试..."
python tests/test_roi.py
if [ $? -eq 0 ]; then
    echo "✓ ROI 测试通过"
else
    echo "✗ ROI 测试失败"
    exit 1
fi

echo ""
echo "3. 运行报警逻辑测试..."
python tests/test_alarm.py
if [ $? -eq 0 ]; then
    echo "✓ 报警逻辑测试通过"
else
    echo "✗ 报警逻辑测试失败"
    exit 1
fi

echo ""
echo "================================"
echo "✓ 所有测试通过"
echo "================================"

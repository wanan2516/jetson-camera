#!/usr/bin/env python3
"""
ONNX 验证脚本
验证 ONNX 模型的输入输出格式，并可选地在测试图像上运行推理
"""
import argparse
import json
import sys
from pathlib import Path

# 添加 python_prototype 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "python_prototype"))

from onnx_validator import ONNXValidator


def validate_onnx(model_path: str, image_path: str | None = None, imgsz: int = 640) -> dict:
    """验证 ONNX 模型"""
    validator = ONNXValidator(model_path=model_path, imgsz=imgsz)
    result = validator.validate(image_path=image_path)
    return result.to_dict()


def main():
    parser = argparse.ArgumentParser(description="验证 ONNX 模型格式和推理")
    parser.add_argument("--model", type=str, required=True, help="ONNX 模型路径")
    parser.add_argument("--image", type=str, default=None, help="测试图像路径（可选）")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图像尺寸（默认 640）")

    args = parser.parse_args()

    print(f"开始验证 ONNX 模型...")
    print(f"  模型路径: {args.model}")
    if args.image:
        print(f"  测试图像: {args.image}")

    result = validate_onnx(model_path=args.model, image_path=args.image, imgsz=args.imgsz)

    print("\n" + "=" * 60)
    print("ONNX 模型验证结果")
    print("=" * 60)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("=" * 60)

    print(f"\n✓ ONNX 模型验证成功")
    print(f"  输入名称: {result['input_name']}")
    print(f"  输入形状: {result['input_shape']}")
    print(f"  输出数量: {len(result['output_names'])}")

    if result["output_summaries"]:
        print(f"\n推理输出统计:")
        for summary in result["output_summaries"]:
            print(f"  {summary['name']}: shape={summary['shape']}, dtype={summary['dtype']}")
            if summary["min"] is not None:
                print(f"    min={summary['min']:.4f}, max={summary['max']:.4f}, mean={summary['mean']:.4f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
ONNX 导出脚本
将 YOLOv11n PyTorch 模型导出为 ONNX 格式
"""
import argparse
import json
import sys
from pathlib import Path

# 添加 python_prototype 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "python_prototype"))

from detector import PersonDetector
from data_models import ThresholdConfig


def export_onnx(
    model_path: str,
    output_path: str | None = None,
    imgsz: int = 640,
    opset: int = 12,
    simplify: bool = True,
) -> str:
    """导出 ONNX 模型"""
    thresholds = ThresholdConfig(conf_thres=0.35, iou_thres=0.45)
    detector = PersonDetector(
        model_path=model_path,
        thresholds=thresholds,
        imgsz=imgsz,
    )

    onnx_path = detector.export_onnx(output_path=output_path, opset=opset, simplify=simplify)

    if not Path(onnx_path).exists():
        raise FileNotFoundError(f"ONNX 导出失败: {onnx_path}")

    return onnx_path


def main():
    parser = argparse.ArgumentParser(description="导出 YOLOv11n 模型为 ONNX 格式")
    parser.add_argument("--model", type=str, required=True, help="PyTorch 模型路径 (.pt)")
    parser.add_argument("--output", type=str, default=None, help="输出 ONNX 路径（默认与模型同名）")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图像尺寸（默认 640）")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset 版本（默认 12）")
    parser.add_argument("--no-simplify", action="store_true", help="不简化 ONNX 模型")

    args = parser.parse_args()

    print(f"开始导出 ONNX 模型...")
    print(f"  模型路径: {args.model}")
    print(f"  输入尺寸: {args.imgsz}")
    print(f"  ONNX opset: {args.opset}")
    print(f"  简化模型: {not args.no_simplify}")

    onnx_path = export_onnx(
        model_path=args.model,
        output_path=args.output,
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=not args.no_simplify,
    )

    result = {
        "status": "success",
        "onnx_path": str(Path(onnx_path).resolve()),
        "file_size_mb": round(Path(onnx_path).stat().st_size / 1024 / 1024, 2),
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n✓ ONNX 导出成功: {onnx_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from .model import DEFAULT_CONFIG, Model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Python TensorRT inference on one image.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to AIConfig.yaml")
    parser.add_argument("--image", default=None, help="Input image path")
    parser.add_argument("--output", default="outputs/python_tensorrt_result.jpg", help="Output image path")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    image_path = Path(args.image).resolve() if args.image else config_path.parent / "../inference/test_img/image.png"
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"无法读取测试图片: {image_path}")

    with Model(config=config_path) as model:
        result = model.inference(frame)

    if not cv2.imwrite(str(output_path), result):
        raise RuntimeError(f"无法写入输出图片: {output_path}")

    print(f"Python TensorRT output: {output_path}")


if __name__ == "__main__":
    main()


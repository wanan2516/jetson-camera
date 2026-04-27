#!/usr/bin/env python3
"""
演示脚本
快速运行人员检测演示
"""
import argparse
import sys
from pathlib import Path

# 添加 python_prototype 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "python_prototype"))

from main import SafetyVisionSystem, load_config, process_image, process_stream


def main():
    parser = argparse.ArgumentParser(description="YOLOv11n 人员检测演示")
    parser.add_argument("--config", type=str, default="configs/config.json", help="配置文件路径")
    parser.add_argument("--mode", type=str, choices=["image", "video", "camera"], default="image", help="运行模式")
    parser.add_argument("--input", type=str, help="输入图像/视频路径或摄像头索引")
    parser.add_argument("--output", type=str, default=None, help="输出路径（可选）")
    parser.add_argument("--prestart", action="store_true", help="启用启动前清扫区检查")
    parser.add_argument("--no-display", action="store_true", help="不显示窗口（仅保存结果）")

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    # 解析配置文件路径：优先按当前工作目录解析，便于从项目根目录运行 README 命令。
    config_path = Path(args.config)
    if not config_path.is_absolute():
        cwd_path = Path.cwd() / config_path
        config_path = cwd_path if cwd_path.exists() else project_root / config_path
    config_path = config_path.resolve()

    if not config_path.exists():
        print(f"错误: 配置文件不存在: {config_path}")
        sys.exit(1)

    print(f"加载配置: {config_path}")
    config = load_config(str(config_path))
    system = SafetyVisionSystem(config)

    if args.mode == "image":
        if not args.input:
            print("错误: image 模式需要 --input 参数")
            sys.exit(1)
        print(f"处理图像: {args.input}")
        process_image(system, args.input, args.output, args.prestart)

    elif args.mode == "video":
        if not args.input:
            print("错误: video 模式需要 --input 参数")
            sys.exit(1)
        print(f"处理视频: {args.input}")
        process_stream(system, args.input, args.prestart, args.output, display=not args.no_display)

    else:  # camera
        camera_index = int(args.input) if args.input else 0
        print(f"打开摄像头: {camera_index}")
        process_stream(system, camera_index, args.prestart, args.output, display=not args.no_display)


if __name__ == "__main__":
    main()

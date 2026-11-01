"""Small command-line interface; the UI calls the same implementation."""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from .features import METHODS
from .protocols import PROTOCOLS


def main():
    parser = argparse.ArgumentParser(description="全景拼接与可复现评测")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("stitch", help="拼接目录中的全部图片")
    run.add_argument("folder")
    run.add_argument("--output")
    run.add_argument("--feature", choices=list(METHODS), default="sift")
    run.add_argument("--projection", choices=["spherical", "planar"], default="spherical")
    run.add_argument("--blending", choices=["multiband", "feather"], default="multiband")
    run.add_argument("--max-size", type=int, default=1200)
    run.add_argument("--max-features", type=int, default=1800)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--no-optimize", action="store_true")
    run.add_argument("--no-exposure", action="store_true")
    ui = commands.add_parser("ui", help="打开本地操作面板")
    ui.add_argument("--port", type=int, default=8501)
    bench = commands.add_parser("benchmark", help="评测数据集清单")
    bench.add_argument("manifest")
    bench.add_argument("--output", required=True)
    bench.add_argument("--methods", nargs="+", choices=list(METHODS), default=list(METHODS))
    bench.add_argument("--max-size", type=int, help="仅自定义协议可用")
    bench.add_argument(
        "--protocol",
        choices=["auto", *PROTOCOLS],
        default="auto",
    )
    bench.add_argument("--workers", type=int, default=4)
    bench.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.command == "ui":
        app = Path(__file__).with_name("ui.py")
        os.execv(
            sys.executable,
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(app),
                "--server.address=127.0.0.1",
                f"--server.port={args.port}",
                "--server.headless=true",
                "--browser.gatherUsageStats=false",
            ],
        )
    if args.command == "benchmark":
        from .benchmark import evaluate

        evaluate(
            args.manifest,
            args.output,
            args.methods,
            args.max_size,
            args.workers,
            args.limit,
            args.protocol,
        )
        return
    from .pipeline import Config, stitch

    config = Config(
        feature=args.feature,
        projection=args.projection,
        blending=args.blending,
        max_size=args.max_size,
        max_features=args.max_features,
        seed=args.seed,
        optimize=not args.no_optimize,
        exposure=not args.no_exposure,
    )
    output = args.output or f"runs/{datetime.now():%Y%m%d-%H%M%S}-{args.feature}"

    def progress(fraction, message):
        print(f"[{fraction:5.0%}] {message}", flush=True)

    try:
        result = stitch(args.folder, output, config, progress)
    except ValueError as error:
        parser.exit(2, f"无法完成拼接：{error}\n")
    print("已保存：", Path(output).resolve(), "· 图像数量：", len(result["images"]))

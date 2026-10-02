"""Command line interface for the local pyramid builder.

Usage:
    python -m pyramid_tool INPUT OUTPUT_DIR [options]
    python -m pyramid_tool --verify OUTPUT_DIR
"""

from __future__ import annotations

import argparse
import sys

from .pyramid import BuildConfig, RESAMPLE_MODES, build_pyramid, verify_pyramid


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="pyramid_tool",
        description="Local image pyramid & tile generator (no external services).",
    )
    parser.add_argument("input", help="input image (PNG/JPEG) or, with --verify, the pyramid dir")
    parser.add_argument("output_dir", nargs="?", help="output directory for tiles + manifest")
    parser.add_argument("--tile-size", type=int, default=256, help="tile size in pixels (default: 256)")
    parser.add_argument("--resample", choices=RESAMPLE_MODES, default="bilinear", help="resampling filter")
    parser.add_argument("--min-size", type=int, default=256, help="stop when max(level w,h) <= this (default: 256)")
    parser.add_argument("--verify", action="store_true", help="verify an existing pyramid directory instead of building")
    args = parser.parse_args(argv)

    if args.verify:
        problems = verify_pyramid(args.input)
        if problems:
            for p in problems:
                print(f"FAIL: {p}")
            return 1
        print(f"OK: pyramid at {args.input} is consistent with its manifest")
        return 0

    if not args.output_dir:
        parser.error("output_dir is required when building")

    config = BuildConfig(tile_size=args.tile_size, resample=args.resample, min_size=args.min_size)
    stats = build_pyramid(args.input, args.output_dir, config)
    print(
        f"levels={stats.levels} tiles_total={stats.tiles_total} "
        f"written={stats.tiles_written} reused={stats.tiles_reused}"
    )
    print(f"manifest: {stats.manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

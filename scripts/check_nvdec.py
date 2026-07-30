#!/usr/bin/env python3
"""Probe NVDEC in a child process so a native codec crash is recoverable."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check whether PyNvVideoCodec can decode one exact capture."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    return parser


def _child(video: Path) -> int:
    from gpu_decode import GpuVideoDecoder

    decoder = GpuVideoDecoder(str(video), verbose=False)
    frame = next(decoder.frames())
    print(
        json.dumps(
            {
                "schema": "cubed-core/nvdec-check-v1",
                "ok": True,
                "width": decoder.width,
                "height": decoder.height,
                "frames": decoder.n_frames,
                "first_frame_shape": list(frame.bgr.shape),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def main() -> int:
    args = _parser().parse_args()
    video = args.video.expanduser().resolve()
    if not video.is_file():
        print(f"NVDEC check failed: video is not a file: {video}", file=sys.stderr)
        return 2
    if args.child:
        return _child(video)

    command = [sys.executable, str(Path(__file__).resolve()), "--child", "--video", str(video)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        print("NVDEC probe timed out", file=sys.stderr)
        return 1
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "native decoder exited"
        print(
            f"NVDEC probe failed (child return {result.returncode}): {detail}",
            file=sys.stderr,
        )
        return 1
    print(result.stdout.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

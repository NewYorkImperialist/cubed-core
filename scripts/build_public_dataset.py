#!/usr/bin/env python3
"""Build one deterministic, rights-reviewed public corpus candidate."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from public_dataset import PublicDatasetError, build_public_dataset


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a public corpus candidate from a private, explicit source "
            "allowlist. The command is local-only and never uploads data."
        )
    )
    parser.add_argument(
        "--plan",
        type=Path,
        required=True,
        help="absolute path to the private build-plan JSON outside the Git checkout",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="new absolute output directory outside the Git checkout",
    )
    parser.add_argument(
        "--ffprobe-timeout",
        type=int,
        default=30,
        help="per-video ffprobe timeout in seconds (default: 30)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        report = build_public_dataset(
            arguments.plan,
            arguments.output_dir,
            timeout_seconds=arguments.ffprobe_timeout,
        )
    except (PublicDatasetError, OSError) as exc:
        print(f"public dataset build failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Built validated public dataset candidate: {arguments.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

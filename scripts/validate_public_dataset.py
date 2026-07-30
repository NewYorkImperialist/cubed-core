#!/usr/bin/env python3
"""Validate one complete local Cubed Core public dataset candidate."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from public_dataset import PublicDatasetError, validate_public_dataset


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate hashes, schemas, rights, media facts, viewer metadata, "
            "checksums, and the exact file inventory of a local public dataset."
        )
    )
    parser.add_argument("dataset_root", type=Path, help="local dataset root")
    parser.add_argument(
        "--ffprobe-timeout",
        type=int,
        default=300,
        help="per-video ffprobe timeout in seconds (default: 300)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        report = validate_public_dataset(
            arguments.dataset_root,
            timeout_seconds=arguments.ffprobe_timeout,
        )
    except (PublicDatasetError, OSError) as exc:
        print(f"public dataset validation failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

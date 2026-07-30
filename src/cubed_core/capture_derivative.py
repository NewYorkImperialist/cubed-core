from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any

FRAME_SELECTION_FILTER = "select=not(mod(n\\,2))"
TRANSCODE_TIMEOUT_SECONDS = 30 * 60
VERSION_TIMEOUT_SECONDS = 10
MAX_VERSION_OUTPUT_BYTES = 64 * 1024
MAX_RATE_COMPONENT = 1_000_000_000


class CaptureDerivativeError(ValueError):
    pass


def _workspace_path(root: Path, relative: str, *, must_exist: bool) -> Path:
    path = PurePosixPath(relative)
    if (
        not relative
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in relative
        or path.as_posix() != relative
    ):
        raise CaptureDerivativeError("derivative path must be workspace-relative")
    candidate = root / relative
    if candidate.is_symlink():
        raise CaptureDerivativeError("derivative paths may not be symlinks")
    try:
        resolved_root = root.resolve(strict=True)
        resolved = candidate.resolve(strict=must_exist)
        resolved.relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise CaptureDerivativeError("derivative path is outside the workspace") from exc
    if must_exist and not resolved.is_file():
        raise CaptureDerivativeError("source video is unavailable")
    return resolved


def _ffmpeg_version(executable: str) -> dict[str, Any]:
    argv = [executable, "-version"]
    try:
        result = subprocess.run(
            argv,
            capture_output=True,
            check=False,
            timeout=VERSION_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CaptureDerivativeError(f"ffmpeg version check failed: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise CaptureDerivativeError((detail or "ffmpeg version check failed")[:1000])
    if not result.stdout or len(result.stdout) > MAX_VERSION_OUTPUT_BYTES:
        raise CaptureDerivativeError("ffmpeg version output is unavailable or too large")
    return {
        "argv": argv,
        "output": result.stdout.decode("utf-8", errors="replace").rstrip("\n"),
        "output_sha256": hashlib.sha256(result.stdout).hexdigest(),
    }


def create_240_to_120_video(
    workspace_root: Path,
    *,
    source_relative: str,
    output_relative: str,
    target_frame_rate: str,
) -> dict[str, Any]:
    """Select frames 0, 2, 4, ... and retime them to one derived CFR cadence."""

    root = workspace_root.resolve(strict=True)
    _workspace_path(root, source_relative, must_exist=True)
    output_path = _workspace_path(root, output_relative, must_exist=False)
    if output_path.exists():
        raise CaptureDerivativeError("derivative output already exists")
    if output_path.parent.is_symlink():
        raise CaptureDerivativeError("derivative output directory may not be a symlink")
    output_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)

    if not isinstance(target_frame_rate, str) or not re.fullmatch(
        r"[1-9][0-9]*/[1-9][0-9]*", target_frame_rate
    ):
        raise CaptureDerivativeError("target frame rate must be a positive rational")
    numerator_text, denominator_text = target_frame_rate.split("/", 1)
    if (
        len(numerator_text) > 10
        or len(denominator_text) > 10
        or int(numerator_text) > MAX_RATE_COMPONENT
        or int(denominator_text) > MAX_RATE_COMPONENT
    ):
        raise CaptureDerivativeError("target frame-rate rational is too large")
    target_rate = Fraction(int(numerator_text), int(denominator_text))
    if not 110 <= target_rate <= 121:
        raise CaptureDerivativeError("target frame rate must be in the 110-121 fps regime")
    target_rate_text = f"{target_rate.numerator}/{target_rate.denominator}"
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise CaptureDerivativeError("ffmpeg is not installed")
    executable = str(Path(ffmpeg).resolve(strict=True))
    version = _ffmpeg_version(executable)
    cadence_filter = (
        f"{FRAME_SELECTION_FILTER},setpts=N*{target_rate.denominator}/({target_rate.numerator}*TB)"
    )
    argv = [
        executable,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-n",
        "-i",
        source_relative,
        "-map",
        "0:v:0",
        "-vf",
        cadence_filter,
        "-an",
        "-sn",
        "-dn",
        "-r",
        target_rate_text,
        "-fps_mode",
        "cfr",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-threads",
        "1",
        "-map_metadata",
        "-1",
        "-map_chapters",
        "-1",
        "-fflags",
        "+bitexact",
        "-flags:v",
        "+bitexact",
        "-movflags",
        "+faststart",
        output_relative,
    ]
    try:
        result = subprocess.run(
            argv,
            cwd=root,
            capture_output=True,
            check=False,
            timeout=TRANSCODE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        output_path.unlink(missing_ok=True)
        raise CaptureDerivativeError(f"240-to-120 derivation failed: {exc}") from exc
    if result.returncode != 0:
        output_path.unlink(missing_ok=True)
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise CaptureDerivativeError((detail or "ffmpeg rejected the derivative")[:1000])
    if output_path.is_symlink() or not output_path.is_file() or output_path.stat().st_size == 0:
        output_path.unlink(missing_ok=True)
        raise CaptureDerivativeError("ffmpeg did not create a non-empty derivative")

    return {
        "executable": executable,
        "version_argv": version["argv"],
        "version_output": version["output"],
        "version_output_sha256": version["output_sha256"],
        "argv": argv,
        "working_directory": "workspace-root",
        "target_frame_rate": target_rate_text,
    }

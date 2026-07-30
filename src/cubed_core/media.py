from __future__ import annotations

import json
import math
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
MAX_FRAME_JPEG_BYTES = 16 * 1024**2
STANDARD_CAPTURE_FPS_MIN = 110.0
STANDARD_CAPTURE_FPS_MAX = 121.0
NATIVE_240_CAPTURE_FPS_MIN = 220.0
NATIVE_240_CAPTURE_FPS_MAX = 242.0
MIN_CAPTURE_SHORT_EDGE_PIXELS = 1080


class MediaError(ValueError):
    pass


def _rate_fraction(value: object) -> Fraction | None:
    if not isinstance(value, str) or not value or value == "0/0":
        return None
    try:
        rate = Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None
    return rate if rate > 0 else None


def _rate(value: object) -> float | None:
    rate = _rate_fraction(value)
    return round(float(rate), 4) if rate is not None else None


def _finite_frame_rate(fps: object) -> float | None:
    if isinstance(fps, bool) or not isinstance(fps, (int, float)):
        return None
    normalized = float(fps)
    return normalized if math.isfinite(normalized) else None


def is_standard_capture_frame_rate(fps: object) -> bool:
    normalized = _finite_frame_rate(fps)
    return (
        normalized is not None
        and STANDARD_CAPTURE_FPS_MIN <= normalized <= STANDARD_CAPTURE_FPS_MAX
    )


def is_standard_capture_resolution(width: object, height: object) -> bool:
    """Return whether encoded dimensions meet the standard decode floor."""

    return (
        type(width) is int
        and type(height) is int
        and width > 0
        and height > 0
        and min(width, height) >= MIN_CAPTURE_SHORT_EDGE_PIXELS
    )


def is_native_240_capture_frame_rate(fps: object) -> bool:
    normalized = _finite_frame_rate(fps)
    return (
        normalized is not None
        and NATIVE_240_CAPTURE_FPS_MIN <= normalized <= NATIVE_240_CAPTURE_FPS_MAX
    )


def classify_frame_rate(fps: float | None) -> tuple[str, str]:
    normalized = _finite_frame_rate(fps)
    if normalized is None:
        return "unknown", "Frame rate could not be determined; inspect before decoding."
    if is_standard_capture_frame_rate(normalized):
        return "target", "The intended 120 fps operating regime."
    if is_native_240_capture_frame_rate(normalized):
        return (
            "research-high-speed",
            "Preserve and evaluate the native high-speed source. For comparison with "
            "120 fps baselines, create a separately checksummed deterministic derivative.",
        )
    if 60 <= normalized < STANDARD_CAPTURE_FPS_MIN:
        return (
            "borderline",
            "Retained for diagnostics and labeling; camera-to-moves decode is expected "
            "to be unreliable below the 120 fps regime.",
        )
    if normalized > STANDARD_CAPTURE_FPS_MAX:
        return (
            "unsupported",
            "Retained as nonstandard high-frame-rate evidence; only measured "
            "220–242 fps qualifies as native-240 research footage.",
        )
    return "unsupported", "Retained as evidence only; 30 fps is not a supported decode input."


def probe_video(path: Path, *, count_frames: bool = False) -> dict[str, Any]:
    """Inspect one video without changing it.

    Failure is represented in the receipt so importing footage never depends on
    ffprobe being installed. The doctor command tells the user how to fix it.
    """

    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return {"status": "unavailable", "error": "ffprobe is not installed"}

    def _command(include_side_data: bool) -> list[str]:
        stream_entries = (
            "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,duration,"
            "nb_frames,nb_read_frames:stream_tags=rotate"
        )
        if include_side_data:
            stream_entries += ":stream_side_data=rotation"
        return [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            *(["-count_frames"] if count_frames else []),
            "-show_entries",
            stream_entries,
            "-show_entries",
            "format=format_name,duration",
            "-of",
            "json",
            str(path),
        ]

    try:
        result = subprocess.run(
            _command(include_side_data=True),
            capture_output=True,
            check=False,
            text=True,
            timeout=30,
        )
        # ffprobe before version 5 has no stream_side_data section selector
        # and rejects the whole invocation. Rotation on such builds is still
        # visible through the rotate stream tag, so retry without the
        # side-data request instead of failing the probe.
        if result.returncode != 0 and "stream_side_data" in (result.stderr or ""):
            result = subprocess.run(
                _command(include_side_data=False),
                capture_output=True,
                check=False,
                text=True,
                timeout=30,
            )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "error", "error": str(exc)}
    if result.returncode != 0:
        message = result.stderr.strip() or "ffprobe rejected the file"
        return {"status": "error", "error": message[:500]}
    try:
        payload = json.loads(result.stdout)
        stream = payload.get("streams", [])[0]
    except (IndexError, TypeError, json.JSONDecodeError):
        return {"status": "error", "error": "no readable video stream"}
    average_rate = _rate_fraction(stream.get("avg_frame_rate"))
    nominal_rate = _rate_fraction(stream.get("r_frame_rate"))
    selected_rate = average_rate or nominal_rate
    fps = round(float(selected_rate), 4) if selected_rate is not None else None
    fps_rational = (
        f"{selected_rate.numerator}/{selected_rate.denominator}"
        if selected_rate is not None
        else None
    )
    fps_basis = (
        "avg_frame_rate"
        if average_rate is not None
        else "r_frame_rate"
        if nominal_rate is not None
        else None
    )
    duration_value = stream.get("duration") or payload.get("format", {}).get("duration")
    try:
        duration_seconds = round(float(duration_value), 3)
    except (TypeError, ValueError):
        duration_seconds = None
    frame_count = None
    for field in ("nb_frames", "nb_read_frames"):
        try:
            frame_count = int(stream.get(field))
        except (TypeError, ValueError):
            continue
        if frame_count > 0:
            break
        frame_count = None
    # Keep the same decoder-facing convention as Cubed's established native
    # ingest path: ffprobe display-matrix side data is counter-clockwise
    # positive, while the legacy rotate tag is already clockwise positive.
    # Prefer the display matrix when both are present.
    side_rotation = next(
        (
            side_data["rotation"]
            for side_data in stream.get("side_data_list") or []
            if isinstance(side_data, dict) and "rotation" in side_data
        ),
        None,
    )
    tagged_rotation = stream.get("tags", {}).get("rotate")
    try:
        clockwise_rotation = (
            -float(side_rotation) if side_rotation is not None else float(tagged_rotation)
        ) % 360.0
        distance_from_right_angle = min(
            clockwise_rotation % 90.0,
            90.0 - clockwise_rotation % 90.0,
        )
        if distance_from_right_angle > 1.0:
            raise ValueError("rotation is not a right angle")
        rotation_degrees = int(round(clockwise_rotation / 90.0)) * 90 % 360
    except (TypeError, ValueError):
        rotation_degrees = 0

    capture_class, guidance = classify_frame_rate(fps)

    return {
        "status": "ok",
        "codec": stream.get("codec_name"),
        "container": payload.get("format", {}).get("format_name"),
        "width": stream.get("width"),
        "height": stream.get("height"),
        "fps": fps,
        "fps_rational": fps_rational,
        "fps_basis": fps_basis,
        "avg_frame_rate": (
            f"{average_rate.numerator}/{average_rate.denominator}"
            if average_rate is not None
            else None
        ),
        "r_frame_rate": (
            f"{nominal_rate.numerator}/{nominal_rate.denominator}"
            if nominal_rate is not None
            else None
        ),
        "frame_count": frame_count,
        "rotation_degrees": rotation_degrees,
        "duration_seconds": duration_seconds,
        "capture_class": capture_class,
        "guidance": guidance,
    }


def extract_frame_jpeg(path: Path, frame_index: int) -> bytes:
    """Decode one exact display-order frame without materializing the full movie."""
    if frame_index < 0:
        raise MediaError("frame index must be nonnegative")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise MediaError("ffmpeg is not installed")
    command = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        str(path),
        "-vf",
        f"select=eq(n\\,{frame_index})",
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "pipe:1",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MediaError(f"frame extraction failed: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise MediaError((detail or "ffmpeg rejected the frame request")[:500])
    if not result.stdout:
        raise MediaError("frame is outside the video")
    if len(result.stdout) > MAX_FRAME_JPEG_BYTES:
        raise MediaError("decoded frame exceeds the safety limit")
    return result.stdout

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from cubed_core import media
from cubed_core.media import classify_frame_rate, is_standard_capture_resolution


@pytest.mark.parametrize(
    ("fps", "expected"),
    [
        (None, "unknown"),
        (30, "unsupported"),
        (60, "borderline"),
        (110, "target"),
        (121, "target"),
        (121.001, "unsupported"),
        (219.999, "unsupported"),
        (220, "research-high-speed"),
        (242, "research-high-speed"),
        (242.001, "unsupported"),
    ],
)
def test_frame_rate_contract(fps: float | None, expected: str) -> None:
    assert classify_frame_rate(fps)[0] == expected


@pytest.mark.parametrize(
    ("width", "height", "expected"),
    [
        (1920, 1080, True),
        (1080, 1920, True),
        (1280, 720, False),
        (1079, 1920, False),
        (None, 1080, False),
        (True, 1920, False),
    ],
)
def test_standard_decode_resolution_uses_the_encoded_short_edge(
    width: object,
    height: object,
    expected: bool,
) -> None:
    assert is_standard_capture_resolution(width, height) is expected


def test_counted_probe_uses_decoded_frame_count_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "source.mov"
    video.write_bytes(b"video")
    payload = {
        "streams": [
            {
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "240/1",
                "r_frame_rate": "240/1",
                "duration": "1.0",
                "nb_frames": "N/A",
                "nb_read_frames": "240",
            }
        ],
        "format": {"format_name": "mov,mp4", "duration": "1.0"},
    }
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append(list(command))
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    monkeypatch.setattr(media.shutil, "which", lambda command: "/usr/bin/ffprobe")
    monkeypatch.setattr(media.subprocess, "run", fake_run)

    probe = media.probe_video(video, count_frames=True)

    assert probe["frame_count"] == 240
    assert probe["fps"] == 240.0
    assert probe["fps_rational"] == "240/1"
    assert probe["fps_basis"] == "avg_frame_rate"
    assert probe["avg_frame_rate"] == "240/1"
    assert probe["r_frame_rate"] == "240/1"
    assert "-count_frames" in commands[0]
    entries = commands[0][commands[0].index("-show_entries") + 1]
    assert "nb_read_frames" in entries
    assert "stream_side_data=rotation" in entries
    assert ":side_data=rotation" not in entries


def test_probe_converts_display_matrix_rotation_to_decoder_clockwise(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "portrait.mov"
    video.write_bytes(b"video")
    payload = {
        "streams": [
            {
                "avg_frame_rate": "120/1",
                "side_data_list": [{"rotation": -90}],
                # Display-matrix side data is authoritative when both forms
                # happen to be present.
                "tags": {"rotate": "270"},
            }
        ],
        "format": {"format_name": "mov,mp4"},
    }

    monkeypatch.setattr(media.shutil, "which", lambda command: "/usr/bin/ffprobe")
    monkeypatch.setattr(
        media.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command,
            0,
            json.dumps(payload),
            "",
        ),
    )

    probe = media.probe_video(video)

    assert probe["rotation_degrees"] == 90


def test_probe_keeps_legacy_rotate_tag_clockwise(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "legacy.mov"
    video.write_bytes(b"video")
    payload = {
        "streams": [
            {
                "avg_frame_rate": "120/1",
                "tags": {"rotate": "90"},
            }
        ],
        "format": {"format_name": "mov,mp4"},
    }

    monkeypatch.setattr(media.shutil, "which", lambda command: "/usr/bin/ffprobe")
    monkeypatch.setattr(
        media.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command,
            0,
            json.dumps(payload),
            "",
        ),
    )

    probe = media.probe_video(video)

    assert probe["rotation_degrees"] == 90


def test_probe_retries_without_side_data_on_legacy_ffprobe(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ffprobe 4.x rejects the stream_side_data section selector.

    The probe must retry without it so imports work on hosts whose ffmpeg
    predates version 5, such as the conda ffmpeg inside the pytorch Docker
    images the GPU guide recommends. Rotation still arrives via the rotate
    stream tag on those builds.
    """

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"stub")
    payload = {
        "streams": [
            {
                "codec_name": "h264",
                "width": 320,
                "height": 240,
                "avg_frame_rate": "240/1",
                "r_frame_rate": "240/1",
                "nb_frames": "3",
                "tags": {"rotate": "90"},
            }
        ],
        "format": {"format_name": "mp4", "duration": "0.0125"},
    }
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(list(command))
        joined = " ".join(command)
        if "stream_side_data" in joined:
            return subprocess.CompletedProcess(
                command,
                1,
                "",
                "No match for section 'stream_side_data'\n"
                "Failed to set value ... for option 'show_entries': Invalid argument",
            )
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    monkeypatch.setattr(media.shutil, "which", lambda command: "/usr/bin/ffprobe")
    monkeypatch.setattr(media.subprocess, "run", fake_run)

    probe = media.probe_video(video)

    assert len(calls) == 2
    assert any("stream_side_data" in " ".join(c) for c in calls[:1])
    assert all("stream_side_data" not in " ".join(c) for c in calls[1:])
    assert probe["status"] == "ok"
    assert probe.get("frame_count") == 3

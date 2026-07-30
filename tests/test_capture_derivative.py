from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from fractions import Fraction
from io import BytesIO
from pathlib import Path

import pytest

from cubed_core import capture_derivative as derivative_module
from cubed_core import workspace as workspace_module
from cubed_core.capture_derivative import (
    FRAME_SELECTION_FILTER,
    CaptureDerivativeError,
    create_240_to_120_video,
)
from cubed_core.workspace import Workspace, WorkspaceError


def _source_probe(*, fps: float = 240.0, frames: int = 9) -> dict[str, object]:
    rate = Fraction(str(fps))
    return {
        "status": "ok",
        "codec": "h264",
        "container": "mov,mp4",
        "width": 1920,
        "height": 1080,
        "fps": fps,
        "fps_rational": f"{rate.numerator}/{rate.denominator}",
        "fps_basis": "avg_frame_rate",
        "avg_frame_rate": f"{rate.numerator}/{rate.denominator}",
        "r_frame_rate": f"{rate.numerator}/{rate.denominator}",
        "frame_count": frames,
        "rotation_degrees": 0,
        "duration_seconds": 0.0375,
        "capture_class": "research-high-speed",
        "guidance": "high speed",
    }


def _output_probe(*, fps: float = 120.0, frames: int = 5) -> dict[str, object]:
    rate = Fraction(str(fps))
    return {
        "status": "ok",
        "codec": "h264",
        "container": "mov,mp4",
        "width": 1920,
        "height": 1080,
        "fps": fps,
        "fps_rational": f"{rate.numerator}/{rate.denominator}",
        "fps_basis": "avg_frame_rate",
        "avg_frame_rate": f"{rate.numerator}/{rate.denominator}",
        "r_frame_rate": f"{rate.numerator}/{rate.denominator}",
        "frame_count": frames,
        "rotation_degrees": 0,
        "duration_seconds": 0.0375,
        "capture_class": "target",
        "guidance": "target",
    }


def _ffmpeg_receipt(
    source_relative: str,
    output_relative: str,
    target_frame_rate: str,
) -> dict[str, object]:
    version = b"ffmpeg version test-build\nconfiguration: --enable-libx264\n"
    target = Fraction(target_frame_rate)
    cadence_filter = (
        f"{FRAME_SELECTION_FILTER},setpts=N*{target.denominator}/({target.numerator}*TB)"
    )
    return {
        "executable": "/usr/bin/ffmpeg",
        "version_argv": ["/usr/bin/ffmpeg", "-version"],
        "version_output": version.decode().rstrip("\n"),
        "version_output_sha256": hashlib.sha256(version).hexdigest(),
        "argv": [
            "/usr/bin/ffmpeg",
            "-i",
            source_relative,
            "-vf",
            cadence_filter,
            "-r",
            target_frame_rate,
            "-fps_mode",
            "cfr",
            output_relative,
        ],
        "working_directory": "workspace-root",
        "target_frame_rate": target_frame_rate,
    }


def _install_fake_derivative_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    output_fps: float = 120.0,
    output_frames: int = 5,
) -> list[tuple[str, str]]:
    calls: list[tuple[str, str]] = []

    def fake_probe(path: Path, *, count_frames: bool = False) -> dict[str, object]:
        del count_frames
        if path.read_bytes() == b"native-240-original":
            return _source_probe()
        return _output_probe(fps=output_fps, frames=output_frames)

    def fake_create(
        workspace_root: Path,
        *,
        source_relative: str,
        output_relative: str,
        target_frame_rate: str,
    ) -> dict[str, object]:
        calls.append((source_relative, output_relative))
        output = workspace_root / output_relative
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"deterministic-120-derivative")
        return _ffmpeg_receipt(source_relative, output_relative, target_frame_rate)

    monkeypatch.setattr(workspace_module, "probe_video", fake_probe)
    monkeypatch.setattr(workspace_module, "create_240_to_120_video", fake_create)
    return calls


def _import_source(workspace: Workspace) -> dict[str, object]:
    return workspace.import_video(
        BytesIO(b"native-240-original"),
        filename="native.mov",
        source="native-ios",
        capture_session_id="attempt-1",
        scramble="R U R'",
        camera_facing="front",
        mirrored=False,
        camera_intrinsics={
            "matrix": [
                [1200.0, 0.0, 960.0],
                [0.0, 1180.0, 540.0],
                [0.0, 0.0, 1.0],
            ],
            "ref_w": 1920,
            "ref_h": 1080,
        },
    )


def test_derivative_preserves_original_and_creates_linked_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_derivative_runtime(monkeypatch)
    workspace = Workspace(
        tmp_path / "workspace",
        max_upload_bytes=2048,
    )
    source = _import_source(workspace)
    source_dir = tmp_path / "workspace" / "captures" / str(source["capture_id"])
    before = {
        path.name: path.read_bytes()
        for path in (
            source_dir / "source.mov",
            source_dir / "capture.json",
            source_dir / "checksums.sha256",
        )
    }

    derived = workspace.derive_240_to_120(str(source["capture_id"]))

    assert calls and calls[0][0] == source["video"]["path"]
    assert derived["capture_id"] != source["capture_id"]
    assert derived["source"] == "derived-240-to-120"
    assert derived["state"] == "incomplete"
    assert derived["capture_session_id"] == "attempt-1"
    assert derived["solve"]["scramble"] == "R U R'"
    assert derived["camera"]["facing"] == "front"
    assert source["camera"]["intrinsics"] is not None
    assert derived["camera"]["intrinsics"] is None
    assert derived["video"]["mirrored"] is False
    assert derived["video"]["actual_fps"] == 120.0
    assert derived["video"]["frame_count"] == 5
    assert derived["provenance"]["normalized_from_sha256"] == source["video"]["sha256"]

    linkage = derived["derivation"]
    assert linkage["schema"] == "cubed-core/frame-rate-derivation"
    assert linkage["source"]["recording_id"] == source["capture_id"]
    assert linkage["source"]["sha256"] == source["video"]["sha256"]
    assert linkage["source"]["probe_before_derivation"]["frame_count"] == 9
    assert linkage["transform"]["source_frame_index_origin"] == 0
    assert linkage["transform"]["source_frame_index_step"] == 2
    assert linkage["transform"]["source_frame_rate"] == "240/1"
    assert linkage["transform"]["target_frame_rate"] == "120/1"
    assert linkage["transform"]["expected_output_frame_count"] == 5
    assert linkage["ffmpeg"]["argv"][-1] == calls[0][1]
    assert linkage["derivative"]["recording_id"] == derived["capture_id"]
    assert linkage["derivative"]["sha256"] == derived["video"]["sha256"]
    assert linkage["derivative"]["probe"]["frame_count"] == 5
    assert linkage["derivative"]["probe_at_import"]["frame_count"] == 5
    assert (
        "not promised across FFmpeg builds" in linkage["transform"]["compressed_byte_determinism"]
    )

    after = {
        path.name: path.read_bytes()
        for path in (
            source_dir / "source.mov",
            source_dir / "capture.json",
            source_dir / "checksums.sha256",
        )
    }
    assert after == before
    assert len(workspace.list_captures()) == 2
    assert list((tmp_path / "workspace" / "exports").iterdir()) == []


def test_derivative_rejects_sealed_source_before_transcode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_derivative_runtime(monkeypatch)
    workspace = Workspace(
        tmp_path / "workspace",
        max_upload_bytes=2048,
    )
    source = _import_source(workspace)
    workspace.seal_capture(str(source["capture_id"]), purpose="label")

    with pytest.raises(WorkspaceError, match="sealed captures"):
        workspace.derive_240_to_120(str(source["capture_id"]))

    assert calls == []
    assert len(workspace.list_captures()) == 1


@pytest.mark.parametrize(
    "probe",
    [
        {"status": "error", "error": "unreadable"},
        _source_probe(fps=120.0),
        _source_probe(fps=180.0),
    ],
)
def test_derivative_rejects_unprobed_or_non_240_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    probe: dict[str, object],
) -> None:
    monkeypatch.setattr(workspace_module, "probe_video", lambda path: probe)
    workspace = Workspace(
        tmp_path / "workspace",
        max_upload_bytes=2048,
    )
    source = workspace.import_video(
        BytesIO(b"source"),
        filename="source.mov",
        source="import",
    )
    called = False

    def should_not_run(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("transcode should not run")

    monkeypatch.setattr(workspace_module, "create_240_to_120_video", should_not_run)
    with pytest.raises(WorkspaceError, match="unprobed|verified 240 fps regime"):
        workspace.derive_240_to_120(str(source["capture_id"]))
    assert called is False


def test_derivative_rejects_source_hash_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_derivative_runtime(monkeypatch)
    workspace = Workspace(
        tmp_path / "workspace",
        max_upload_bytes=2048,
    )
    source = _import_source(workspace)
    workspace.capture_video_path(str(source["capture_id"])).write_bytes(b"tampered")

    with pytest.raises(WorkspaceError, match="do not match"):
        workspace.derive_240_to_120(str(source["capture_id"]))
    assert calls == []


def test_derivative_rejects_mismatched_source_identity_before_transcode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_derivative_runtime(monkeypatch)
    workspace = Workspace(
        tmp_path / "workspace",
        max_upload_bytes=2048,
    )
    source = _import_source(workspace)
    capture_dir = tmp_path / "workspace" / "captures" / str(source["capture_id"])
    receipt_path = capture_dir / "capture.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["recording_id"] = "b" * 32
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(WorkspaceError, match="identity does not match"):
        workspace.derive_240_to_120(str(source["capture_id"]))

    assert calls == []
    assert len(workspace.list_captures()) == 1


def test_derivative_accepts_older_numeric_import_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_derivative_runtime(monkeypatch)
    workspace = Workspace(
        tmp_path / "workspace",
        max_upload_bytes=2048,
    )
    source = _import_source(workspace)
    receipt_path = tmp_path / "workspace" / "captures" / str(source["capture_id"]) / "capture.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    for field in ("fps_rational", "fps_basis", "avg_frame_rate", "r_frame_rate"):
        receipt["probe"].pop(field, None)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    derived = workspace.derive_240_to_120(str(source["capture_id"]))

    assert derived["video"]["actual_fps"] == 120.0
    assert derived["derivation"]["transform"]["target_frame_rate"] == "120/1"


@pytest.mark.parametrize(
    ("output_fps", "output_frames"),
    [
        (100.0, 5),
        (120.0, 4),
    ],
)
def test_derivative_rejects_bad_output_probe_without_creating_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    output_fps: float,
    output_frames: int,
) -> None:
    _install_fake_derivative_runtime(
        monkeypatch,
        output_fps=output_fps,
        output_frames=output_frames,
    )
    workspace = Workspace(
        tmp_path / "workspace",
        max_upload_bytes=2048,
    )
    source = _import_source(workspace)

    with pytest.raises(WorkspaceError, match="target cadence and exact"):
        workspace.derive_240_to_120(str(source["capture_id"]))

    assert len(workspace.list_captures()) == 1
    assert list((tmp_path / "workspace" / "exports").iterdir()) == []


def test_direct_derived_import_requires_internal_linkage(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "workspace", max_upload_bytes=1024)
    with pytest.raises(WorkspaceError, match="valid linkage"):
        workspace.import_video(
            BytesIO(b"not-a-derived-receipt"),
            filename="fake.mp4",
            source="derived-240-to-120",
        )


def test_changed_derived_bytes_fail_inside_atomic_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(workspace_module, "probe_video", lambda path: _output_probe())
    workspace = Workspace(tmp_path / "workspace", max_upload_bytes=1024)
    linkage = {
        "schema": "cubed-core/frame-rate-derivation",
        "schema_version": 1,
        "derivative": {
            "bytes": 4,
            "sha256": hashlib.sha256(b"expected").hexdigest(),
            "probe": _output_probe(),
        },
    }

    with pytest.raises(WorkspaceError, match="changed before it was stored"):
        workspace.import_video(
            BytesIO(b"actual"),
            filename="derived.mp4",
            source="derived-240-to-120",
            derivation=linkage,
            normalized_from_sha256="a" * 64,
        )

    assert list((tmp_path / "workspace" / "captures").iterdir()) == []


def test_ffmpeg_wrapper_records_exact_command_and_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    source = workspace / "captures" / ("a" * 32) / "source.mov"
    output = workspace / "exports" / "job" / "derived.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source")
    output.parent.mkdir(parents=True)
    executable = tmp_path / "ffmpeg"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    version_bytes = b"ffmpeg version pinned-test\nconfiguration: --enable-libx264\n"
    commands: list[tuple[list[str], Path | None]] = []

    def fake_run(argv, **kwargs):
        commands.append((list(argv), kwargs.get("cwd")))
        if argv[-1] == "-version":
            return subprocess.CompletedProcess(argv, 0, version_bytes, b"")
        (workspace / argv[-1]).write_bytes(b"derived")
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(derivative_module.shutil, "which", lambda command: str(executable))
    monkeypatch.setattr(derivative_module.subprocess, "run", fake_run)

    receipt = create_240_to_120_video(
        workspace,
        source_relative=f"captures/{'a' * 32}/source.mov",
        output_relative="exports/job/derived.mp4",
        target_frame_rate="120/1",
    )

    assert output.read_bytes() == b"derived"
    assert receipt["version_output_sha256"] == hashlib.sha256(version_bytes).hexdigest()
    assert receipt["argv"] == commands[1][0]
    assert commands[1][1] == workspace.resolve()
    assert receipt["argv"][receipt["argv"].index("-vf") + 1] == (
        f"{FRAME_SELECTION_FILTER},setpts=N*1/(120*TB)"
    )
    assert receipt["argv"][receipt["argv"].index("-r") + 1] == "120/1"
    assert receipt["argv"][receipt["argv"].index("-fps_mode") + 1] == "cfr"
    assert receipt["target_frame_rate"] == "120/1"
    assert receipt["argv"][receipt["argv"].index("-threads") + 1] == "1"
    assert "-an" in receipt["argv"]
    assert "-fps_mode" in receipt["argv"]


def test_ffmpeg_wrapper_refuses_workspace_escape(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with pytest.raises(CaptureDerivativeError, match="workspace-relative"):
        create_240_to_120_video(
            workspace,
            source_relative="../outside.mov",
            output_relative="exports/derived.mp4",
            target_frame_rate="120/1",
        )


@pytest.mark.parametrize(
    ("source_rate", "duration", "target_rate"),
    [
        ("240/1", "0.25", "120/1"),
        ("240000/1001", "0.25025", "120000/1001"),
    ],
)
def test_real_ffmpeg_short_high_speed_clip_produces_exact_half_rate_cfr(
    tmp_path: Path,
    source_rate: str,
    duration: str,
    target_rate: str,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        pytest.skip("real FFmpeg integration requires ffmpeg and ffprobe")

    original = tmp_path / "native-240.mp4"
    generated = subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size=64x64:rate={source_rate}:duration={duration}",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-threads",
            "1",
            str(original),
        ],
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert generated.returncode == 0, generated.stderr.decode(errors="replace")

    original_bytes = original.read_bytes()
    workspace = Workspace(
        tmp_path / "workspace",
        max_upload_bytes=200 * 1024**2,
    )
    with original.open("rb") as stream:
        source = workspace.import_video(
            stream,
            filename=original.name,
            source="external-camera",
        )

    derived = workspace.derive_240_to_120(str(source["capture_id"]))

    assert workspace.capture_video_path(str(source["capture_id"])).read_bytes() == original_bytes
    assert source["video"]["frame_count"] == 60
    assert derived["video"]["actual_fps"] == round(float(Fraction(target_rate)), 4)
    assert derived["video"]["frame_count"] == 30
    assert derived["probe"]["fps_rational"] == target_rate
    assert derived["derivation"]["derivative"]["probe"]["fps_rational"] == target_rate
    assert derived["derivation"]["transform"]["target_frame_rate"] == target_rate
    argv = derived["derivation"]["ffmpeg"]["argv"]
    assert argv[argv.index("-r") + 1] == target_rate
    assert argv[argv.index("-fps_mode") + 1] == "cfr"


def test_derivation_schema_is_linked_from_capture_bundle() -> None:
    root = Path(__file__).resolve().parents[1]
    capture_schema = json.loads((root / "schemas" / "capture-bundle-v1.schema.json").read_text())
    derivation_schema = json.loads(
        (root / "schemas" / "capture-derivation-v1.schema.json").read_text()
    )
    assert capture_schema["properties"]["derivation"]["oneOf"][0]["$ref"] == (
        "capture-derivation-v1.schema.json"
    )
    assert (
        derivation_schema["properties"]["transform"]["properties"]["source_frame_index_step"][
            "const"
        ]
        == 2
    )

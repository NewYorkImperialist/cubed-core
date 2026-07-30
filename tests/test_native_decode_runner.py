from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from cubed_core import native_decode_runner
from cubed_core.native_decode_runner import DecodeRunContext, NativeDecodeRunnerError

_FLAT_CENTROIDS = {
    "red": [114.5, 195.682, 170.477],
    "blue": [98.622, 132.489, 83.889],
    "green": [159.6, 75.156, 154.378],
    "white": [205.143, 129.381, 135.595],
    "orange": [143.178, 182.489, 189.244],
    "yellow": [199.333, 118.289, 202.022],
}


def _context(job_dir: Path, calibration_path: Path) -> DecodeRunContext:
    return DecodeRunContext(
        job_id="a" * 32,
        capture_id="b" * 32,
        job_dir=job_dir,
        output_path=job_dir / "decode-result.json",
        video_path=job_dir / "source.mov",
        calibration_path=calibration_path,
        scramble="R U R'",
        runtime_assets={},
    )


def _write_calibration(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_identity_bound_calibration_rejects_changed_snapshot(tmp_path: Path) -> None:
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_bytes(b'{"schema":"cubed-core/color-centroids-v1"}')
    expected = hashlib.sha256(calibration_path.read_bytes()).hexdigest()

    assert (
        native_decode_runner._identity_bound_file(
            str(calibration_path),
            expected,
            field="decode request calibration",
            maximum_bytes=native_decode_runner.CALIBRATION_MAX_BYTES,
        )
        == calibration_path
    )
    calibration_path.write_bytes(b'{"schema":"changed"}')
    with pytest.raises(NativeDecodeRunnerError, match="request digest"):
        native_decode_runner._identity_bound_file(
            str(calibration_path),
            expected,
            field="decode request calibration",
            maximum_bytes=native_decode_runner.CALIBRATION_MAX_BYTES,
        )


def test_write_centroids_projects_color_calibration_v1(
    tmp_path: Path,
    color_calibration_payload: dict[str, object],
) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    calibration_path = tmp_path / "calibration.json"
    _write_calibration(calibration_path, color_calibration_payload)

    destination = native_decode_runner._write_centroids(_context(job_dir, calibration_path))

    projected = json.loads(destination.read_text())
    assert projected == {
        color: [float(component) for component in vector]
        for color, vector in color_calibration_payload["centroids"].items()
    }


def test_write_centroids_projects_color_centroids_v1_envelope(tmp_path: Path) -> None:
    """A wrapped color-centroids-v1 document projects to the identical flat file
    that scripts/geo_read.py's calib_util.load_centroids expects, byte-equivalent
    in content to the color-calibration-v1 projection above (same six keys, same
    numbers; key order does not matter)."""

    job_dir = tmp_path / "job"
    job_dir.mkdir()
    calibration_path = tmp_path / "calibration.json"
    _write_calibration(
        calibration_path,
        {
            "schema": "cubed-core/color-centroids-v1",
            "schema_version": 1,
            "color_space": "cielab",
            "centroids": _FLAT_CENTROIDS,
            "provenance": "imported-centroids sha256:" + "0" * 64,
        },
    )

    destination = native_decode_runner._write_centroids(_context(job_dir, calibration_path))

    projected = json.loads(destination.read_text())
    assert projected == {
        color: [float(v) for v in vector] for color, vector in _FLAT_CENTROIDS.items()
    }


def test_write_centroids_rejects_bare_flat_map_without_envelope(tmp_path: Path) -> None:
    """The native decode runner only projects declared sidecars; the workspace
    is responsible for normalizing a bare flat map into a color-centroids-v1
    envelope before it ever reaches this stage."""

    job_dir = tmp_path / "job"
    job_dir.mkdir()
    calibration_path = tmp_path / "calibration.json"
    _write_calibration(calibration_path, _FLAT_CENTROIDS)

    with pytest.raises(NativeDecodeRunnerError, match="color-calibration-v1 or"):
        native_decode_runner._write_centroids(_context(job_dir, calibration_path))


def test_write_centroids_rejects_unrecognized_schema(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    calibration_path = tmp_path / "calibration.json"
    _write_calibration(calibration_path, {"schema": "cubed-core/not-a-real-schema"})

    with pytest.raises(NativeDecodeRunnerError):
        native_decode_runner._write_centroids(_context(job_dir, calibration_path))


def test_base_environment_strips_inherited_cubed_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Host shells with stale CUBED_ exports must not reach pipeline stages.

    A leaked CUBED_TRUST_NPZ rewrites a stamped flag token and breaks the
    canonical CFG_HASH, so the runner builds its child environment from a
    CUBED_-free base with an explicit passthrough allowlist.
    """

    monkeypatch.setenv("CUBED_TRUST_NPZ", "/stale/container/trust.npz")
    monkeypatch.setenv("CUBED_ALIGNED_MODEL", "/stale/container/model.onnx")
    monkeypatch.setenv("CUBED_GPU_SCORING", "1")
    monkeypatch.setenv("CUBED_NVDEC", "off")
    monkeypatch.setenv("BASH_ENV", "/stale/container/cv_env.sh")
    monkeypatch.setenv("ENV", "/stale/container/sh_env")
    monkeypatch.setenv("UNRELATED_VALUE", "kept")

    environment = native_decode_runner._base_environment(tmp_path)

    assert "CUBED_TRUST_NPZ" not in environment
    assert "CUBED_ALIGNED_MODEL" not in environment
    assert "CUBED_GPU_SCORING" not in environment
    assert environment["CUBED_NVDEC"] == "off"
    assert "BASH_ENV" not in environment
    assert "ENV" not in environment
    assert environment["UNRELATED_VALUE"] == "kept"
    assert str(tmp_path) in environment["PYTHONPATH"]
    assert environment["CUBED_ORT_PROVIDERS"] == "cuda"
    assert environment["CUBED_ORT_REQUIRE_CUDA"] == "1"


def test_base_environment_puts_the_runner_interpreter_first_on_path(
    tmp_path: Path,
) -> None:
    """Shell stages call plain python3, which must resolve to the venv.

    A fresh compute box has no cv2 or torch outside the runner's virtual
    environment, so the interpreter's bin directory leads PATH.
    """

    environment = native_decode_runner._base_environment(tmp_path)

    interpreter_bin = os.path.dirname(sys.executable)
    assert environment["PATH"].split(os.pathsep)[0] == interpreter_bin


def test_prepare_workstation_payload_merges_same_pass_alignment(
    tmp_path: Path,
) -> None:
    numpy = pytest.importorskip("numpy")
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    calibration_path = tmp_path / "calibration.json"
    context = replace(
        _context(job_dir, calibration_path),
        workstation_context={
            "video": {
                "sha256": "c" * 64,
                "bytes": 2048,
                "fps": 120.0,
                "frame_count": 12,
                "width": 1920,
                "height": 1080,
            },
            "warnings": [
                {
                    "code": "calibration.transfer",
                    "message": "Shared centroids may transfer imperfectly.",
                }
            ],
        },
    )
    workstation_path = tmp_path / "workstation.json"
    workstation_path.write_text(
        json.dumps(
            {
                "video": {
                    "fps": 120.0,
                    "frame_count": 12,
                    # OpenCV applies the container rotation before producing
                    # the overlay coordinate space.
                    "width": 1080,
                    "height": 1920,
                },
                "frames": {
                    "5": {
                        "motion": 1.5,
                        "face_count": 1,
                        "faces": [
                            {
                                "corners": [[1, 2], [3, 4], [5, 6], [7, 8]],
                                "confidence": 0.95,
                                "keypoint_confidence": [0.9, 0.9, 0.9, 0.9],
                            }
                        ],
                    }
                },
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )
    alignfeat_path = tmp_path / "alignfeat.npz"
    numpy.savez(
        alignfeat_path,
        frame=numpy.asarray([5, 6, 7], dtype=numpy.int64),
        aligned=numpy.asarray([0.8, 0.9, 0.2], dtype=numpy.float32),
    )

    assert native_decode_runner._prepare_workstation_payload(
        context,
        workstation_path=workstation_path,
        alignfeat_path=alignfeat_path,
    )

    value = json.loads(workstation_path.read_text(encoding="utf-8"))
    assert value["schema"] == "cubed-core/decode-workstation-v1"
    assert value["video"]["sha256"] == "c" * 64
    assert value["video"]["width"] == 1080
    assert value["video"]["height"] == 1920
    assert value["video"]["encoded"] == {
        "fps": 120.0,
        "frame_count": 12,
        "width": 1920,
        "height": 1080,
    }
    assert value["initialization"] == {"scramble": "R U R'"}
    assert value["window"] == [0, 11]
    assert value["frames"]["5"]["motion"] == 1.5
    assert value["frames"]["5"]["aligned"] == pytest.approx(0.8, abs=0.001)
    assert value["frames"]["5"]["aligned_streak"] == 1
    assert value["frames"]["6"]["aligned_streak"] == 2
    assert value["frames"]["7"]["aligned_streak"] == 0
    assert value["warnings"] == context.workstation_context["warnings"]


def test_workstation_context_rejects_paths_and_unknown_fields() -> None:
    with pytest.raises(NativeDecodeRunnerError, match="workstation"):
        native_decode_runner._load_workstation_context(
            {
                "video": {
                    "sha256": "c" * 64,
                    "bytes": 2048,
                    "fps": 120.0,
                    "frame_count": 12,
                    "width": 1920,
                    "height": 1080,
                    "path": "/private/video.mov",
                },
                "warnings": [],
            }
        )


def test_main_reports_a_bounded_exception_message_for_unexpected_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Anything main() did not already turn into a structured
    NativeDecodeRunnerError still reports enough to diagnose from the job
    log: the exception type and a bounded slice of str(exc), e.g.
    distinguishing an ENOSPC OSError from other failures without needing the
    full traceback.
    """
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    request_path = job_dir / "job-request.json"
    output_path = job_dir / "decode-result.json"
    long_message = "no space left on device: " + "x" * 300

    def _boom(*_args: Any, **_kwargs: Any) -> None:
        raise OSError(long_message)

    monkeypatch.setattr(native_decode_runner, "run", _boom)

    assert (
        native_decode_runner.main(["--request", str(request_path), "--output", str(output_path)])
        == 9
    )
    error = json.loads(capsys.readouterr().err)["error"]
    assert error["code"] == "internal_error"
    assert error["details"]["exception_type"] == "OSError"
    assert error["details"]["exception_message"] == long_message[:200]
    assert len(error["details"]["exception_message"]) == 200

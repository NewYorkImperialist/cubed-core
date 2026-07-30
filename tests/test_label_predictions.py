from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from cubed_core import label_predictions
from cubed_core.label_predictions import (
    LABEL_CURRENT_FRAME_FACE_THRESHOLD,
    LabelPredictionDisabled,
    LabelPredictionError,
    prediction_capability,
    run_label_alignment_scan,
    run_label_prediction,
)
from cubed_core.settings import Settings
from cubed_core.vision import PosePostprocessConfig


def settings_for(
    tmp_path: Path,
    *,
    command: tuple[str, ...] = (),
    model: Path | None = None,
) -> Settings:
    return Settings(
        repo_root=tmp_path,
        workspace=tmp_path / "workspace",
        max_upload_bytes=1024,
        admin_token="admin-secret",
        label_predict_command=command,
        label_model_path=model,
    )


def runner_script(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "runner.py"
    path.write_text(body, encoding="utf-8")
    return path


def test_prediction_capability_requires_runner_and_regular_model(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    disabled = prediction_capability(settings)
    assert disabled["enabled"] is False
    assert disabled["status"] == "disabled"
    assert disabled["executable"] is None
    with pytest.raises(LabelPredictionDisabled):
        run_label_prediction(
            settings,
            capture_id="a" * 32,
            video_path=tmp_path / "video.mp4",
            request={"width": 640, "height": 480, "frame_indices": [1]},
        )

    target = tmp_path / "real.bin"
    target.write_bytes(b"weights")
    symlink = tmp_path / "model.bin"
    symlink.symlink_to(target)
    settings = settings_for(tmp_path, command=(sys.executable,), model=symlink)
    invalid_model = prediction_capability(settings)
    assert invalid_model["enabled"] is False
    assert invalid_model["status"] == "misconfigured"
    assert invalid_model["executable"] == sys.executable
    assert "regular local file" in invalid_model["reason"]

    settings = settings_for(
        tmp_path,
        command=("cubed-core-predictor-that-does-not-exist",),
        model=target,
    )
    invalid_runner = prediction_capability(settings)
    assert invalid_runner["enabled"] is False
    assert invalid_runner["status"] == "misconfigured"
    assert invalid_runner["executable"] is None
    assert "executable was not found" in invalid_runner["reason"]


def test_prediction_runner_is_validated_and_receives_scrubbed_environment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model = tmp_path / "model.bin"
    model.write_bytes(b"weights")
    script = runner_script(
        tmp_path,
        """
import argparse, json, os, sys
p = argparse.ArgumentParser()
p.add_argument("--request"); p.add_argument("--output"); p.add_argument("--model")
a = p.parse_args()
if (os.environ.get("CUBED_CORE_ADMIN_TOKEN") or os.environ.get("MY_API_KEY")
    or os.environ.get("KUBECONFIG") or os.environ.get("NETRC")
    or os.environ.get("GPG_AGENT_INFO") or os.environ.get("SSH_AUTH_SOCK")):
    sys.exit(9)
if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
    sys.exit(10)
request = json.load(open(a.request))
frame = request["frame_indices"][0]
json.dump({
  "schema": "cubed-core/label-predictions-v1",
  "frames": [{"frame_index": frame, "faces": [{
    "corners": [[10,10],[20,10],[20,20],[10,20]],
    "visible": [True, True, False, True],
    "confidence": 0.8
  }]}]
}, open(a.output, "w"))
""",
    )
    monkeypatch.setenv("CUBED_CORE_ADMIN_TOKEN", "do-not-pass")
    monkeypatch.setenv("MY_API_KEY", "do-not-pass")
    monkeypatch.setenv("KUBECONFIG", "do-not-pass")
    monkeypatch.setenv("NETRC", "do-not-pass")
    monkeypatch.setenv("GPG_AGENT_INFO", "do-not-pass")
    monkeypatch.setenv("SSH_AUTH_SOCK", "do-not-pass")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    result = run_label_prediction(
        settings_for(tmp_path, command=(sys.executable, str(script)), model=model),
        capture_id="a" * 32,
        video_path=tmp_path / "video.mp4",
        request={"width": 640, "height": 480, "frame_indices": [7]},
    )
    face = result["frames"][0]["faces"][0]
    assert face["origin"] == "model"
    assert face["confidence"] == 0.8


def test_prediction_rejects_invalid_output_and_times_out(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model = tmp_path / "model.bin"
    model.write_bytes(b"weights")
    invalid = runner_script(
        tmp_path,
        """
import argparse
p = argparse.ArgumentParser()
p.add_argument("--request"); p.add_argument("--output"); p.add_argument("--model")
a = p.parse_args()
open(a.output, "w").write('{"schema":"wrong","frames":[]}')
""",
    )
    with pytest.raises(LabelPredictionError, match="must declare"):
        run_label_prediction(
            settings_for(tmp_path, command=(sys.executable, str(invalid)), model=model),
            capture_id="a" * 32,
            video_path=tmp_path / "video.mp4",
            request={"width": 640, "height": 480, "frame_indices": [1]},
        )

    sleeping = runner_script(
        tmp_path,
        """
import argparse, time
p = argparse.ArgumentParser()
p.add_argument("--request"); p.add_argument("--output"); p.add_argument("--model")
p.parse_args()
time.sleep(30)
""",
    )
    monkeypatch.setattr(label_predictions, "LABEL_PREDICTION_TIMEOUT_SECONDS", 0.05)
    with pytest.raises(LabelPredictionError, match="timed out"):
        run_label_prediction(
            settings_for(tmp_path, command=(sys.executable, str(sleeping)), model=model),
            capture_id="a" * 32,
            video_path=tmp_path / "video.mp4",
            request={"width": 640, "height": 480, "frame_indices": [1]},
        )


class _FakeCapture:
    def __init__(self, frames):
        self.frames = list(frames)
        self.position = 0
        self.released = False

    def set(self, _field, value) -> None:
        self.position = int(value)

    def read(self):
        if self.position >= len(self.frames):
            return False, None
        frame = self.frames[self.position]
        self.position += 1
        return True, frame

    def release(self) -> None:
        self.released = True


class _FakeNativeClient:
    def __init__(self) -> None:
        self.pose_config = PosePostprocessConfig()
        self.pose_configs = []
        self.alignment_values = iter((0.4994, 0.5004, 0.9))

    def infer_faces(self, _frame, *, pose_config=None):
        self.pose_configs.append(pose_config)
        return (
            SimpleNamespace(
                corners=((10.04, 10.05), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0)),
                confidence=0.8765,
            ),
        )

    def classify_alignment(self, _frame) -> float:
        return next(self.alignment_values)


def _native_readiness():
    return SimpleNamespace(
        enabled=True,
        status="available",
        reason=None,
        models={"profile": "camera-tracker-v1"},
    )


def test_native_label_path_reuses_tracker_manifest_for_predict_and_alignment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    settings = settings_for(tmp_path)
    settings = Settings(
        **{
            **settings.__dict__,
            "tracker_model_manifest": tmp_path / "manifest.json",
        }
    )
    client = _FakeNativeClient()
    captures = [
        _FakeCapture([SimpleNamespace(shape=(480, 640, 3)) for _ in range(3)]),
        _FakeCapture([SimpleNamespace(shape=(480, 640, 3)) for _ in range(3)]),
        _FakeCapture([SimpleNamespace(shape=(480, 640, 3)) for _ in range(3)]),
    ]
    monkeypatch.setattr(
        label_predictions,
        "build_camera_tracker_readiness",
        lambda _settings: _native_readiness(),
    )
    monkeypatch.setattr(
        label_predictions,
        "_native_vision_client",
        lambda _settings: (client, "camera-tracker-v1"),
    )
    monkeypatch.setattr(
        label_predictions,
        "_open_native_capture",
        lambda _path: (SimpleNamespace(CAP_PROP_POS_FRAMES=1), captures.pop(0)),
    )
    monkeypatch.setattr(label_predictions, "rotate_frame", lambda frame, _rotation: frame)

    capability = prediction_capability(settings)
    assert capability["backend"] == "camera-tracker-v1"
    assert capability["model_aligned_navigation"] is True

    current = run_label_prediction(
        settings,
        capture_id="a" * 32,
        video_path=video,
        request={"width": 640, "height": 480, "frame_indices": [1]},
        rotation_degrees=0,
    )
    assert current["frames"][0]["faces"][0]["corners"][0] == [10.0, 10.1]
    assert client.pose_configs[0].confidence_threshold == LABEL_CURRENT_FRAME_FACE_THRESHOLD

    all_frames = run_label_prediction(
        settings,
        capture_id="a" * 32,
        video_path=video,
        request={
            "width": 640,
            "height": 480,
            "frame_indices": None,
            "skip_frame_indices": [1],
        },
        rotation_degrees=0,
    )
    assert [row["frame_index"] for row in all_frames["frames"]] == [0, 2]
    assert client.pose_configs[1:] == [None, None]

    alignment = run_label_alignment_scan(
        settings,
        video_path=video,
        rotation_degrees=0,
        expected_frame_count=3,
    )
    assert alignment == {
        "schema": "cubed-core/label-alignment-v1",
        "threshold": 0.5,
        "model_profile": "camera-tracker-v1",
        "width": 640,
        "height": 480,
        "frame_count": 3,
        "aligned_frames": [1, 2],
        "alignment_confs": [0.499, 0.5, 0.9],
    }

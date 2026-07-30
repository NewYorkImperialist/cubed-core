from __future__ import annotations

import copy
import json
from io import BytesIO

import pytest

from cubed_core import workspace as workspace_module
from cubed_core.color_calibration import ColorCalibrationError, validate_color_calibration
from cubed_core.workspace import Workspace, WorkspaceError


def test_exact_server_grid_color_calibration_contract_is_valid(
    color_calibration_payload,
) -> None:
    validate_color_calibration(color_calibration_payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["order"].reverse(),
            "order must equal",
        ),
        (
            lambda value: value["geometry"].__setitem__("grid_rows", 4),
            "geometry.grid_rows must equal 3",
        ),
        (
            lambda value: value["thresholds"].__setitem__("collision_minimum", 14),
            "thresholds.collision_minimum must equal 15",
        ),
        (
            lambda value: value["samples"]["white"].pop(),
            "samples.white must contain exactly 45",
        ),
        (
            lambda value: value["samples"]["white"][0].__setitem__(0, float("nan")),
            "must be a finite number",
        ),
        (
            lambda value: value["samples"]["white"][0].__setitem__(2, 256),
            "must be at most 255",
        ),
        (
            lambda value: value["provenance"].__setitem__("frame_width", 0),
            "frame_width must be a positive integer",
        ),
    ],
)
def test_color_calibration_contract_rejects_drift_and_invalid_values(
    color_calibration_payload,
    mutation,
    message,
) -> None:
    invalid = copy.deepcopy(color_calibration_payload)
    mutation(invalid)

    with pytest.raises(ColorCalibrationError, match=message):
        validate_color_calibration(invalid)


def test_color_calibration_rejects_centroid_sample_mismatch(
    color_calibration_payload,
) -> None:
    color_calibration_payload["centroids"]["white"][1] += 1

    with pytest.raises(ColorCalibrationError, match="does not match the exported samples"):
        validate_color_calibration(color_calibration_payload)


def test_color_calibration_rejects_colliding_centroids(
    color_calibration_payload,
) -> None:
    white = list(color_calibration_payload["centroids"]["white"])
    color_calibration_payload["centroids"]["green"] = white
    color_calibration_payload["samples"]["green"] = [list(white) for _ in range(45)]

    with pytest.raises(ColorCalibrationError, match="are not distinct"):
        validate_color_calibration(color_calibration_payload)


def test_workspace_rejects_partial_calibration_without_writing(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        workspace_module,
        "probe_video",
        lambda path: {"status": "ok", "fps": 120.0},
    )
    workspace = Workspace(tmp_path / "workspace", max_upload_bytes=1024)
    receipt = workspace.import_video(
        BytesIO(b"video"),
        filename="solve.mp4",
        source="native-ios",
    )

    with pytest.raises(WorkspaceError, match="color calibration is missing"):
        workspace.attach_json_sidecar(
            receipt["capture_id"],
            BytesIO(json.dumps({"schema_version": 1}).encode()),
            kind="calibration",
        )

    capture_dir = tmp_path / "workspace" / "captures" / receipt["capture_id"]
    assert not (capture_dir / "calibration.json").exists()
    stored_receipt = json.loads((capture_dir / "capture.json").read_text(encoding="utf-8"))
    assert stored_receipt["calibration"] is None

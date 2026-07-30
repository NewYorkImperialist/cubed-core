from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from cubed_core import label_assist

cv2 = pytest.importorskip("cv2")
numpy = pytest.importorskip("numpy")

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "label_assist_reference_golden_v1.json"
GOLDEN = json.loads(FIXTURE_PATH.read_text())
ABSOLUTE_TOLERANCE = GOLDEN["tolerance"]["absolute"]
RELATIVE_TOLERANCE = GOLDEN["tolerance"]["relative"]


def _assert_numeric(actual: Any, expected: Any) -> None:
    numpy.testing.assert_allclose(
        actual,
        expected,
        atol=ABSOLUTE_TOLERANCE,
        rtol=RELATIVE_TOLERANCE,
    )


def _assert_pose_matches_reference(
    actual: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    assert actual["ok"] is expected["ok"]
    assert actual["intrinsics_source"] == "explicit-camera-matrix"
    assert actual["inferred_corner"] is None
    _assert_numeric(actual["rvec"], expected["rvec"])
    _assert_numeric(actual["tvec"], expected["tvec"])
    _assert_numeric(actual["wireframe"], expected["wireframe"])

    assert [{"name": face["name"], "vertices": face["vertices"]} for face in actual["faces"]] == [
        {"name": face["name"], "vertices": face["vertices"]} for face in expected["faces"]
    ]
    for actual_face, expected_face in zip(actual["faces"], expected["faces"], strict=True):
        _assert_numeric(actual_face["corners"], expected_face["corners"])


def _projected_extrusion_sign(
    result: dict[str, Any],
    camera_matrix: list[list[float]],
) -> int:
    cube_front = numpy.array(
        [
            [-0.5, 0.5, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, -0.5, 0.0],
            [-0.5, -0.5, 0.0],
        ],
        float,
    )
    errors = {}
    for sign in (1, -1):
        cube = numpy.vstack([cube_front, cube_front + [0, 0, sign]])
        projected = cv2.projectPoints(
            cube,
            numpy.asarray(result["rvec"], float),
            numpy.asarray(result["tvec"], float),
            numpy.asarray(camera_matrix, float),
            numpy.zeros(5),
        )[0].reshape(-1, 2)
        errors[sign] = float(
            numpy.max(numpy.abs(projected - numpy.asarray(result["wireframe"], float)))
        )
    return min(errors, key=errors.__getitem__)


def test_reference_golden_preserves_quad_order_and_square_completion() -> None:
    case = GOLDEN["case"]
    expected = GOLDEN["expected"]

    ordered = label_assist._order_quad(case["labeled_corners"], numpy)
    completed = label_assist._complete_square(
        case["three_consecutive_corners"],
        case["camera_matrix"],
        numpy,
    )

    _assert_numeric(ordered, expected["ordered_quad"])
    _assert_numeric(completed, expected["completed_square"])


def test_reference_golden_preserves_pnp_extrusion_and_projection() -> None:
    case = GOLDEN["case"]
    expected = GOLDEN["expected"]["unpinned"]

    actual = label_assist.extrapolate(
        case["labeled_corners"],
        case["camera_matrix"],
        shrink=case["shrink"],
        intrinsics_source="explicit-camera-matrix",
    )

    _assert_pose_matches_reference(actual, expected)
    assert _projected_extrusion_sign(actual, case["camera_matrix"]) == expected["extrusion_sign"]


def test_reference_golden_preserves_pinned_pnp_refinement() -> None:
    case = GOLDEN["case"]
    expected = GOLDEN["expected"]["pinned"]

    actual = label_assist.extrapolate(
        case["labeled_corners"],
        case["camera_matrix"],
        pins=case["pins"],
        shrink=case["shrink"],
        intrinsics_source="explicit-camera-matrix",
    )

    _assert_pose_matches_reference(actual, expected)
    assert _projected_extrusion_sign(actual, case["camera_matrix"]) == expected["extrusion_sign"]
    unpinned_wireframe = numpy.asarray(
        GOLDEN["expected"]["unpinned"]["wireframe"],
        float,
    )
    assert (
        numpy.max(numpy.abs(numpy.asarray(actual["wireframe"], float) - unpinned_wireframe)) > 1.0
    )

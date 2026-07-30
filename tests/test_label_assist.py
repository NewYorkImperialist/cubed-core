from __future__ import annotations

import pytest

from cubed_core import label_assist


def request_body() -> dict[str, object]:
    return {
        "width": 640,
        "height": 480,
        "labeled_corners": [[220, 140], [420, 140], [420, 340]],
    }


def test_label_assist_validates_before_loading_optional_runtime(monkeypatch) -> None:
    with pytest.raises(label_assist.LabelAssistError, match="unsupported field private"):
        label_assist.extrapolate_request({**request_body(), "private": True})

    monkeypatch.setattr(
        label_assist,
        "_runtime",
        lambda: (_ for _ in ()).throw(label_assist.LabelAssistUnavailable("disabled")),
    )
    with pytest.raises(label_assist.LabelAssistUnavailable, match="disabled"):
        label_assist.extrapolate_request(request_body())


def test_cpu_pnp_completes_three_clicks_when_optional_runtime_is_installed() -> None:
    pytest.importorskip("cv2")
    result = label_assist.extrapolate_request(request_body())
    assert result["ok"] is True
    assert result["intrinsics_source"] == "image-centered-focal-prior"
    assert len(result["wireframe"]) == 8
    assert len(next(face for face in result["faces"] if face["name"] == "labeled")["corners"]) == 4


def test_explicit_camera_matrix_provenance_when_runtime_is_installed() -> None:
    pytest.importorskip("cv2")
    body = {
        **request_body(),
        "K": [[600, 0, 320], [0, 600, 240], [0, 0, 1]],
    }
    result = label_assist.extrapolate_request(body)
    assert result["ok"] is True
    assert result["intrinsics_source"] == "explicit-camera-matrix"


@pytest.mark.parametrize(
    "corners",
    [
        [[10, 10], [10, 10], [20, 20]],
        [[10, 10], [20, 20], [30, 30]],
        [[10, 10], [20, 20], [30, 30], [40, 40]],
    ],
)
def test_label_assist_rejects_degenerate_faces_before_loading_runtime(
    corners: list[list[int]],
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        label_assist,
        "_runtime",
        lambda: (_ for _ in ()).throw(
            AssertionError("runtime should not load for invalid geometry")
        ),
    )
    with pytest.raises(label_assist.LabelAssistError, match="distinct|nondegenerate"):
        label_assist.extrapolate_request({**request_body(), "labeled_corners": corners})


def test_label_assist_rejects_singular_camera_matrix() -> None:
    body = {
        **request_body(),
        "K": [[600, 1, 0], [0, 600, 1], [600, 601, 1]],
    }
    with pytest.raises(label_assist.LabelAssistError, match="invertible"):
        label_assist.extrapolate_request(body)

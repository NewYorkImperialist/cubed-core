from __future__ import annotations

import pytest

numpy = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from cubed_core.vision.reader import (  # noqa: E402
    GeometricFaceReader,
    assign_face_slots,
)
from cubed_core.vision.sampling import (  # noqa: E402
    OcclusionPolicy,
    SamplingError,
    bgr_to_lab,
    low_chroma_mask,
    occlusion_mask,
    sample_masked_lab_cells,
)
from cubed_core.vision.types import FaceDetection  # noqa: E402


def _face(
    corners: tuple[tuple[float, float], ...],
    *,
    confidence: float = 0.9,
) -> FaceDetection:
    xs = [point[0] for point in corners]
    ys = [point[1] for point in corners]
    return FaceDetection(
        corners=corners,
        keypoint_confidences=(0.9, 0.9, 0.9, 0.9),
        confidence=confidence,
        bbox_xywh=(
            (min(xs) + max(xs)) / 2,
            (min(ys) + max(ys)) / 2,
            max(xs) - min(xs),
            max(ys) - min(ys),
        ),
    )


def test_low_chroma_mask_protects_bright_neutral_pixels() -> None:
    lab = numpy.asarray(
        [
            [[100, 128, 128], [220, 128, 128]],
            [[150, 180, 180], [150, 170, 90]],
        ],
        dtype=numpy.float32,
    )
    assert numpy.array_equal(
        low_chroma_mask(lab),
        [[True, False], [False, False]],
    )


def test_occlusion_mask_combines_low_chroma_and_centroid_distance() -> None:
    lab = numpy.asarray(
        [[[150, 180, 180], [100, 128, 128], [230, 128, 128]]],
        dtype=numpy.float32,
    )
    centroids = numpy.asarray(((150, 180, 180), (230, 128, 128)), numpy.float32)
    mask = occlusion_mask(
        lab,
        centroids=centroids,
        policy=OcclusionPolicy(maximum_centroid_distance=15),
    )
    assert numpy.array_equal(mask, [[False, True, False]])


def test_masked_sampler_returns_medians_and_explicit_support() -> None:
    lab = numpy.full((96, 96, 3), (150, 180, 180), dtype=numpy.float32)
    # Four low-chroma pixels inside the first 16x16 sampling window.
    lab[16:18, 16:18] = (100, 128, 128)
    sampled = sample_masked_lab_cells(lab)

    assert sampled.lab.shape == (9, 3)
    assert sampled.confidence.shape == (9,)
    assert sampled.valid_pixels[0] == 252
    assert sampled.total_pixels[0] == 256
    assert sampled.confidence[0] == pytest.approx(252 / 256)
    assert numpy.array_equal(sampled.lab[0], [150, 180, 180])
    assert sampled.used_fallback == (False,) * 9


def test_masked_sampler_marks_low_support_fallback_without_hiding_it() -> None:
    lab = numpy.full((96, 96, 3), (100, 128, 128), dtype=numpy.float32)
    sampled = sample_masked_lab_cells(lab)

    assert numpy.all(sampled.confidence == 0)
    assert sampled.valid_pixels == (0,) * 9
    assert sampled.used_fallback == (True,) * 9
    assert numpy.all(sampled.lab == [100, 128, 128])


def test_masked_sampler_rejects_out_of_bounds_or_nonfinite_inputs() -> None:
    with pytest.raises(SamplingError, match="outside"):
        sample_masked_lab_cells(numpy.zeros((50, 50, 3), numpy.float32))
    invalid = numpy.zeros((96, 96, 3), numpy.float32)
    invalid[0, 0, 0] = numpy.nan
    with pytest.raises(SamplingError, match="finite"):
        sample_masked_lab_cells(invalid)
    with pytest.raises(SamplingError, match="window_half"):
        sample_masked_lab_cells(numpy.zeros((96, 96, 3), numpy.float32), window_half=13)


def test_bgr_to_lab_has_stable_shape_and_dtype() -> None:
    bgr = numpy.full((8, 10, 3), (0, 0, 255), dtype=numpy.uint8)
    lab = bgr_to_lab(bgr)
    assert lab.shape == bgr.shape
    assert lab.dtype == numpy.float32
    assert numpy.array_equal(lab[0, 0], cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0, 0])


def test_positional_face_slot_assignment_is_input_order_independent() -> None:
    up = _face(((60, 10), (90, 10), (90, 40), (60, 40)))
    front = _face(((10, 70), (50, 70), (50, 110), (10, 110)))
    right = _face(((90, 65), (140, 65), (140, 110), (90, 110)))

    assigned = assign_face_slots((right, up, front))

    assert assigned == (("up", up), ("front", front), ("right", right))


def test_geometric_reader_connects_warp_and_masked_sampling() -> None:
    frame = numpy.zeros((150, 160, 3), dtype=numpy.uint8)
    up = _face(((60, 10), (90, 10), (90, 40), (60, 40)))
    front = _face(((10, 70), (50, 70), (50, 110), (10, 110)))
    right = _face(((90, 65), (140, 65), (140, 110), (90, 110)))
    cv2.fillConvexPoly(frame, numpy.asarray(up.corners, numpy.int32), (0, 0, 255))
    cv2.fillConvexPoly(frame, numpy.asarray(front.corners, numpy.int32), (0, 255, 0))
    cv2.fillConvexPoly(frame, numpy.asarray(right.corners, numpy.int32), (255, 0, 0))

    result = GeometricFaceReader()(frame, (right, up, front))

    assert [face.slot for face in result.faces] == ["up", "front", "right"]
    assert [face.relative_area for face in result.faces] == [0.4, 0.711, 1.0]
    assert all(len(face.lab) == 9 for face in result.faces)
    assert all(min(face.confidence) > 0.95 for face in result.faces)
    assert result.slot_reads[0][0] == "up"

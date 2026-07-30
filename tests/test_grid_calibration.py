from __future__ import annotations

import random
from typing import Any

import pytest

from cubed_core import grid_calibration
from cubed_core.vision import VisionRuntimeUnavailable


@pytest.fixture(scope="module")
def cv_runtime() -> tuple[Any, Any]:
    numpy = pytest.importorskip("numpy")
    cv2 = pytest.importorskip("cv2")
    return cv2, numpy


def _source_cell_patches(width: float, height: float) -> list[tuple[int, int, int, int]]:
    """Independent geometry oracle."""

    side = 0.56 * min(width, height)
    gx0 = width / 2.0 - side / 2.0
    gy0 = height / 2.0 - side / 2.0
    cell = side / 3
    patch = 0.40 * cell
    rects = []
    for r in range(3):
        for c in range(3):
            pcx = gx0 + (c + 0.5) * cell
            pcy = gy0 + (r + 0.5) * cell
            rects.append(
                (
                    int(round(pcx - patch / 2.0)),
                    int(round(pcy - patch / 2.0)),
                    int(round(pcx + patch / 2.0)),
                    int(round(pcy + patch / 2.0)),
                )
            )
    return rects


def _source_wdist(numpy: Any, a: Any, b: Any) -> float:
    """Independent distance oracle."""

    weights = numpy.array([0.15, 1.0, 1.0])
    difference = numpy.asarray(a, float) - numpy.asarray(b, float)
    return float(numpy.sqrt(((difference**2) * weights).sum()))


def _source_robust_centroid(numpy: Any, samples: Any) -> Any:
    """Independent robust-centroid oracle."""

    centroid_weights = numpy.array([0.15, 1.0, 1.0])
    array = numpy.asarray(samples, dtype=numpy.float64)
    if len(array) < 6:
        return array.mean(axis=0)
    median = numpy.median(array, axis=0)
    distances = numpy.sqrt((((array - median) ** 2) * centroid_weights).sum(axis=1))
    median_absolute_distance = numpy.median(distances) + 1e-6
    keep = distances <= max(3.0 * median_absolute_distance, 10.0)
    inliers = array[keep]
    return inliers.mean(axis=0) if len(inliers) >= 3 else median


def _source_sample_grid(cv2: Any, numpy: Any, frame_bgr: Any) -> list[Any]:
    """Independent OpenCV sampling oracle."""

    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    height, width = lab.shape[:2]
    medians = []
    for x0, y0, x1, y1 in _source_cell_patches(width, height):
        patch = lab[y0:y1, x0:x1].reshape(-1, 3).astype(numpy.float64)
        medians.append(numpy.median(patch, axis=0))
    return medians


def _source_evaluate_frame(
    numpy: Any,
    medians: Any,
    prev_medians: Any | None,
    min_l: float,
) -> tuple[bool, str | None, bool]:
    """Independent frame-acceptance oracle."""

    if any(median[0] < min_l for median in medians):
        return False, "dark", False
    for i in range(len(medians)):
        for j in range(i + 1, len(medians)):
            if _source_wdist(numpy, medians[i], medians[j]) > 28.0:
                return False, "not_uniform", False
    if prev_medians is None:
        return False, "moving", True
    mean_distance = float(
        numpy.mean(
            [
                _source_wdist(numpy, median, previous)
                for median, previous in zip(medians, prev_medians, strict=False)
            ]
        )
    )
    if mean_distance > 10.0:
        return False, "moving", True
    return True, None, True


def _source_looks_like_collected(
    numpy: Any,
    medians: Any,
    centroids: Any,
    exclude: Any,
) -> Any | None:
    """Independent collected-color oracle."""

    mean = numpy.mean(numpy.asarray(medians, float), axis=0)
    best, best_distance = None, 15.0
    for name, centroid in centroids.items():
        if name == exclude:
            continue
        distance = _source_wdist(numpy, mean, centroid)
        if distance < best_distance:
            best, best_distance = name, distance
    return best


def _source_find_collision(
    numpy: Any,
    centroids: Any,
    order: Any,
) -> tuple[Any, Any] | None:
    """Independent collision oracle."""

    worst = None
    worst_distance = 15.0
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            first, second = order[i], order[j]
            if first not in centroids or second not in centroids:
                continue
            distance = _source_wdist(numpy, centroids[first], centroids[second])
            if distance < worst_distance:
                worst_distance = distance
                worst = (first, second)
    return worst


def test_geometry_golden_matches_reference_capture_dimensions() -> None:
    assert grid_calibration.cell_patches(900, 600) == [
        (316, 166, 360, 210),
        (428, 166, 472, 210),
        (540, 166, 584, 210),
        (316, 278, 360, 322),
        (428, 278, 472, 322),
        (540, 278, 584, 322),
        (316, 390, 360, 434),
        (428, 390, 472, 434),
        (540, 390, 584, 434),
    ]
    assert grid_calibration.cell_patches(1920, 1080) == [
        (718, 298, 799, 379),
        (920, 298, 1000, 379),
        (1121, 298, 1202, 379),
        (718, 500, 799, 580),
        (920, 500, 1000, 580),
        (1121, 500, 1202, 580),
        (718, 701, 799, 782),
        (920, 701, 1000, 782),
        (1121, 701, 1202, 782),
    ]


def test_opencv_lab_sampling_golden_is_exact(cv_runtime: tuple[Any, Any]) -> None:
    _, numpy = cv_runtime
    frame = numpy.zeros((600, 900, 3), dtype=numpy.uint8)
    patch_colors = (
        (60, 200, 40),
        (40, 40, 210),
        (250, 250, 250),
        (8, 8, 8),
        (0, 0, 0),
        (255, 255, 255),
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
    )
    for rectangle, bgr in zip(
        grid_calibration.cell_patches(900, 600),
        patch_colors,
        strict=True,
    ):
        x0, y0, x1, y1 = rectangle
        frame[y0:y1, x0:x1] = bgr

    actual = numpy.asarray(grid_calibration.sample_grid(frame))
    expected = numpy.asarray(
        (
            (181, 62, 184),
            (118, 192, 171),
            (251, 128, 128),
            (6, 128, 128),
            (0, 128, 128),
            (255, 128, 128),
            (82, 207, 20),
            (224, 42, 211),
            (136, 208, 195),
        ),
        dtype=numpy.float64,
    )

    assert actual.dtype == numpy.float64
    assert numpy.array_equal(actual, expected)


def test_robust_centroid_golden_preserves_small_sample_mean_and_outlier_rejection(
    cv_runtime: tuple[Any, Any],
) -> None:
    _, numpy = cv_runtime

    # This deliberately distinguishes the short-sample plain-mean branch
    # from the pre-existing public validator's similarly named approximation.
    small = numpy.asarray(
        (
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (255.0, 255.0, 255.0),
        )
    )
    assert numpy.array_equal(
        grid_calibration._robust_centroid(small),
        numpy.asarray((85.0, 85.0, 85.0)),
    )

    with_outlier = numpy.asarray(
        (
            (100.0, 100.0, 100.0),
            (101.0, 100.0, 100.0),
            (99.0, 100.0, 100.0),
            (100.0, 101.0, 99.0),
            (99.0, 100.0, 101.0),
            (101.0, 99.0, 100.0),
            (250.0, 250.0, 250.0),
        )
    )
    assert numpy.array_equal(
        grid_calibration._robust_centroid(with_outlier),
        numpy.asarray((100.0, 100.0, 100.0)),
    )


def test_acceptance_gates_preserve_source_order_and_strict_boundaries(
    cv_runtime: tuple[Any, Any],
) -> None:
    _, numpy = cv_runtime
    base = [numpy.asarray((100.0, 128.0, 128.0)) for _ in range(9)]

    dark = list(base)
    dark[4] = numpy.asarray((39.999, 128.0, 128.0))
    assert grid_calibration.evaluate_frame(dark, dark, 40.0) == (
        False,
        "dark",
        False,
    )

    uniformity_edge = list(base)
    uniformity_edge[4] = numpy.asarray((100.0, 156.0, 128.0))
    assert grid_calibration.evaluate_frame(
        uniformity_edge,
        uniformity_edge,
        40.0,
    ) == (True, None, True)

    not_uniform = list(base)
    not_uniform[4] = numpy.asarray((100.0, 156.001, 128.0))
    assert grid_calibration.evaluate_frame(
        not_uniform,
        not_uniform,
        40.0,
    ) == (False, "not_uniform", False)

    assert grid_calibration.evaluate_frame(base, None, 40.0) == (
        False,
        "moving",
        True,
    )
    previous_at_edge = [median + numpy.asarray((0.0, 10.0, 0.0)) for median in base]
    assert grid_calibration.evaluate_frame(base, previous_at_edge, 40.0) == (
        True,
        None,
        True,
    )
    previous_over_edge = [median + numpy.asarray((0.0, 10.001, 0.0)) for median in base]
    assert grid_calibration.evaluate_frame(base, previous_over_edge, 40.0) == (
        False,
        "moving",
        True,
    )


def test_collision_floor_is_strict_and_ordered(
    cv_runtime: tuple[Any, Any],
) -> None:
    _, numpy = cv_runtime
    order = ("white", "green", "red")
    centroids = {
        "white": numpy.asarray((100.0, 100.0, 100.0)),
        "green": numpy.asarray((100.0, 115.0, 100.0)),
        "red": numpy.asarray((100.0, 114.999, 100.0)),
    }

    # white/green is exactly at the floor and is not a collision. The closest
    # below-floor pair in traversal order is returned.
    assert grid_calibration.find_collision(centroids, order) == ("green", "red")
    assert (
        grid_calibration.looks_like_collected(
            [centroids["white"]] * 9,
            {"green": centroids["green"]},
            "white",
        )
        is None
    )


def test_random_opencv_sampling_differential_matches_reference_body(
    cv_runtime: tuple[Any, Any],
) -> None:
    cv2, numpy = cv_runtime
    rng = numpy.random.default_rng(0x6CA1)

    for _ in range(80):
        width = int(rng.integers(64, 720))
        height = int(rng.integers(64, 720))
        frame = rng.integers(
            0,
            256,
            size=(height, width, 3),
            dtype=numpy.uint8,
        )

        assert grid_calibration.cell_patches(width, height) == _source_cell_patches(
            width,
            height,
        )
        actual = grid_calibration.sample_grid(frame)
        expected = _source_sample_grid(cv2, numpy, frame)
        assert len(actual) == len(expected) == 9
        assert all(numpy.array_equal(got, want) for got, want in zip(actual, expected, strict=True))


def test_random_acceptance_and_collision_differential_matches_reference_body(
    cv_runtime: tuple[Any, Any],
) -> None:
    _, numpy = cv_runtime
    rng = random.Random(0xC0111DE)

    for _ in range(500):
        medians = [
            numpy.asarray(
                (
                    rng.uniform(0.0, 255.0),
                    rng.uniform(0.0, 255.0),
                    rng.uniform(0.0, 255.0),
                )
            )
            for _ in range(9)
        ]
        previous = (
            None
            if rng.random() < 0.25
            else [
                median
                + numpy.asarray(
                    (
                        rng.uniform(-15.0, 15.0),
                        rng.uniform(-15.0, 15.0),
                        rng.uniform(-15.0, 15.0),
                    )
                )
                for median in medians
            ]
        )
        minimum_l = rng.choice((0.0, 40.0, 100.0, 255.0))

        assert grid_calibration.evaluate_frame(
            medians,
            previous,
            minimum_l,
        ) == _source_evaluate_frame(
            numpy,
            medians,
            previous,
            minimum_l,
        )

        names = ("white", "green", "red", "blue", "orange", "yellow")
        centroids = {
            name: numpy.asarray(
                (
                    rng.uniform(0.0, 255.0),
                    rng.uniform(0.0, 255.0),
                    rng.uniform(0.0, 255.0),
                )
            )
            for name in names
            if rng.random() < 0.8
        }
        exclude = rng.choice(names)
        assert grid_calibration.looks_like_collected(
            medians,
            centroids,
            exclude,
        ) == _source_looks_like_collected(
            numpy,
            medians,
            centroids,
            exclude,
        )
        assert grid_calibration.find_collision(
            centroids,
            names,
        ) == _source_find_collision(
            numpy,
            centroids,
            names,
        )

        first = medians[rng.randrange(9)]
        second = medians[rng.randrange(9)]
        assert grid_calibration.wdist(first, second) == _source_wdist(
            numpy,
            first,
            second,
        )


def test_random_robust_centroid_differential_matches_reference_body(
    cv_runtime: tuple[Any, Any],
) -> None:
    _, numpy = cv_runtime
    rng = numpy.random.default_rng(0xCE47)

    for sample_count in range(1, 81):
        for _ in range(10):
            samples = rng.uniform(
                0.0,
                255.0,
                size=(sample_count, 3),
            )
            actual = grid_calibration._robust_centroid(samples)
            expected = _source_robust_centroid(numpy, samples)
            assert numpy.array_equal(actual, expected)


@pytest.mark.parametrize(
    "operation",
    [
        lambda: grid_calibration.sample_grid(object()),
        lambda: grid_calibration.wdist((0, 0, 0), (0, 0, 0)),
        lambda: grid_calibration._robust_centroid(((0, 0, 0),)),
        lambda: grid_calibration.evaluate_frame([], None, 40.0),
        lambda: grid_calibration.looks_like_collected([], {}, None),
        lambda: grid_calibration.find_collision({}, ()),
    ],
)
def test_calibration_operations_fail_closed_without_opencv(
    monkeypatch: pytest.MonkeyPatch,
    operation: Any,
) -> None:
    def unavailable() -> Any:
        raise VisionRuntimeUnavailable("OpenCV unavailable")

    monkeypatch.setattr(grid_calibration, "require_vision_runtime", unavailable)

    with pytest.raises(VisionRuntimeUnavailable, match="OpenCV unavailable"):
        operation()


def test_grid_calibration_constants_match_reference_values() -> None:
    assert grid_calibration.GRID_FRACTION == 0.56
    assert grid_calibration.PATCH_FRACTION == 0.40
    assert grid_calibration.GRID_N == 3
    assert grid_calibration.UNIFORMITY_MAX == 28.0
    assert grid_calibration.STABILITY_MAX == 10.0
    assert grid_calibration.COLLISION_MIN == 15.0
    assert grid_calibration.SAMPLES_PER_FRAME == 9
    assert grid_calibration.FRAMES_PER_COLOR == 5
    assert grid_calibration.SAMPLES_PER_COLOR == 45

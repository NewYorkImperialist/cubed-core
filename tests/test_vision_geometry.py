from __future__ import annotations

import itertools

import pytest

numpy = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from cubed_core.vision.geometry import (  # noqa: E402
    DEFAULT_WARP_SPEC,
    GeometryError,
    WarpSpec,
    cell_centers,
    cell_centers_image,
    cell_quads,
    destination_quad,
    order_quad,
    polygon_area,
    quad_iou,
    shrink_quad,
    warp_quad,
)


def test_order_quad_is_permutation_invariant_and_canonical() -> None:
    expected = numpy.asarray(
        ((10.0, 20.0), (90.0, 15.0), (100.0, 80.0), (5.0, 90.0)),
        dtype=numpy.float32,
    )
    for permutation in itertools.permutations(expected.tolist()):
        result = order_quad(permutation)
        assert result.dtype == numpy.float32
        assert numpy.array_equal(result, expected)


def test_order_quad_matches_established_angular_sort_on_randomized_valid_quads() -> None:
    def reference(quad: object) -> numpy.ndarray:
        points = numpy.asarray(quad, numpy.float32)
        center = points.mean(axis=0)
        angles = numpy.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        return points[numpy.argsort(angles)]

    rng = numpy.random.default_rng(20260723)
    base = numpy.asarray(((-1, -1), (1, -1), (1, 1), (-1, 1)), numpy.float32)
    for _ in range(500):
        linear = rng.normal(size=(2, 2))
        if numpy.linalg.det(linear) < 0:
            linear[:, 0] *= -1
        linear += numpy.eye(2) * 2.0
        center = rng.uniform(-500, 500, size=2)
        quad = base @ linear.T * rng.uniform(5, 100) + center
        quad += rng.normal(0, 0.02, size=(4, 2))
        shuffled = quad[rng.permutation(4)]
        assert numpy.array_equal(order_quad(shuffled), reference(shuffled))


@pytest.mark.parametrize(
    "quad",
    [
        [[0, 0], [1, 0], [1, 1]],
        [[0, 0], [1, 0], [1, 1], [1, 1]],
        [[0, 0], [4, 0], [1, 1], [0, 4]],
        [[0, 0], [1, 0], [1, float("nan")], [0, 1]],
    ],
)
def test_order_quad_rejects_invalid_or_nonconvex_geometry(quad: object) -> None:
    with pytest.raises(GeometryError):
        order_quad(quad)


def test_default_warp_geometry_has_row_major_cell_contract() -> None:
    assert DEFAULT_WARP_SPEC.margin == 12
    assert DEFAULT_WARP_SPEC.cell_size == 24
    assert numpy.array_equal(
        destination_quad(),
        numpy.asarray(((12, 12), (84, 12), (84, 84), (12, 84)), numpy.float32),
    )
    assert numpy.array_equal(
        cell_centers(),
        numpy.asarray(
            (
                (24, 24),
                (48, 24),
                (72, 24),
                (24, 48),
                (48, 48),
                (72, 48),
                (24, 72),
                (48, 72),
                (72, 72),
            ),
            numpy.float32,
        ),
    )
    quads = cell_quads()
    assert quads.shape == (9, 4, 2)
    assert numpy.array_equal(quads[0], [[12, 12], [36, 12], [36, 36], [12, 36]])
    assert numpy.array_equal(quads[-1], [[60, 60], [84, 60], [84, 84], [60, 84]])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"output_size": 11},
        {"output_size": 95, "face_size": 72},
        {"face_size": 73},
        {"grid_size": 0},
        {"supersample": 9},
    ],
)
def test_warp_spec_rejects_ambiguous_grid_geometry(kwargs: dict[str, int]) -> None:
    with pytest.raises(GeometryError):
        WarpSpec(**kwargs)


def test_cell_centers_project_back_to_axis_aligned_source_face() -> None:
    source = numpy.asarray(((20, 30), (92, 30), (92, 102), (20, 102)), numpy.float32)
    projected = cell_centers_image(source)
    expected = cell_centers() + numpy.asarray((8, 18), numpy.float32)
    assert numpy.allclose(projected, expected, atol=1e-5)


def test_warp_quad_preserves_synthetic_cell_centers() -> None:
    frame = numpy.zeros((130, 130, 3), dtype=numpy.uint8)
    source = numpy.asarray(((20, 20), (92, 20), (92, 92), (20, 92)), numpy.float32)
    colors = []
    for row in range(3):
        for column in range(3):
            color = (20 + row * 60, 30 + column * 60, 40 + (row + column) * 30)
            colors.append(color)
            frame[
                20 + row * 24 : 20 + (row + 1) * 24,
                20 + column * 24 : 20 + (column + 1) * 24,
            ] = color

    crop = warp_quad(frame, source)

    assert crop.shape == (96, 96, 3)
    for center, color in zip(cell_centers().astype(int), colors, strict=True):
        x, y = center
        assert numpy.allclose(crop[y, x], color, atol=1)


def test_ordered_warp_rejects_reflected_correspondence() -> None:
    frame = numpy.zeros((100, 100, 3), dtype=numpy.uint8)
    counterclockwise = numpy.asarray(
        ((10, 10), (10, 90), (90, 90), (90, 10)),
        numpy.float32,
    )
    with pytest.raises(GeometryError, match="clockwise"):
        warp_quad(frame, counterclockwise, ordered=True)


def test_shrink_area_and_quad_iou_are_deterministic() -> None:
    outer = numpy.asarray(((0, 0), (100, 0), (100, 100), (0, 100)), numpy.float32)
    inner = shrink_quad(outer, 0.25)
    assert numpy.array_equal(
        inner,
        [[12.5, 12.5], [87.5, 12.5], [87.5, 87.5], [12.5, 87.5]],
    )
    assert polygon_area(outer) == 10_000.0
    assert polygon_area(inner) == 5_625.0
    assert quad_iou(outer, inner) == pytest.approx(0.5625)
    disjoint = outer + numpy.asarray((200, 0), numpy.float32)
    assert quad_iou(outer, disjoint) == 0.0

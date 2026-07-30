from __future__ import annotations

import math
from typing import Any


class LabelAssistError(ValueError):
    pass


class LabelAssistUnavailable(LabelAssistError):
    pass


def _runtime() -> tuple[Any, Any]:
    try:
        import cv2
        import numpy
    except (ImportError, OSError) as exc:
        raise LabelAssistUnavailable(
            "PnP assist requires the optional local geometry runtime; "
            "install it with `pip install -e '.[label]'`"
        ) from exc
    return cv2, numpy


def runtime_capability() -> dict[str, Any]:
    try:
        cv2, numpy = _runtime()
    except LabelAssistUnavailable as exc:
        return {
            "enabled": False,
            "status": "disabled",
            "reason": str(exc),
            "execution_host": "api-host-cpu",
        }
    return {
        "enabled": True,
        "status": "available",
        "reason": None,
        "execution_host": "api-host-cpu",
        "opencv_version": str(cv2.__version__),
        "numpy_version": str(numpy.__version__),
    }


def _number(
    value: Any,
    *,
    field: str,
    minimum: float,
    maximum: float,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not minimum <= float(value) <= maximum
    ):
        raise LabelAssistError(f"{field} must be a finite number between {minimum} and {maximum}")
    return float(value)


def _dimension(value: Any, *, field: str) -> int:
    if type(value) is not int or not 1 <= value <= 100_000:
        raise LabelAssistError(f"{field} must be an integer between 1 and 100000")
    return value


def _points(
    value: Any,
    *,
    field: str,
    width: int,
    height: int,
    lengths: set[int],
) -> list[list[float]]:
    if not isinstance(value, list) or len(value) not in lengths:
        rendered = " or ".join(str(length) for length in sorted(lengths))
        raise LabelAssistError(f"{field} must contain exactly {rendered} points")
    points: list[list[float]] = []
    for index, point in enumerate(value):
        if not isinstance(point, list) or len(point) != 2:
            raise LabelAssistError(f"{field}[{index}] must contain x and y")
        points.append(
            [
                _number(
                    point[0],
                    field=f"{field}[{index}][0]",
                    minimum=0,
                    maximum=width,
                ),
                _number(
                    point[1],
                    field=f"{field}[{index}][1]",
                    minimum=0,
                    maximum=height,
                ),
            ]
        )
    return points


def _camera_matrix(value: Any, *, width: int, height: int) -> tuple[list[list[float]], str]:
    if value is None:
        focal = float(max(width, height))
        return (
            [
                [focal, 0.0, width / 2.0],
                [0.0, focal, height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            "image-centered-focal-prior",
        )
    if not isinstance(value, list) or len(value) != 3:
        raise LabelAssistError("K must be a 3 by 3 camera matrix")
    matrix: list[list[float]] = []
    for row_index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != 3:
            raise LabelAssistError("K must be a 3 by 3 camera matrix")
        matrix.append(
            [
                _number(
                    entry,
                    field=f"K[{row_index}][{column_index}]",
                    minimum=-1_000_000,
                    maximum=1_000_000,
                )
                for column_index, entry in enumerate(row)
            ]
        )
    if matrix[0][0] <= 0 or matrix[1][1] <= 0 or abs(matrix[2][2]) < 1e-12:
        raise LabelAssistError("K must have positive focal lengths and a nonzero scale")
    determinant = (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )
    if abs(determinant) < 1e-12:
        raise LabelAssistError("K must be invertible")
    return matrix, "explicit-camera-matrix"


def _validate_labeled_geometry(points: list[list[float]]) -> None:
    for index, point in enumerate(points):
        for previous in points[:index]:
            distance_squared = (point[0] - previous[0]) ** 2 + (point[1] - previous[1]) ** 2
            if distance_squared < 1e-6:
                raise LabelAssistError("labeled_corners must contain distinct points")

    if len(points) == 3:
        point_a, point_b, point_c = points
        signed_area_twice = (point_b[0] - point_a[0]) * (point_c[1] - point_a[1]) - (
            point_b[1] - point_a[1]
        ) * (point_c[0] - point_a[0])
    else:
        center_x = sum(point[0] for point in points) / len(points)
        center_y = sum(point[1] for point in points) / len(points)
        ordered = sorted(
            points,
            key=lambda point: math.atan2(point[1] - center_y, point[0] - center_x),
        )
        signed_area_twice = sum(
            point[0] * ordered[(index + 1) % len(ordered)][1]
            - point[1] * ordered[(index + 1) % len(ordered)][0]
            for index, point in enumerate(ordered)
        )
    if abs(signed_area_twice) < 1e-3:
        raise LabelAssistError("labeled_corners must define a nondegenerate face")


def extrapolate_request(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LabelAssistError("request body must be an object")
    allowed = {"labeled_corners", "width", "height", "K", "pins", "shrink"}
    extra = value.keys() - allowed
    if extra:
        raise LabelAssistError(f"request contains unsupported field {sorted(extra)[0]}")
    if not {"labeled_corners", "width", "height"} <= value.keys():
        raise LabelAssistError("request requires labeled_corners, width, and height")
    width = _dimension(value["width"], field="width")
    height = _dimension(value["height"], field="height")
    labeled_corners = _points(
        value["labeled_corners"],
        field="labeled_corners",
        width=width,
        height=height,
        lengths={3, 4},
    )
    _validate_labeled_geometry(labeled_corners)
    matrix, intrinsics_source = _camera_matrix(
        value.get("K"),
        width=width,
        height=height,
    )
    pins_value = value.get("pins", [])
    if not isinstance(pins_value, list) or len(pins_value) > 8:
        raise LabelAssistError("pins must be an array with at most 8 items")
    pins: list[dict[str, Any]] = []
    seen_vertices: set[int] = set()
    for index, pin in enumerate(pins_value):
        if not isinstance(pin, dict) or set(pin) != {"xy", "vertex"}:
            raise LabelAssistError(f"pins[{index}] must contain only xy and vertex")
        vertex = pin["vertex"]
        if type(vertex) is not int or not 0 <= vertex <= 7:
            raise LabelAssistError(f"pins[{index}].vertex must be an integer from 0 to 7")
        if vertex in seen_vertices:
            raise LabelAssistError(f"pins repeats cube vertex {vertex}")
        seen_vertices.add(vertex)
        xy = _points(
            [pin["xy"]],
            field=f"pins[{index}].xy",
            width=width,
            height=height,
            lengths={1},
        )[0]
        pins.append({"xy": xy, "vertex": vertex})
    shrink = _number(
        value.get("shrink", 1.0),
        field="shrink",
        minimum=0.1,
        maximum=1.5,
    )
    try:
        return extrapolate(
            labeled_corners,
            matrix,
            pins=pins,
            shrink=shrink,
            intrinsics_source=intrinsics_source,
        )
    except (LabelAssistError, LabelAssistUnavailable):
        raise
    except Exception as exc:
        raise LabelAssistError("PnP could not solve the labeled face") from exc


def _order_quad(corners: Any, numpy: Any) -> Any:
    corners = numpy.asarray(corners, float)
    center = corners.mean(0)
    corners = corners[
        numpy.argsort(numpy.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0]))
    ]
    corners = numpy.roll(corners, -int(numpy.argmin(corners.sum(1))), axis=0)
    edge_a = corners[1] - corners[0]
    diagonal = corners[2] - corners[0]
    edge_b = corners[3] - corners[0]
    area = (
        edge_a[0] * diagonal[1]
        - edge_a[1] * diagonal[0]
        + diagonal[0] * edge_b[1]
        - diagonal[1] * edge_b[0]
    )
    if area > 0:
        corners = corners[[0, 3, 2, 1]]
    return corners


def _complete_square(three: Any, matrix: Any, numpy: Any) -> list[float]:
    point_a, point_b, point_c = [numpy.asarray(point, float) for point in three]
    affine = point_a + point_c - point_b
    inverse = numpy.linalg.inv(numpy.asarray(matrix, float))
    ray_a = inverse @ numpy.array([point_a[0], point_a[1], 1.0])
    ray_b = inverse @ numpy.array([point_b[0], point_b[1], 1.0])
    ray_c = inverse @ numpy.array([point_c[0], point_c[1], 1.0])
    aa, bb, cc = ray_a @ ray_a, ray_b @ ray_b, ray_c @ ray_c
    ab, ac, bc = ray_a @ ray_b, ray_a @ ray_c, ray_b @ ray_c

    def c_of(a: float) -> float | None:
        denominator = a * ac - bc
        return (a * ab - bb) / denominator if abs(denominator) > 1e-12 else None

    def residual(a: float) -> float | None:
        c_value = c_of(a)
        if c_value is None:
            return None
        return (a * a * aa - 2 * a * ab) - (c_value * c_value * cc - 2 * c_value * bc)

    solutions: list[tuple[float, Any]] = []
    grid = numpy.linspace(0.05, 25.0, 4000)
    previous_a, previous_residual = grid[0], residual(float(grid[0]))
    for a_value in grid[1:]:
        current_a = float(a_value)
        current_residual = residual(current_a)
        if (
            previous_residual is not None
            and current_residual is not None
            and previous_residual * current_residual < 0
        ):
            low, high, low_residual = previous_a, current_a, previous_residual
            for _ in range(60):
                middle = 0.5 * (low + high)
                middle_residual = residual(middle)
                if middle_residual is None:
                    break
                if low_residual * middle_residual <= 0:
                    high = middle
                else:
                    low, low_residual = middle, middle_residual
            solved_a = 0.5 * (low + high)
            solved_c = c_of(solved_a)
            if solved_c is not None and 0.2 < solved_a < 5 and 0.2 < solved_c < 5:
                corner_3d = solved_a * ray_a + solved_c * ray_c - ray_b
                if corner_3d[2] > 1e-6:
                    projected = numpy.asarray(matrix, float) @ corner_3d
                    projected = projected[:2] / projected[2]
                    solutions.append((float(numpy.linalg.norm(projected - affine)), projected))
        previous_a, previous_residual = current_a, current_residual
    if not solutions:
        return affine.tolist()
    solutions.sort(key=lambda solution: solution[0])
    return solutions[0][1].tolist()


def extrapolate(
    labeled_corners: list[list[float]],
    matrix: list[list[float]],
    *,
    pins: list[dict[str, Any]] | None = None,
    shrink: float = 1.0,
    intrinsics_source: str,
) -> dict[str, Any]:
    cv2, numpy = _runtime()
    cube_front = numpy.array(
        [
            [-0.5, 0.5, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, -0.5, 0.0],
            [-0.5, -0.5, 0.0],
        ],
        float,
    )
    labeled = numpy.asarray(labeled_corners, float)
    inferred_corner = None
    if len(labeled) == 3:
        inferred_corner = _complete_square(labeled, matrix, numpy)
        labeled = numpy.vstack([labeled, numpy.asarray(inferred_corner, float)])
    ordered = _order_quad(labeled, numpy)
    camera_matrix = numpy.asarray(matrix, float)
    ok, rotation, translation = cv2.solvePnP(
        cube_front,
        ordered,
        camera_matrix,
        numpy.zeros(5),
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not ok:
        return {
            "ok": False,
            "reason": "PnP could not solve the labeled face",
            "intrinsics_source": intrinsics_source,
        }

    best_extrusion: tuple[float, int] | None = None
    for sign in (1, -1):
        candidate = numpy.vstack([cube_front, cube_front + [0, 0, sign]])
        camera = (cv2.Rodrigues(rotation)[0] @ candidate.T + translation).T
        depth_delta = float(camera[4:, 2].mean() - camera[:4, 2].mean())
        if best_extrusion is None or depth_delta > best_extrusion[0]:
            best_extrusion = (depth_delta, sign)
    assert best_extrusion is not None
    cube = numpy.vstack([cube_front, cube_front + [0, 0, best_extrusion[1]]])

    if pins:
        ids = [0, 1, 2, 3] + [int(pin["vertex"]) for pin in pins]
        points = numpy.array(
            list(ordered) + [[float(pin["xy"][0]), float(pin["xy"][1])] for pin in pins],
            float,
        )
        refined, next_rotation, next_translation = cv2.solvePnP(
            cube[ids],
            points,
            camera_matrix,
            numpy.zeros(5),
            rvec=rotation.copy(),
            tvec=translation.copy(),
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if refined:
            rotation, translation = next_rotation, next_translation

    projected = cv2.projectPoints(
        cube,
        rotation,
        translation,
        camera_matrix,
        numpy.zeros(5),
    )[0].reshape(-1, 2)
    if not (
        numpy.isfinite(projected).all()
        and numpy.isfinite(rotation).all()
        and numpy.isfinite(translation).all()
    ):
        return {
            "ok": False,
            "reason": "PnP produced non-finite geometry",
            "intrinsics_source": intrinsics_source,
        }
    rotation_matrix = cv2.Rodrigues(rotation)[0]
    cube_center = cube.mean(0)
    face_indices = {
        "labeled": [0, 1, 2, 3],
        "top": [0, 1, 5, 4],
        "right": [1, 2, 6, 5],
        "bottom": [3, 2, 6, 7],
        "left": [0, 3, 7, 4],
    }
    faces = []
    for name, vertices in face_indices.items():
        face_center = cube[vertices].mean(0)
        outward_camera = rotation_matrix @ (face_center - cube_center)
        face_camera = rotation_matrix @ face_center + translation.ravel()
        if name != "labeled" and float(numpy.dot(outward_camera, face_camera)) >= 0:
            continue
        corners = projected[vertices].copy()
        if shrink != 1.0:
            center = corners.mean(0)
            corners = center + (corners - center) * shrink
        faces.append(
            {
                "name": name,
                "vertices": vertices,
                "corners": corners.tolist(),
            }
        )
    return {
        "ok": True,
        "intrinsics_source": intrinsics_source,
        "inferred_corner": inferred_corner,
        "rvec": rotation.ravel().tolist(),
        "tvec": translation.ravel().tolist(),
        "wireframe": projected.tolist(),
        "faces": faces,
    }

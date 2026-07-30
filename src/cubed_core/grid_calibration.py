"""Grid-target color calibration.

The user supplies the color label. This module performs only the existing
geometric OpenCV-Lab sampling and acceptance checks:

* a centered square with side ``0.56 * min(frame_width, frame_height)``;
* a 3x3 grid;
* the central 40% patch of each cell; and
* median OpenCV-encoded Lab per patch.

The lazy vision-runtime guard keeps OpenCV optional at package import time and
fails closed when calibration is requested without it.
"""

from __future__ import annotations

from typing import Any

from .vision._runtime import require_vision_runtime

# Geometry contract constants.
GRID_FRACTION = 0.56
PATCH_FRACTION = 0.40
GRID_N = 3

# Acceptance thresholds.
UNIFORMITY_MAX = 28.0
STABILITY_MAX = 10.0
COLLISION_MIN = 15.0
SAMPLES_PER_FRAME = GRID_N * GRID_N
FRAMES_PER_COLOR = 5
SAMPLES_PER_COLOR = FRAMES_PER_COLOR * SAMPLES_PER_FRAME

# Chroma-weighted distance. A tuple keeps module import dependency-free; NumPy
# applies the same elementwise weights as the reference NumPy calculation.
_W = (0.15, 1.0, 1.0)


def wdist(a: Any, b: Any) -> float:
    """Return the chroma-weighted OpenCV-Lab distance."""

    _, numpy = require_vision_runtime()
    d = numpy.asarray(a, float) - numpy.asarray(b, float)
    return float(numpy.sqrt(((d**2) * _W).sum()))


def _robust_centroid(samples: Any) -> Any:
    """Return the calibration path's outlier-robust centroid."""

    _, numpy = require_vision_runtime()
    arr = numpy.asarray(samples, dtype=numpy.float64)
    if len(arr) < 6:
        return arr.mean(axis=0)
    med = numpy.median(arr, axis=0)
    d = numpy.sqrt((((arr - med) ** 2) * _W).sum(axis=1))
    mad = numpy.median(d) + 1e-6
    keep = d <= max(3.0 * mad, 10.0)
    inliers = arr[keep]
    return inliers.mean(axis=0) if len(inliers) >= 3 else med


def cell_patches(width: float, height: float) -> list[tuple[int, int, int, int]]:
    """Return the nine sampled patch rectangles in row-major order."""

    side = GRID_FRACTION * min(width, height)
    gx0 = width / 2.0 - side / 2.0
    gy0 = height / 2.0 - side / 2.0
    cell = side / GRID_N
    patch = PATCH_FRACTION * cell
    rects = []
    for r in range(GRID_N):
        for c in range(GRID_N):
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


def sample_grid(frame_bgr: Any) -> list[Any]:
    """Return median OpenCV Lab for each grid-cell patch, row-major."""

    cv2, numpy = require_vision_runtime()
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    h, w = lab.shape[:2]
    medians = []
    for x0, y0, x1, y1 in cell_patches(w, h):
        patch = lab[y0:y1, x0:x1].reshape(-1, 3).astype(numpy.float64)
        medians.append(numpy.median(patch, axis=0))
    return medians


def evaluate_frame(
    medians: Any,
    prev_medians: Any | None,
    min_l: float,
) -> tuple[bool, str | None, bool]:
    """Apply the brightness, uniformity, and stability gates in order."""

    _, numpy = require_vision_runtime()
    if any(m[0] < min_l for m in medians):
        return False, "dark", False
    for i in range(len(medians)):
        for j in range(i + 1, len(medians)):
            if wdist(medians[i], medians[j]) > UNIFORMITY_MAX:
                return False, "not_uniform", False
    if prev_medians is None:
        return False, "moving", True
    mean_d = float(numpy.mean([wdist(m, p) for m, p in zip(medians, prev_medians, strict=False)]))
    if mean_d > STABILITY_MAX:
        return False, "moving", True
    return True, None, True


def looks_like_collected(
    medians: Any,
    centroids: Any,
    exclude: Any,
) -> Any | None:
    """Return the closest already-collected color below the collision floor."""

    _, numpy = require_vision_runtime()
    mean = numpy.mean(numpy.asarray(medians, float), axis=0)
    best, best_d = None, COLLISION_MIN
    for name, centroid in centroids.items():
        if name == exclude:
            continue
        distance = wdist(mean, centroid)
        if distance < best_d:
            best, best_d = name, distance
    return best


def find_collision(centroids: Any, order: Any) -> tuple[Any, Any] | None:
    """Return the closest ordered centroid pair below the collision floor."""

    require_vision_runtime()
    worst = None
    worst_d = COLLISION_MIN
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            a, b = order[i], order[j]
            if a not in centroids or b not in centroids:
                continue
            distance = wdist(centroids[a], centroids[b])
            if distance < worst_d:
                worst_d = distance
                worst = (a, b)
    return worst

"""
Extract sticker colors from a detected cube face.

Two modes:
1. Direct sampling from the original frame using face quad corners (preferred).
2. Sampling from a warped face image (legacy).
"""

import os

import cv2
import numpy as np

# Canonical unit-square corners (TL, TR, BR, BL) and the 9 cell centers at
# parametric ((2c+1)/6, (2r+1)/6), row-major. Used by the PERSPECTIVE-GRID
# sampling mode (see _perspective_cell_centers).
_UNIT_SQUARE = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
_CANON_CENTERS = np.array(
    [[[(2 * col + 1) / 6.0, (2 * row + 1) / 6.0]]
     for row in range(3) for col in range(3)],
    dtype=np.float32,
)  # shape (9, 1, 2)


def _perspective_grid_enabled(flag):
    """Resolve the perspective-grid flag: explicit param wins; else env
    CUBED_PERSPECTIVE_GRID. DEFAULT OFF (byte-identical bilinear path)."""
    if flag is not None:
        return bool(flag)
    return os.environ.get("CUBED_PERSPECTIVE_GRID", "").lower() in (
        "1", "true", "yes", "on")


def _perspective_cell_centers(tl, tr, br, bl):
    """Image-space centers of the 9 cells via the planar homography from the
    canonical unit square to the (ordered, shrunk) image corners.

    A cube face is a planar quad; under perspective projection it maps to the
    image by a homography, so mapping the canonical cell centers through that
    homography places them ON-facelet even under foreshortening — unlike a
    bilinear blend of the 4 corners, which drifts off the far/bottom cells.
    Returns a (9, 2) float array, row-major (row*3 + col).
    """
    H = cv2.getPerspectiveTransform(
        _UNIT_SQUARE, np.array([tl, tr, br, bl], dtype=np.float32))
    return cv2.perspectiveTransform(_CANON_CENTERS, H).reshape(9, 2)


def sample_stickers_from_quad(frame, corners, shrink=0.35, sample_radius=6,
                              perspective_grid=None):
    """
    Sample 9 sticker colors directly from the original frame using
    the face quadrilateral — no perspective warp needed.

    Computes the center of each sticker cell, then samples a small region
    around each point. Two center-placement modes:
      - bilinear (DEFAULT): bilinear interpolation of the 4 quad corners.
      - perspective-grid (flag-gated): planar homography from the canonical
        unit square, geometrically correct under foreshortening.

    Args:
        frame: Original BGR image
        corners: np.array of shape (4, 2) in TL, TR, BR, BL order
        shrink: Fraction to shrink quad inward (avoids borders/fingers)
        sample_radius: Pixel radius around each sticker center to average
        perspective_grid: None (default) -> read env CUBED_PERSPECTIVE_GRID
            (default OFF); True/False to force the mode. OFF is byte-identical
            to the legacy bilinear path.

    Returns:
        List of 9 LAB color values (numpy arrays), row-major order
    """
    from detect.face_warper import order_corners, shrink_corners

    ordered = order_corners(corners)
    if shrink > 0:
        ordered = shrink_corners(ordered, shrink)

    tl, tr, br, bl = ordered

    # PERSPECTIVE-GRID center placement (flag-gated, default OFF). When off,
    # `pgrid_centers` stays None and the bilinear path below is byte-identical.
    pgrid_centers = None
    if _perspective_grid_enabled(perspective_grid):
        pgrid_centers = _perspective_cell_centers(tl, tr, br, bl)

    # Convert to LAB
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    h, w = lab.shape[:2]

    colors = []
    for row in range(3):
        for col in range(3):
            if pgrid_centers is not None:
                px, py = pgrid_centers[row * 3 + col]
            else:
                # Parametric position within the quad
                # t: vertical (0=top, 1=bottom), s: horizontal (0=left, 1=right)
                t = (2 * row + 1) / 6.0
                s = (2 * col + 1) / 6.0

                # Bilinear interpolation of quad corners
                px = ((1 - t) * (1 - s) * tl[0] + (1 - t) * s * tr[0] +
                      t * s * br[0] + t * (1 - s) * bl[0])
                py = ((1 - t) * (1 - s) * tl[1] + (1 - t) * s * tr[1] +
                      t * s * br[1] + t * (1 - s) * bl[1])

            # Sample region around this point
            cx, cy = int(round(px)), int(round(py))
            sy = max(0, cy - sample_radius)
            sx = max(0, cx - sample_radius)
            ey = min(h, cy + sample_radius)
            ex = min(w, cx + sample_radius)

            if ey <= sy or ex <= sx:
                colors.append(np.array([128, 128, 128], dtype=np.float64))
                continue

            region = lab[sy:ey, sx:ex]
            colors.append(region.mean(axis=(0, 1)))

    return colors


def sample_stickers_bgr_from_quad(frame, corners, shrink=0.20, sample_radius=8):
    """Same as sample_stickers_from_quad but returns BGR values for visualization."""
    from detect.face_warper import order_corners, shrink_corners

    ordered = order_corners(corners)
    if shrink > 0:
        ordered = shrink_corners(ordered, shrink)

    tl, tr, br, bl = ordered
    h, w = frame.shape[:2]

    colors = []
    for row in range(3):
        for col in range(3):
            t = (2 * row + 1) / 6.0
            s = (2 * col + 1) / 6.0
            px = ((1 - t) * (1 - s) * tl[0] + (1 - t) * s * tr[0] +
                  t * s * br[0] + t * (1 - s) * bl[0])
            py = ((1 - t) * (1 - s) * tl[1] + (1 - t) * s * tr[1] +
                  t * s * br[1] + t * (1 - s) * bl[1])

            cx, cy = int(round(px)), int(round(py))
            sy = max(0, cy - sample_radius)
            sx = max(0, cx - sample_radius)
            ey = min(h, cy + sample_radius)
            ex = min(w, cx + sample_radius)

            if ey <= sy or ex <= sx:
                colors.append((128, 128, 128))
                continue

            region = frame[sy:ey, sx:ex]
            avg = region.mean(axis=(0, 1)).astype(np.uint8)
            colors.append(tuple(int(c) for c in avg))

    return colors


def border_score_from_quad(frame, corners, shrink=0.20, sample_radius=4):
    """
    Score a detection by checking for dark borders between sticker positions.

    On a stickered cube, the black plastic between stickers creates dark regions
    at the midpoints between adjacent sticker centers. This function samples
    those midpoints and compares their brightness to the sticker centers.

    Args:
        frame: Original BGR image
        corners: np.array of shape (4, 2)
        shrink: Fraction to shrink quad inward
        sample_radius: Pixel radius for sampling border brightness

    Returns:
        Float score 0-1. Higher = more border-like dark gaps detected.
        A good stickered face typically scores > 0.5.
    """
    from detect.face_warper import order_corners, shrink_corners

    ordered = order_corners(corners)
    if shrink > 0:
        ordered = shrink_corners(ordered, shrink)

    tl, tr, br, bl = ordered
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    def interp(t, s):
        px = (1 - t) * (1 - s) * tl[0] + (1 - t) * s * tr[0] + t * s * br[0] + t * (1 - s) * bl[0]
        py = (1 - t) * (1 - s) * tl[1] + (1 - t) * s * tr[1] + t * s * br[1] + t * (1 - s) * bl[1]
        return int(round(px)), int(round(py))

    def sample_brightness(cx, cy):
        sy = max(0, cy - sample_radius)
        sx = max(0, cx - sample_radius)
        ey = min(h, cy + sample_radius)
        ex = min(w, cx + sample_radius)
        if ey <= sy or ex <= sx:
            return 128
        return float(gray[sy:ey, sx:ex].mean())

    # Sample sticker centers (9 points)
    sticker_vals = []
    for row in range(3):
        for col in range(3):
            t = (2 * row + 1) / 6.0
            s = (2 * col + 1) / 6.0
            cx, cy = interp(t, s)
            sticker_vals.append(sample_brightness(cx, cy))

    # Sample border midpoints between adjacent stickers
    # Horizontal borders: between col j and col j+1 for each row (6 points)
    # Vertical borders: between row i and row i+1 for each col (6 points)
    border_vals = []
    for row in range(3):
        for j in range(2):  # between col j and j+1
            t = (2 * row + 1) / 6.0
            s = (j + 1) / 3.0
            cx, cy = interp(t, s)
            border_vals.append(sample_brightness(cx, cy))

    for i in range(2):  # between row i and i+1
        for col in range(3):
            t = (i + 1) / 3.0
            s = (2 * col + 1) / 6.0
            cx, cy = interp(t, s)
            border_vals.append(sample_brightness(cx, cy))

    if not border_vals or not sticker_vals:
        return 0.0

    avg_sticker = np.mean(sticker_vals)

    # Count how many border points are darker than the average sticker brightness
    threshold = avg_sticker * 0.65  # border should be significantly darker
    dark_count = sum(1 for v in border_vals if v < threshold)

    return dark_count / len(border_vals)


def grid_line_score(warped_face):
    """
    Detect black grid lines between stickers on a stickered cube.

    Computes 1D brightness profiles (averaged across rows and columns),
    then looks for two dark valleys in each direction, roughly evenly
    spaced — the signature of a 3x3 sticker grid with black borders.

    Returns:
        (h_lines, v_lines): Number of dark valleys found in each direction.
        A good warp has (2, 2) = 4 total.
    """
    gray = cv2.cvtColor(warped_face, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape

    def count_valleys(profile, min_depth=15):
        """Find valleys (local minima darker than neighbors) in a 1D profile."""
        # Smooth to reduce noise
        kernel = np.ones(7) / 7
        smoothed = np.convolve(profile, kernel, mode='same')

        # Find local minima in the middle 80% (avoid edges)
        margin = len(smoothed) // 10
        search = smoothed[margin:-margin]

        valleys = []
        for i in range(1, len(search) - 1):
            if search[i] < search[i - 1] and search[i] < search[i + 1]:
                # Check depth: valley should be darker than its surroundings
                left_peak = max(search[max(0, i - 15):i])
                right_peak = max(search[i + 1:min(len(search), i + 16)])
                depth = min(left_peak, right_peak) - search[i]
                if depth > min_depth:
                    valleys.append(i + margin)

        return len(valleys)

    # Row profile (average brightness per row) — finds horizontal lines
    row_profile = gray.mean(axis=1)
    h_lines = count_valleys(row_profile)

    # Column profile — finds vertical lines
    col_profile = gray.mean(axis=0)
    v_lines = count_valleys(col_profile)

    return h_lines + v_lines


def is_valid_warp(warped_face, min_grid_lines=2):
    """
    Check if a warped face shows the black grid pattern of a stickered cube.

    Requires at least min_grid_lines dark valleys (out of ~4 possible)
    in the brightness profiles.
    """
    return grid_line_score(warped_face) >= min_grid_lines


def extract_sticker_bgr(warped_face, margin=0.15, sample_ratio=0.4):
    """
    Extract the center sample from each grid cell as a BGR value.
    """
    h, w = warped_face.shape[:2]
    cell_h = h / 3
    cell_w = w / 3

    colors = []
    for row in range(3):
        for col in range(3):
            y_start = int(row * cell_h)
            x_start = int(col * cell_w)
            sample_h = int(cell_h * sample_ratio)
            sample_w = int(cell_w * sample_ratio)
            cy = y_start + int(cell_h / 2)
            cx = x_start + int(cell_w / 2)
            sy = max(0, cy - sample_h // 2)
            sx = max(0, cx - sample_w // 2)
            ey = min(h, sy + sample_h)
            ex = min(w, sx + sample_w)

            region = warped_face[sy:ey, sx:ex]
            avg_color = region.mean(axis=(0, 1)).astype(np.uint8)
            colors.append(tuple(int(c) for c in avg_color))

    return colors

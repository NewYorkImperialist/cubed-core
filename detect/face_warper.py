"""
Perspective warp for detected cube faces.

Takes 4 face corner points and warps the face region to a flat square,
producing a normalized top-down view of the 3x3 sticker grid.
"""

import cv2
import numpy as np


def order_corners(corners):
    """
    Order 4 corner points as: top-left, top-right, bottom-right, bottom-left.

    Uses centroid-relative angle sorting to handle arbitrary rotations.

    Args:
        corners: np.array of shape (4, 2)

    Returns:
        np.array of shape (4, 2) in TL, TR, BR, BL order
    """
    centroid = corners.mean(axis=0)
    angles = np.arctan2(corners[:, 1] - centroid[1], corners[:, 0] - centroid[0])
    sorted_idx = np.argsort(angles)
    sorted_pts = corners[sorted_idx]

    # After angle sort: roughly left, bottom, right, top (counterclockwise from -pi)
    # We want TL, TR, BR, BL — rearrange based on y-position
    # Top two points have smaller y, bottom two have larger y
    top_idx = np.argsort(sorted_pts[:, 1])[:2]
    bottom_idx = np.argsort(sorted_pts[:, 1])[2:]

    top_pts = sorted_pts[top_idx]
    bottom_pts = sorted_pts[bottom_idx]

    # Within top/bottom, left has smaller x
    tl = top_pts[np.argmin(top_pts[:, 0])]
    tr = top_pts[np.argmax(top_pts[:, 0])]
    bl = bottom_pts[np.argmin(bottom_pts[:, 0])]
    br = bottom_pts[np.argmax(bottom_pts[:, 0])]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def shrink_corners(corners, factor=0.15):
    """
    Shrink corner points toward the centroid to crop tighter.

    Detected face corners often extend slightly beyond the actual face,
    capturing fingers and background at the edges. Shrinking inward
    focuses the warp on the sticker grid.

    Args:
        corners: np.array of shape (4, 2)
        factor: fraction to shrink (0.15 = 15% inward from each edge)

    Returns:
        np.array of shape (4, 2), shrunk corners
    """
    centroid = corners.mean(axis=0)
    return corners + (centroid - corners) * factor


def warp_face(frame, corners, output_size=150, shrink=0.15):
    """
    Warp a detected face quad to a flat square image.

    Args:
        frame: BGR image (numpy array)
        corners: np.array of shape (4, 2) — corner points from face detection
        output_size: side length of output square in pixels
        shrink: fraction to shrink corners inward (reduces finger/edge noise)

    Returns:
        Warped face image (output_size x output_size, BGR)
    """
    ordered = order_corners(corners)
    if shrink > 0:
        ordered = shrink_corners(ordered, shrink)

    dst = np.array([
        [0, 0],
        [output_size - 1, 0],
        [output_size - 1, output_size - 1],
        [0, output_size - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(frame, M, (output_size, output_size))

    return warped

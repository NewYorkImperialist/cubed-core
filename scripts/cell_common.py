"""Geometry shared by the maintained CPU and CUDA cell readers."""

import cv2
import numpy as np

CROP = 96          # crop side
FACE = 72          # face square inside the crop
MARGIN = (CROP - FACE) // 2
CELL = FACE // 3   # 24


def order_quad(quad):
    """Angular sort around the centroid — pipeline corner convention."""
    pts = np.asarray(quad, np.float32)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    return pts[np.argsort(ang)]


_DST = np.array([[MARGIN, MARGIN], [MARGIN + FACE, MARGIN],
                 [MARGIN + FACE, MARGIN + FACE], [MARGIN, MARGIN + FACE]],
                np.float32)
_SS = 3  # supersample factor: warp at 3x then INTER_AREA down so cell
         # samples are area means, not bilinear point-samples (the emission
         # model was fitted on full-res area-averaged patches)


def warp_crop(frame, quad, ordered=False):
    """Canonical CROP x CROP BGR crop; face fills the central FACE square.

    ordered=True means quad corners are ALREADY in order_quad order and must
    not be re-sorted — required whenever corner correspondence matters (e.g.
    jittered duplicates: re-sorting jittered corners can flip the angular
    sort's cyclic start near diamond orientations, silently rotating the
    crop 90 deg against its labels).
    """
    oq = np.asarray(quad, np.float32) if ordered else order_quad(quad)
    M = cv2.getPerspectiveTransform(oq, _DST * _SS)
    big = cv2.warpPerspective(frame, M, (CROP * _SS, CROP * _SS))
    return cv2.resize(big, (CROP, CROP), interpolation=cv2.INTER_AREA)


def cell_centers():
    """(9, 2) crop-space centers, grid pos = row*3 + col."""
    return np.array([[MARGIN + (c + 0.5) * CELL, MARGIN + (r + 0.5) * CELL]
                     for r in range(3) for c in range(3)], np.float32)

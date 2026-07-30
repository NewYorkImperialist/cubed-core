"""Color distance utilities for puzzle-cube sticker classification.

Provides CIEDE2000 perceptual color difference computation, both scalar
and vectorized (numpy). Handles OpenCV LAB <-> standard CIE LAB conversion.

OpenCV LAB: L in [0, 255], a in [0, 255], b in [0, 255]
Standard LAB: L in [0, 100], a in [-128, +127], b in [-128, +127]
"""

import math

import numpy as np


def cv_lab_to_std(lab_cv):
    """Convert OpenCV-scale LAB to standard CIE LAB.

    Works on any shape with last dim = 3.
    """
    lab_cv = np.asarray(lab_cv, dtype=np.float64)
    out = np.empty_like(lab_cv)
    out[..., 0] = lab_cv[..., 0] * (100.0 / 255.0)
    out[..., 1] = lab_cv[..., 1] - 128.0
    out[..., 2] = lab_cv[..., 2] - 128.0
    return out


def ciede2000(lab1, lab2):
    """CIEDE2000 color difference between two standard CIE LAB colors.

    Each input: (3,) array or 3-tuple of (L, a, b) in standard scale.
    Returns float >= 0.

    Reference: Sharma, Wu, Dalal (2005) — "The CIEDE2000 Color-Difference Formula"
    """
    L1, a1, b1 = float(lab1[0]), float(lab1[1]), float(lab1[2])
    L2, a2, b2 = float(lab2[0]), float(lab2[1]), float(lab2[2])

    C1 = math.hypot(a1, b1)
    C2 = math.hypot(a2, b2)
    C_avg = (C1 + C2) * 0.5

    C_avg7 = C_avg**7
    POW25_7 = 6103515625.0  # 25^7
    G = 0.5 * (1.0 - math.sqrt(C_avg7 / (C_avg7 + POW25_7)))

    a1p = a1 * (1.0 + G)
    a2p = a2 * (1.0 + G)

    C1p = math.hypot(a1p, b1)
    C2p = math.hypot(a2p, b2)

    h1p = math.degrees(math.atan2(b1, a1p)) % 360.0
    h2p = math.degrees(math.atan2(b2, a2p)) % 360.0

    dLp = L2 - L1
    dCp = C2p - C1p

    C1pC2p = C1p * C2p
    dhp_abs = abs(h1p - h2p)

    if C1pC2p == 0.0:
        dhp = 0.0
    elif dhp_abs <= 180.0:
        dhp = h2p - h1p
    elif h2p - h1p > 180.0:
        dhp = h2p - h1p - 360.0
    else:
        dhp = h2p - h1p + 360.0

    dHp = 2.0 * math.sqrt(C1pC2p) * math.sin(math.radians(dhp * 0.5))

    Lp_avg = (L1 + L2) * 0.5
    Cp_avg = (C1p + C2p) * 0.5

    if C1pC2p == 0.0:
        hp_avg = h1p + h2p
    elif dhp_abs <= 180.0:
        hp_avg = (h1p + h2p) * 0.5
    elif h1p + h2p < 360.0:
        hp_avg = (h1p + h2p + 360.0) * 0.5
    else:
        hp_avg = (h1p + h2p - 360.0) * 0.5

    T = (
        1.0
        - 0.17 * math.cos(math.radians(hp_avg - 30.0))
        + 0.24 * math.cos(math.radians(2.0 * hp_avg))
        + 0.32 * math.cos(math.radians(3.0 * hp_avg + 6.0))
        - 0.20 * math.cos(math.radians(4.0 * hp_avg - 63.0))
    )

    Lp_diff = Lp_avg - 50.0
    SL = 1.0 + 0.015 * Lp_diff**2 / math.sqrt(20.0 + Lp_diff**2)
    SC = 1.0 + 0.045 * Cp_avg
    SH = 1.0 + 0.015 * Cp_avg * T

    Cp_avg7 = Cp_avg**7
    RC = 2.0 * math.sqrt(Cp_avg7 / (Cp_avg7 + POW25_7))
    d_theta = 30.0 * math.exp(-(((hp_avg - 275.0) / 25.0) ** 2))
    RT = -math.sin(math.radians(2.0 * d_theta)) * RC

    term_L = dLp / SL
    term_C = dCp / SC
    term_H = dHp / SH

    return math.sqrt(term_L**2 + term_C**2 + term_H**2 + RT * term_C * term_H)


def ciede2000_batch(labs, refs):
    """Vectorized CIEDE2000: (N, 3) queries vs (M, 3) references.

    All inputs must be in standard CIE LAB.
    Returns (N, M) distance matrix.
    """
    # Broadcast: labs (N,1,3), refs (1,M,3)
    L1 = labs[:, np.newaxis, 0]
    a1 = labs[:, np.newaxis, 1]
    b1 = labs[:, np.newaxis, 2]
    L2 = refs[np.newaxis, :, 0]
    a2 = refs[np.newaxis, :, 1]
    b2 = refs[np.newaxis, :, 2]

    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    C_avg = (C1 + C2) * 0.5

    C_avg7 = C_avg**7
    POW25_7 = 6103515625.0
    G = 0.5 * (1.0 - np.sqrt(C_avg7 / (C_avg7 + POW25_7)))

    a1p = a1 * (1.0 + G)
    a2p = a2 * (1.0 + G)

    C1p = np.sqrt(a1p**2 + b1**2)
    C2p = np.sqrt(a2p**2 + b2**2)

    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0

    dLp = L2 - L1
    dCp = C2p - C1p

    C1pC2p = C1p * C2p
    dhp_abs = np.abs(h1p - h2p)

    dhp = np.where(
        C1pC2p == 0.0,
        0.0,
        np.where(
            dhp_abs <= 180.0,
            h2p - h1p,
            np.where(h2p - h1p > 180.0, h2p - h1p - 360.0, h2p - h1p + 360.0),
        ),
    )

    dHp = 2.0 * np.sqrt(np.maximum(C1pC2p, 0.0)) * np.sin(np.radians(dhp * 0.5))

    Lp_avg = (L1 + L2) * 0.5
    Cp_avg = (C1p + C2p) * 0.5

    hp_sum = h1p + h2p
    hp_avg = np.where(
        C1pC2p == 0.0,
        hp_sum,
        np.where(
            dhp_abs <= 180.0,
            hp_sum * 0.5,
            np.where(
                hp_sum < 360.0, (hp_sum + 360.0) * 0.5, (hp_sum - 360.0) * 0.5
            ),
        ),
    )

    T = (
        1.0
        - 0.17 * np.cos(np.radians(hp_avg - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * hp_avg))
        + 0.32 * np.cos(np.radians(3.0 * hp_avg + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * hp_avg - 63.0))
    )

    Lp_diff = Lp_avg - 50.0
    SL = 1.0 + 0.015 * Lp_diff**2 / np.sqrt(20.0 + Lp_diff**2)
    SC = 1.0 + 0.045 * Cp_avg
    SH = 1.0 + 0.015 * Cp_avg * T

    Cp_avg7 = Cp_avg**7
    RC = 2.0 * np.sqrt(Cp_avg7 / (Cp_avg7 + POW25_7))
    d_theta = 30.0 * np.exp(-(((hp_avg - 275.0) / 25.0) ** 2))
    RT = -np.sin(np.radians(2.0 * d_theta)) * RC

    term_L = dLp / SL
    term_C = dCp / SC
    term_H = dHp / SH

    return np.sqrt(term_L**2 + term_C**2 + term_H**2 + RT * term_C * term_H)


def lab_chroma(lab_std):
    """Chroma (saturation) from standard CIE LAB: sqrt(a² + b²).

    White ≈5, colored stickers ≈30-60.
    """
    return math.hypot(float(lab_std[1]), float(lab_std[2]))


def lab_hue_angle(lab_std):
    """Hue angle in degrees [0, 360) from standard CIE LAB.

    Returns 0.0 for achromatic colors (chroma near zero).
    Red ≈30, orange ≈58, yellow ≈103, green ≈142, blue ≈265.
    """
    a, b = float(lab_std[1]), float(lab_std[2])
    if abs(a) < 1e-6 and abs(b) < 1e-6:
        return 0.0
    return math.degrees(math.atan2(b, a)) % 360.0

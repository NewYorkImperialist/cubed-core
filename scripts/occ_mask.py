"""Per-pixel occlusion mask for classical cell reads.

Low-chroma, non-bright pixels are treated as skin or gray occlusion. Pixels far
from every calibrated centroid are treated as gaps, glare, or other occlusion.
Bright low-chroma pixels remain eligible for the white face.
"""
import numpy as np
import cv2

C_SKIN = 52.0      # chroma below this (and dark) = skin/gray, not a sticker
L_WHITE = 185.0    # above this, low chroma is the WHITE sticker, not skin
T_FAR = 58.0       # Lab dist to nearest centroid above this = occluder/gap


def skin_mask(lab, c_skin=C_SKIN, l_white=L_WHITE):
    """Return low-chroma, non-bright pixels that may be skin or gray.

    Centroid-aware recovery in ``masked_cell_labs`` prevents nearby blue
    sticker pixels from being rejected solely by this coarse mask.
    """
    L = lab[..., 0]
    chroma = np.hypot(lab[..., 1] - 128.0, lab[..., 2] - 128.0)
    return (chroma < c_skin) & (L < l_white)


def occlusion_mask(lab, cents, c_skin=C_SKIN, l_white=L_WHITE, t_far=T_FAR):
    """lab: HxWx3 float (OpenCV Lab). cents: (6,3). -> HxW bool, True=occlusion."""
    flat = lab.reshape(-1, 3)
    pd = np.linalg.norm(flat[:, None, :] - cents[None], axis=2).min(1)
    far = (pd > t_far).reshape(lab.shape[:2])
    return skin_mask(lab, c_skin, l_white) | far


def masked_cell_labs(crop, half=8):
    """Return per-cell median Lab values and visible-pixel confidence.

    Low-chroma, non-bright pixels are excluded from each central cell window.
    Confidence is the remaining visible-pixel fraction. If too few pixels
    remain, the function falls back to the unmasked window median.
    """
    from cell_common import cell_centers
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    sk = skin_mask(lab)
    cc = cell_centers().astype(int)
    out = np.zeros((9, 3)); conf = np.zeros(9)
    for i, (x, y) in enumerate(cc):
        win = lab[y - half:y + half, x - half:x + half].reshape(-1, 3)
        m = sk[y - half:y + half, x - half:x + half].reshape(-1)
        conf[i] = 1.0 - float(m.mean()) if m.size else 0.0
        valid = win[~m]
        out[i] = (np.median(valid, 0) if len(valid) >= 8
                  else (np.median(win, 0) if win.size else [0., 128., 128.]))
    return out, conf

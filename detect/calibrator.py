"""
Color classification for puzzle-cube stickers.

Two modes:
1. Hardcoded centroids (ColorClassifier) — works without calibration
2. Calibrated from known state (CalibratedClassifier) — uses the scramble
   state + observed frames to learn per-color models under current lighting

The calibration flow:
  - User applies scramble (known state)
  - During inspection, camera sees 2-3 faces per frame
  - Since we know the state, each visible sticker is a labeled sample
  - Over a few frames we accumulate samples for all 6 colors
  - Build per-color centroids that match current lighting/angle
"""

import numpy as np

from detect.color_distance import ciede2000, cv_lab_to_std, lab_chroma, lab_hue_angle
from detect.color_extractor import sample_stickers_from_quad

# Center sticker (index 4) always identifies the face
CENTER_COLOR_TO_FACE = {
    "white": "up", "red": "right", "green": "front",
    "yellow": "down", "orange": "left", "blue": "back",
}

FACE_TO_CENTER_COLOR = {v: k for k, v in CENTER_COLOR_TO_FACE.items()}

ALL_FACES = ["up", "right", "front", "down", "left", "back"]
ALL_COLORS = ["white", "red", "green", "yellow", "orange", "blue"]

# Minimum aspect ratio for a valid face detection (width/height or height/width)
MIN_ASPECT_RATIO = 0.35
MIN_FACE_AREA = 5000  # pixels²

# Minimum L value for a valid sticker sample (below = black border/plastic)
MIN_STICKER_L = 40

# Fallback LAB centroids for uncalibrated operation
# (OpenCV LAB: L=0-255, a=0-255, b=0-255).
DEFAULT_CENTROIDS = {
    "white":  np.array([230, 126, 132], dtype=np.float64),
    "yellow": np.array([220, 115, 185], dtype=np.float64),
    "red":    np.array([100, 175, 155], dtype=np.float64),
    "orange": np.array([170, 160, 180], dtype=np.float64),
    "blue":   np.array([ 90, 125,  95], dtype=np.float64),
    "green":  np.array([140, 100, 150], dtype=np.float64),
}

# Known confusion pairs and the LAB-derived feature that separates them.
# White is the only achromatic color → chroma separates it from any chromatic.
# For two chromatic colors → hue angle is the cleanest separator.
CONFUSION_PAIRS = {
    frozenset({"white", "yellow"}):  "chroma",
    frozenset({"white", "orange"}):  "chroma",
    frozenset({"white", "green"}):   "chroma",
    frozenset({"red", "orange"}):    "hue",
    frozenset({"yellow", "orange"}): "hue",
    frozenset({"yellow", "green"}):  "hue",
    frozenset({"blue", "green"}):    "hue",
}

# Minimum feature separation between centroids to trust disambiguation
_MIN_CHROMA_SEP = 10.0   # chroma units
_MIN_HUE_SEP = 10.0      # degrees
# Minimum margin (fraction of half-separation) to mark as confident
_MIN_DISAMBIG_MARGIN = 0.3


def _angular_distance(a, b):
    """Shortest angular distance in degrees between two hue angles."""
    d = abs(a - b) % 360.0
    return d if d <= 180.0 else 360.0 - d


def disambiguate_pair(lab_std, best, second, centroids_std):
    """Resolve an ambiguous CIEDE2000 classification using pair-specific features.

    When CIEDE2000 can't confidently distinguish two colors, this checks if
    the pair is a known confusion pair and uses the single feature (chroma or
    hue angle) that cleanly separates them.

    Args:
        lab_std: observed LAB in standard CIE scale
        best: best color name from CIEDE2000
        second: second-best color name
        centroids_std: dict of color_name -> standard CIE LAB centroid

    Returns:
        (color_name, confident) or None if disambiguation not applicable.
    """
    pair = frozenset({best, second})
    feature = CONFUSION_PAIRS.get(pair)
    if feature is None:
        return None

    std_best = centroids_std.get(best)
    std_second = centroids_std.get(second)
    if std_best is None or std_second is None:
        return None

    if feature == "chroma":
        obs_val = lab_chroma(lab_std)
        val_best = lab_chroma(std_best)
        val_second = lab_chroma(std_second)
        min_sep = _MIN_CHROMA_SEP
    else:  # hue
        obs_val = lab_hue_angle(lab_std)
        val_best = lab_hue_angle(std_best)
        val_second = lab_hue_angle(std_second)
        min_sep = _MIN_HUE_SEP

    if feature == "hue":
        separation = _angular_distance(val_best, val_second)
        if separation < min_sep:
            return None
        dist_to_best = _angular_distance(obs_val, val_best)
        dist_to_second = _angular_distance(obs_val, val_second)
    else:
        separation = abs(val_best - val_second)
        if separation < min_sep:
            return None
        dist_to_best = abs(obs_val - val_best)
        dist_to_second = abs(obs_val - val_second)

    chosen = best if dist_to_best <= dist_to_second else second

    half_sep = separation / 2.0
    midpoint_dist = abs(dist_to_best - dist_to_second) / 2.0
    margin = midpoint_dist / half_sep if half_sep > 0 else 0.0
    confident = margin >= _MIN_DISAMBIG_MARGIN

    return chosen, confident


def filter_detections(faces):
    """
    Filter out garbage detections (too thin, too small).

    Args:
        faces: list of dicts with 'corners' (4x2 array), 'confidence'

    Returns:
        Filtered list of face dicts.
    """
    good = []
    for f in faces:
        corners = f["corners"]
        # Compute edge lengths
        widths = [
            np.linalg.norm(corners[1] - corners[0]),
            np.linalg.norm(corners[2] - corners[3]),
        ]
        heights = [
            np.linalg.norm(corners[3] - corners[0]),
            np.linalg.norm(corners[2] - corners[1]),
        ]
        avg_w = (widths[0] + widths[1]) / 2
        avg_h = (heights[0] + heights[1]) / 2

        if avg_w < 1 or avg_h < 1:
            continue

        aspect = min(avg_w, avg_h) / max(avg_w, avg_h)
        area = avg_w * avg_h

        if aspect < MIN_ASPECT_RATIO:
            continue
        if area < MIN_FACE_AREA:
            continue

        good.append(f)
    return good


class ColorClassifier:
    """Classify sticker colors using hardcoded LAB centroids."""

    def __init__(self, centroids=None):
        self.centroids = centroids or dict(DEFAULT_CENTROIDS)
        self.centroids_std = {
            name: cv_lab_to_std(lab) for name, lab in self.centroids.items()
        }

    def classify_sticker(self, lab_value):
        """Classify a single LAB value to a color name.
        Returns None if the sample is too dark (border/plastic)."""
        lab = np.array(lab_value, dtype=np.float64)
        if lab[0] < MIN_STICKER_L:
            return None

        lab_std = cv_lab_to_std(lab)
        best_color = None
        best_dist = float("inf")
        second_color = None
        second_dist = float("inf")

        for color_name, centroid_std in self.centroids_std.items():
            dist = ciede2000(lab_std, centroid_std)
            if dist < best_dist:
                second_color, second_dist = best_color, best_dist
                best_color, best_dist = color_name, dist
            elif dist < second_dist:
                second_color, second_dist = color_name, dist

        # Attempt disambiguation when CIEDE2000 is uncertain
        if second_color and best_dist > 1e-6:
            ratio = second_dist / best_dist
            if ratio < 1.5:
                result = disambiguate_pair(
                    lab_std, best_color, second_color, self.centroids_std)
                if result is not None:
                    return result[0]

        return best_color

    def classify_face(self, lab_colors):
        """Classify all 9 stickers of a face."""
        return [self.classify_sticker(c) for c in lab_colors]

    def identify_face(self, lab_colors):
        """Identify which face this is based on the center sticker."""
        center_color = self.classify_sticker(lab_colors[4])
        return CENTER_COLOR_TO_FACE.get(center_color)


# 4 possible face-quad-to-face grid rotations
GRID_ROTATIONS = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8],  # 0°
    [6, 3, 0, 7, 4, 1, 8, 5, 2],  # 90° CW
    [8, 7, 6, 5, 4, 3, 2, 1, 0],  # 180°
    [2, 5, 8, 1, 4, 7, 0, 3, 6],  # 270° CW
]


# Chroma-weighted LAB distance weights (L downweighted; a/b dominate — matches
# _chroma_dist). Used for outlier rejection when estimating centroids.
_CENTROID_W = np.array([0.15, 1.0, 1.0])


def _robust_centroid(samples):
    """Reject chroma-weighted MAD outliers, then average the inliers.

    Falls back to the plain mean when there are too few samples to estimate
    spread.
    """
    arr = np.asarray(samples, dtype=np.float64)
    if len(arr) < 6:
        return arr.mean(axis=0)
    med = np.median(arr, axis=0)
    d = np.sqrt((((arr - med) ** 2) * _CENTROID_W).sum(axis=1))
    mad = np.median(d) + 1e-6
    keep = d <= max(3.0 * mad, 10.0)
    inliers = arr[keep]
    return inliers.mean(axis=0) if len(inliers) >= 3 else med


def refresh_palette_from_glance(base_centroids, observed_labeled, alpha=0.5,
                                min_samples=6):
    """Blend observed colors toward robust session-start samples.

    Colors without enough observations retain their input centroids.
    """
    obs = {}
    for lab, cname in observed_labeled:
        if cname in base_centroids:
            obs.setdefault(cname, []).append(np.asarray(lab, dtype=np.float64))
    pal = {c: np.asarray(v, dtype=np.float64).copy() for c, v in base_centroids.items()}
    for c, v in obs.items():
        if len(v) >= min_samples:
            pal[c] = (1.0 - alpha) * pal[c] + alpha * _robust_centroid(v)
    return pal


def palette_residual(centroids, observed_labeled):
    """Validity check: mean chroma-weighted distance of labeled observations to a
    palette. Small => stored calibration still valid; large => fit lighting / re-sweep."""
    ds = []
    for lab, cname in observed_labeled:
        if cname in centroids:
            d = np.asarray(lab, dtype=np.float64) - np.asarray(centroids[cname], dtype=np.float64)
            ds.append(float(np.sqrt(((d ** 2) * _CENTROID_W).sum())))
    return float(np.mean(ds)) if ds else float("inf")


class AdaptiveCentroids:
    """Track per-color appearance drift with a gated online EMA.

    Calibration is set once at the start, but a color's appearance drifts as the cube
    moves (lighting/angle/exposure). Dark colors suffer most: at low luminance their
    chrominance (a/b) is noisy and can drift as far as the gap to a neighbouring color.
    After each confirmed state, relabel that frame's stickers and nudge
    each color's centroid toward its outlier-robust observed value via EMA, gated so a
    wild read (occlusion/motion blur) can't corrupt the centroid.

    Bootstrap from the start calibration; keep `alpha` small so a single bad frame
    barely moves the estimate. Use `.centroids` for scoring; `.snapshot()` for a frozen
    copy (e.g. to score a segment with the centroids adapted up to that point in time).
    """

    def __init__(self, centroids, alpha=0.2, gate=35.0):
        self.centroids = {c: np.asarray(v, dtype=np.float64).copy() for c, v in centroids.items()}
        self.alpha = alpha
        self.gate = gate

    def _cdist(self, a, b):
        return float(np.sqrt((((np.asarray(a) - np.asarray(b)) ** 2) * _CENTROID_W).sum()))

    def update(self, labeled):
        """labeled: iterable of (lab, color_name). Outlier-robust mean per color this
        frame, then a gated EMA step toward it. Returns the number of colors updated."""
        by = {}
        for lab, col in labeled:
            if lab is None:
                continue
            lab = np.asarray(lab, dtype=np.float64)
            if lab[0] >= MIN_STICKER_L:
                by.setdefault(col, []).append(lab)
        n = 0
        for col, labs in by.items():
            if col not in self.centroids:
                continue
            obs = _robust_centroid(labs)
            if self._cdist(obs, self.centroids[col]) <= self.gate:
                self.centroids[col] = (1.0 - self.alpha) * self.centroids[col] + self.alpha * obs
                n += 1
        return n

    def snapshot(self):
        return {c: v.copy() for c, v in self.centroids.items()}


class CalibrationCollector:
    """Collect labeled color samples from frames with a known cube state.

    Two-pass calibration:
      Pass 1 (centers only): Collect center stickers — no rotation needed,
              face identity gives the color directly.
      Pass 2 (all stickers): Using pass-1 centroids, re-identify faces and
              find grid rotations, then collect all visible stickers with a
              minimum rotation-score filter to reject bad fits.
    """

    def __init__(self, known_state):
        """
        Args:
            known_state: dict mapping face_name -> list of 9 color strings
        """
        self.known_state = known_state
        self.samples = {color: [] for color in ALL_COLORS}
        self._classifier = ColorClassifier()
        self._frames = []  # stored for pass 2

    def add_frame(self, frame, faces):
        """Store frame data and collect center sticker samples (pass 1).

        Args:
            frame: BGR image
            faces: list of dicts with 'corners' (already filtered)
        """
        frame_faces = []
        for f in faces:
            lab_colors = sample_stickers_from_quad(frame, f["corners"])
            face_name = self._classifier.identify_face(lab_colors)
            if face_name is None:
                continue

            # Center sticker (pos 4) is always the face's center color
            center_lab = lab_colors[4]
            if center_lab[0] >= MIN_STICKER_L:
                center_color = FACE_TO_CENTER_COLOR[face_name]
                self.samples[center_color].append(
                    np.array(center_lab, dtype=np.float64))

            frame_faces.append((f["corners"], lab_colors, face_name))

        self._frames.append(frame_faces)

    def add_observations(self, face_observations):
        """Accept pre-sampled face observations (for sticker pipeline).

        Uses classification-based scoring against all 6 faces x 4 rotations.
        Classification works well here because relative ranking (nearest centroid)
        is robust even with rough hardcoded centroids. After calibration,
        the move detector switches to distance-based scoring for precision.

        Args:
            face_observations: list of dicts with 'lab_colors' (list of 9 LAB values)
        """
        frame_faces = []
        for obs in face_observations:
            lab_colors = obs["lab_colors"]

            # Score against all 6 faces × 4 rotations to find best match
            best_face = None
            best_score = -1
            best_rot = None
            for face_name in ALL_FACES:
                expected = self.known_state[face_name]
                for rot in GRID_ROTATIONS:
                    score = 0
                    for i in range(9):
                        if lab_colors[i][0] < MIN_STICKER_L:
                            continue
                        classified = self._classifier.classify_sticker(lab_colors[i])
                        if classified == expected[rot[i]]:
                            score += 1
                    if score > best_score:
                        best_score = score
                        best_face = face_name
                        best_rot = rot

            if best_face is None or best_score < 4:
                continue

            # Only collect samples for stickers that individually classify
            # to the expected color. This prevents contamination from
            # grid position errors or face merges in angle clustering.
            expected = self.known_state[best_face]
            for i in range(9):
                if lab_colors[i][0] < MIN_STICKER_L:
                    continue
                expected_color = expected[best_rot[i]]
                classified = self._classifier.classify_sticker(lab_colors[i])
                if classified == expected_color:
                    self.samples[expected_color].append(
                        np.array(lab_colors[i], dtype=np.float64))

            frame_faces.append((None, lab_colors, best_face))

        self._frames.append(frame_faces)

    def _collect_all_stickers(self, min_rotation_score=5):
        """Pass 2: collect all visible stickers using improved centroids."""
        samples = {color: [] for color in ALL_COLORS}

        for frame_faces in self._frames:
            for _corners, lab_colors, face_name in frame_faces:
                expected = self.known_state[face_name]
                rot, score = self._find_rotation(lab_colors, expected)

                if score < min_rotation_score:
                    continue  # bad fit, skip entire face

                for g in range(9):
                    if lab_colors[g][0] < MIN_STICKER_L:
                        continue
                    face_pos = rot[g]
                    color = expected[face_pos]
                    samples[color].append(
                        np.array(lab_colors[g], dtype=np.float64))

        return samples

    def _find_rotation(self, lab_colors, expected_face):
        """Find which of 4 rotations best matches the known state.

        Returns (best_rotation, best_score).
        """
        best_rot = GRID_ROTATIONS[0]
        best_score = -1
        for rot in GRID_ROTATIONS:
            score = 0
            for g in range(9):
                if lab_colors[g][0] < MIN_STICKER_L:
                    continue
                face_pos = rot[g]
                classified = self._classifier.classify_sticker(lab_colors[g])
                if classified == expected_face[face_pos]:
                    score += 1
            if score > best_score:
                best_score = score
                best_rot = rot
        return best_rot, best_score

    def _determine_face_rotations(self, classifier):
        """Determine the correct grid rotation for each face via majority vote.

        Uses distance-based scoring against expected centroids.
        For each face, the rotation that appears most often wins.

        Returns dict: face_name -> rotation_index (into GRID_ROTATIONS).
        """
        from collections import Counter
        MATCH_DIST = 25.0
        votes = {face: Counter() for face in ALL_FACES}

        for frame_faces in self._frames:
            for _corners, lab_colors, face_name in frame_faces:
                expected = self.known_state[face_name]
                best_idx = 0
                best_score = -float('inf')
                for idx, rot in enumerate(GRID_ROTATIONS):
                    score = 0.0
                    for g in range(9):
                        if lab_colors[g][0] < MIN_STICKER_L:
                            continue
                        expected_color = expected[rot[g]]
                        centroid = classifier.centroids.get(expected_color)
                        if centroid is None:
                            continue
                        dist = classifier._chroma_dist(lab_colors[g], centroid)
                        score += 1.0 - dist / MATCH_DIST
                    if score > best_score:
                        best_score = score
                        best_idx = idx
                if best_score >= 3.0:
                    votes[face_name][best_idx] += 1

        result = {}
        for face_name in ALL_FACES:
            if votes[face_name]:
                result[face_name] = votes[face_name].most_common(1)[0][0]
            else:
                result[face_name] = 0
        return result

    def build_classifier(self, min_confidence=1.5, min_rotation_score=5,
                         use_all_stickers=False):
        """Build a CalibratedClassifier from collected samples.

        By default uses center-only centroids (most reliable). If
        use_all_stickers=True, runs pass 2 to collect all stickers
        using the center-based centroids for rotation detection.

        Also determines the per-face grid rotation from calibration data.

        Returns:
            CalibratedClassifier instance
        """
        # Pass 1 centroids (from centers)
        centroids = {}
        for color in ALL_COLORS:
            if self.samples[color]:
                centroids[color] = _robust_centroid(self.samples[color])
            else:
                centroids[color] = DEFAULT_CENTROIDS[color].copy()

        if use_all_stickers:
            # Update classifier for pass 2
            self._classifier = ColorClassifier(centroids)

            # Pass 2: collect all stickers with better centroids
            pass2_samples = self._collect_all_stickers(min_rotation_score)
            self._pass2_samples = pass2_samples

            # Build final centroids from pass 2 (outlier-robust)
            for color in ALL_COLORS:
                if pass2_samples[color]:
                    centroids[color] = _robust_centroid(pass2_samples[color])

        clf = CalibratedClassifier(centroids, min_confidence)

        # Determine per-face rotations using the built classifier
        face_rotations = self._determine_face_rotations(clf)
        clf.face_rotations = face_rotations

        return clf

    def summary(self):
        """Return a summary string of collected samples."""
        lines = ["  Pass 1 (centers only):"]
        for color in ALL_COLORS:
            n = len(self.samples[color])
            if n > 0:
                arr = np.array(self.samples[color])
                mean = arr.mean(axis=0)
                lines.append(f"    {color:7s}: {n:3d} centers  "
                             f"mean=({mean[0]:.0f},{mean[1]:.0f},{mean[2]:.0f})")
            else:
                lines.append(f"    {color:7s}:   0 centers  (using hardcoded)")

        if hasattr(self, "_pass2_samples"):
            lines.append("  Pass 2 (all stickers, filtered):")
            for color in ALL_COLORS:
                n = len(self._pass2_samples[color])
                if n > 0:
                    arr = np.array(self._pass2_samples[color])
                    mean = arr.mean(axis=0)
                    std = arr.std(axis=0) if n > 1 else np.zeros(3)
                    lines.append(f"    {color:7s}: {n:3d} samples  "
                                 f"mean=({mean[0]:.0f},{mean[1]:.0f},{mean[2]:.0f})  "
                                 f"std=({std[0]:.0f},{std[1]:.0f},{std[2]:.0f})")
                else:
                    lines.append(f"    {color:7s}:   0 samples")

        return "\n".join(lines)


class CalibratedClassifier:
    """Classify sticker colors using calibrated centroids.

    Uses CIEDE2000 perceptual color difference for robust matching
    under varying lighting conditions.

    Returns (color_name, confident) for each sticker.
    Skips occluded stickers (L < threshold).
    Marks uncertain stickers (too close to two centroids).
    """

    def __init__(self, centroids, min_confidence=1.5, max_dist=25.0):
        self.centroids = centroids  # OpenCV LAB scale (kept for L-threshold checks)
        self.min_confidence = min_confidence
        self.max_dist = max_dist  # CIEDE2000 units
        self.face_rotations = {}  # face_name -> rotation_index, set by calibrator

        # Pre-convert centroids to standard CIE LAB for CIEDE2000
        self.centroids_std = {
            name: cv_lab_to_std(lab) for name, lab in centroids.items()
        }

    def _chroma_dist(self, lab, centroid):
        """CIEDE2000 perceptual color difference.

        Accepts OpenCV-scale LAB values. Converts to standard LAB internally.
        """
        return ciede2000(cv_lab_to_std(np.asarray(lab, dtype=np.float64)),
                         cv_lab_to_std(np.asarray(centroid, dtype=np.float64)))

    def classify(self, lab_value):
        """Classify a single LAB value.

        Returns:
            (color_name, confident) — color_name is None if occluded,
            confident is False if the sticker is ambiguous.
        """
        lab = np.array(lab_value, dtype=np.float64)
        if lab[0] < MIN_STICKER_L:
            return None, False

        dists = {}
        for name, centroid in self.centroids.items():
            dists[name] = self._chroma_dist(lab, centroid)

        sorted_colors = sorted(dists, key=lambda c: dists[c])
        best = sorted_colors[0]
        second = sorted_colors[1]

        # Reject samples too far from any known sticker color
        # (catches skin, background, plastic borders)
        if dists[best] > self.max_dist:
            return None, False

        if dists[best] < 1e-6:
            return best, True

        ratio = dists[second] / dists[best]
        if ratio >= self.min_confidence:
            return best, True

        # CIEDE2000 uncertain — attempt pair-specific disambiguation
        lab_std = cv_lab_to_std(lab)
        result = disambiguate_pair(lab_std, best, second, self.centroids_std)
        if result is not None:
            return result

        return best, False

    def classify_face(self, lab_colors):
        """Classify all 9 stickers. Returns list of (color, confident)."""
        return [self.classify(c) for c in lab_colors]

    def identify_face(self, lab_colors):
        """Identify which face this is based on the center sticker.
        Uses chrominance-weighted distance for robustness."""
        center_color, _ = self.classify(lab_colors[4])
        if center_color is None:
            return None
        return CENTER_COLOR_TO_FACE.get(center_color)

"""
Move detection via pattern matching against candidate states.

Given a known cube state (from the scramble) and camera observations
(LAB colors from pose-detected face regions), detect which move was made
by matching observed color patterns against all possible next states.

Key insights:
- Cube colors are well-separated in LAB space (~50+ units apart)
- Position-by-position scoring: +1 match, -1 mismatch, 0 uncertain
- An "unknown" observation against a calibrated expected color is a MISMATCH
  (we know it's NOT that color, even if we can't name what it IS)
- Bad face detections (spanning 2 faces) naturally score low → ignored
- 4 grid rotations handle unknown camera-to-face orientation
"""

import copy
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from detect.color_distance import ciede2000, cv_lab_to_std

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.cube import Cube

FACE_SIDES = ["right", "up", "left", "front", "down", "back"]
SLICES = ["m", "s", "e"]
CUBE_ROTATIONS = ["x", "x'", "y", "y'", "z", "z'"]

FACE_REV = {"right": "R", "up": "U", "left": "L", "front": "F", "down": "D", "back": "B"}
SLICE_REV = {"m": "M", "s": "S", "e": "E"}
WIDE_REV = {"right": "r", "up": "u", "left": "l", "front": "f", "down": "d", "back": "b"}

ALL_FACES = ["up", "right", "front", "down", "left", "back"]

# 3x3 grid rotation indices (handles camera-to-face orientation)
GRID_ROTATIONS = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8],  # 0°
    [6, 3, 0, 7, 4, 1, 8, 5, 2],  # 90° CW
    [8, 7, 6, 5, 4, 3, 2, 1, 0],  # 180°
    [2, 5, 8, 1, 4, 7, 0, 3, 6],  # 270° CW
]

# Only score center-cross positions (avoid edge contamination)
CENTER_CROSS = [1, 3, 4, 5, 7]

# CIEDE2000 scoring normalization — similar cube colors are ~20-25 dE apart
MAX_DIST = 25.0

# Minimum L value for a valid sticker sample (below = black border/plastic)
MIN_STICKER_L = 40


def generate_candidates(cube):
    """Generate all possible single-move next states (face moves only)."""
    candidates = []

    for side in FACE_SIDES:
        letter = FACE_REV[side]
        for method, suffix in [(cube.move, ""), (cube.move_prime, "'"), (cube.move_two, "2")]:
            c = copy.deepcopy(cube)
            getattr(c, method.__name__)(side)
            candidates.append((letter + suffix, c))

    return candidates


def generate_candidates_depth2(cube):
    """Generate all possible 2-move sequences."""
    candidates = []
    depth1 = generate_candidates(cube)
    for notation1, cube1 in depth1:
        for notation2, cube2 in generate_candidates(cube1):
            candidates.append((f"{notation1} {notation2}", cube2))
    return candidates


def _score_face_rotation(obs_labs, expected_face, classifier):
    """Score observed LAB values against expected face using distance to expected centroids.

    Position-by-position scoring (used by calibrator for rotation pinning).

    Score: sum of (1 - dist/MAX_DIST) * weight for each visible sticker.
    Close match → positive, clear mismatch → negative.
    """
    score = 0.0
    n_scored = 0

    for i in range(9):
        lab = obs_labs[i]
        if lab[0] < MIN_STICKER_L:
            continue

        expected_color = expected_face[i]
        centroid = classifier.centroids.get(expected_color)
        if centroid is None:
            continue

        dist = classifier._chroma_dist(lab, centroid)
        n_scored += 1
        weight = 1.0 if i in CENTER_CROSS else 0.5
        score += weight * (1.0 - dist / MAX_DIST)

    return score, n_scored


def _score_face_hungarian(obs_labs, expected_face, classifier):
    """Score using optimal 1-to-1 assignment (Hungarian algorithm).

    Finds the matching between observed stickers and expected face colors
    that minimizes total chrominance distance. No grid positions needed —
    handles rotation ambiguity and position errors automatically.

    This is the 2-way detection: the cube state (graph) constrains which
    color each observed sticker should be, resolving red/orange and
    white/yellow ambiguities.

    Returns (score, n_matched).
    """
    # Collect valid (non-occluded) observations
    valid = []
    for i in range(9):
        if obs_labs[i][0] >= MIN_STICKER_L:
            valid.append(obs_labs[i])

    if not valid:
        return -999.0, 0

    n_obs = len(valid)

    # Build cost matrix: cost[i][j] = chroma distance from obs[i] to expected color at position j
    # Rectangular: n_obs rows × 9 columns
    cost = np.full((n_obs, 9), MAX_DIST * 2, dtype=np.float64)
    for i, lab in enumerate(valid):
        for j in range(9):
            centroid = classifier.centroids.get(expected_face[j])
            if centroid is not None:
                cost[i][j] = classifier._chroma_dist(lab, centroid)

    # Optimal assignment: each observed sticker matched to one expected position
    row_ind, col_ind = linear_sum_assignment(cost)

    # Convert to score: sum of (1 - dist/MAX_DIST)
    score = 0.0
    for r, c in zip(row_ind, col_ind):
        score += 1.0 - cost[r][c] / MAX_DIST

    return score, len(row_ind)


def score_face(obs_labs, expected_face, classifier):
    """Score one observation against one expected face.
    Uses Hungarian matching (no rotations needed)."""
    return _score_face_hungarian(obs_labs, expected_face, classifier)


def find_best_rotation(obs_labs, expected_face, classifier):
    """Find the grid rotation that best aligns obs with expected face.

    Returns (best_rotation_index, best_score).
    """
    best_idx = 0
    best_score = -999
    for idx, rot in enumerate(GRID_ROTATIONS):
        rotated = [obs_labs[rot[i]] for i in range(9)]
        s, _ = _score_face_rotation(rotated, expected_face, classifier)
        if s > best_score:
            best_score = s
            best_idx = idx
    return best_idx, best_score


def pin_and_score(obs_list, current_state, candidate_state, classifier):
    """Score a candidate using face assignment and rotations pinned to current state.

    1. Assign observations to faces via Hungarian matching against current_state
    2. For each face, find the best grid rotation against current_state
    3. Score the candidate at those pinned positions (no optimization)

    This prevents overfitting: candidates can't shop for favorable face
    assignments or grid rotations. Only the expected colors change.

    Returns (total_score, n_faces_scored).
    """
    # Step 1: assign observations to faces using current state
    aligned = assign_faces_hungarian(obs_list, current_state, classifier)
    if not aligned:
        return -999.0, 0

    # Group by face (take best observation per face for duplicates)
    face_obs = {}
    for face_name, obs_labs in aligned:
        if face_name not in face_obs:
            face_obs[face_name] = []
        face_obs[face_name].append(obs_labs)

    total = 0.0
    n_faces = 0

    for face_name, obs_list_for_face in face_obs.items():
        best_face_score = -999.0

        for obs_labs in obs_list_for_face:
            # Step 2: pin grid rotation against current state
            rot_idx, _ = find_best_rotation(
                obs_labs, current_state[face_name], classifier)
            rot = GRID_ROTATIONS[rot_idx]
            rotated = [obs_labs[rot[i]] for i in range(9)]

            # Step 3: score candidate at pinned positions
            s, n = _score_face_rotation(
                rotated, candidate_state[face_name], classifier)
            if s > best_face_score:
                best_face_score = s

        if best_face_score > -999.0:
            total += best_face_score
            n_faces += 1

    return total, n_faces


CENTER_COLOR_TO_FACE = {
    "white": "up", "red": "right", "green": "front",
    "yellow": "down", "orange": "left", "blue": "back",
}

# Ordered color list for indexing into distance tables
COLOR_LIST = ["white", "yellow", "red", "orange", "blue", "green"]
COLOR_INDEX = {c: i for i, c in enumerate(COLOR_LIST)}


# --- Numeric state representation for fast beam search ---
# State = (54,) int8 array: 6 faces × 9 stickers, values 0-5 (color indices)
# Layout: positions 0-8 = ALL_FACES[0], 9-17 = ALL_FACES[1], etc.

def state_to_array(state):
    """Convert dict-of-lists state to (54,) int8 array."""
    arr = np.empty(54, dtype=np.int8)
    for i, face in enumerate(ALL_FACES):
        for j, color in enumerate(state[face]):
            arr[i * 9 + j] = COLOR_INDEX[color]
    return arr


def array_to_state(arr):
    """Convert (54,) int8 array back to dict-of-lists state."""
    state = {}
    for i, face in enumerate(ALL_FACES):
        state[face] = [COLOR_LIST[arr[i * 9 + j]] for j in range(9)]
    return state


def _compute_move_perms():
    """Precompute permutation arrays for all 18 face moves.

    Uses a Cube with unique labels per sticker to track where each
    sticker ends up after each move.

    Returns dict: notation -> (54,) int permutation array.
    Applying a move: new_state = old_state[perm]
    """
    # Label each sticker uniquely
    base = Cube()
    for i, face in enumerate(ALL_FACES):
        for j in range(9):
            base.state[face][j] = f"{i}_{j}"

    perms = {}
    for side in FACE_SIDES:
        letter = FACE_REV[side]
        for method_name, suffix in [("move", ""), ("move_prime", "'"), ("move_two", "2")]:
            c = copy.deepcopy(base)
            getattr(c, method_name)(side)

            perm = np.empty(54, dtype=np.intp)
            for fi, face in enumerate(ALL_FACES):
                for j in range(9):
                    src_face, src_pos = c.state[face][j].split("_")
                    perm[fi * 9 + j] = int(src_face) * 9 + int(src_pos)

            perms[letter + suffix] = perm

    return perms


# Computed once at import time
MOVE_PERMS = _compute_move_perms()
# List of (notation, perm) for iteration
MOVE_LIST = list(MOVE_PERMS.items())


def assign_faces_by_center(obs_list, classifier):
    """Assign observations to faces by classifying the center sticker.

    The center sticker (index 4) always identifies the face — it never
    moves during any face turn. This is deterministic and prevents
    candidates from "shopping" for favorable face assignments.

    Returns list of (face_name, obs_labs) tuples.
    """
    aligned = []
    for obs_labs in obs_list:
        center_lab = obs_labs[4]
        color, confident = classifier.classify(center_lab)
        if color is None:
            continue
        face_name = CENTER_COLOR_TO_FACE.get(color)
        if face_name is not None:
            aligned.append((face_name, obs_labs))
    return aligned


def precompute_frame(obs_list, classifier):
    """Precompute distance tables for one frame's observations.

    For each observation (side), compute the CIEDE2000 distance from each
    of the 9 sticker positions to each of the 6 color centroids.

    No face identity is assigned — scoring determines which candidate face
    each observation best matches.

    Returns list of (9,6) ndarrays (one per valid observation).
    dist_table[sticker_idx, color_idx] = CIEDE2000 distance (or MAX_DIST*2 if occluded).
    """
    from detect.color_distance import ciede2000_batch, cv_lab_to_std

    centroids_std = np.array(
        [classifier.centroids_std[c] for c in COLOR_LIST], dtype=np.float64
    )  # (6, 3) in standard LAB

    results = []
    for obs_labs in obs_list:
        # Build distance table: (9, 6) via vectorized CIEDE2000
        dist_table = np.full((9, 6), MAX_DIST * 2, dtype=np.float64)
        all_labs_cv = np.array(obs_labs, dtype=np.float64)  # (9, 3)
        valid_mask = all_labs_cv[:, 0] >= MIN_STICKER_L
        if valid_mask.any():
            valid_labs_std = cv_lab_to_std(all_labs_cv[valid_mask])  # (N, 3)
            dist_table[valid_mask] = ciede2000_batch(valid_labs_std, centroids_std)
            results.append(dist_table)

    return results


def precompute_cluster(obs_frames, classifier):
    """Precompute distance tables for all frames in a cluster.

    Returns list of precomputed frames (one per frame that has valid observations).
    Each element is a list of (9,6) dist_table arrays.
    """
    precomputed = []
    for obs_list in obs_frames:
        frame_data = precompute_frame(obs_list, classifier)
        if frame_data:
            precomputed.append(frame_data)
    return precomputed


# Precomputed rotation arrays for vectorized scoring
_ROT_ARRAYS = [np.array(r, dtype=np.intp) for r in GRID_ROTATIONS]
_ARANGE9 = np.arange(9)


def _build_face_indices_from_states(states):
    """Extract per-face color index arrays from stacked numeric states.

    states: (N, 54) int8 array
    Returns dict: face_name -> (N, 9) int8 array (already color indices).
    """
    face_indices = {}
    for i, face in enumerate(ALL_FACES):
        face_indices[face] = states[:, i * 9:(i + 1) * 9].astype(np.intp)
    return face_indices


MIN_OBS_SCORE = 2.0  # observation must match at least ~2 stickers to count


def score_batch(precomputed_frames, face_indices, n):
    """Score all N candidates against precomputed frames in one vectorized pass.

    Each observation is scored against all 6 candidate faces — no pre-assigned
    face identity needed. Each observation contributes its best score across
    all faces and rotations.

    Observations where no face/rotation scores above MIN_OBS_SCORE are
    discarded (likely bad bounding boxes or cross-face contamination).

    face_indices: dict face_name -> (N, 9) color index array
    Returns: (N,) score array.
    """
    all_face_cols = [face_indices[f] for f in ALL_FACES]  # list of 6 (N, 9) arrays

    scores = np.zeros(n)
    for frame_faces in precomputed_frames:
        for dist_table in frame_faces:
            valid = dist_table.min(axis=1) < MAX_DIST * 1.5  # (9,) bool

            best = np.full(n, -999.0)
            # Try all 6 candidate faces × 4 rotations
            for cols in all_face_cols:
                for rot in _ROT_ARRAYS:
                    rotated = cols[:, rot]  # (N, 9)
                    dists = dist_table[_ARANGE9, rotated]  # (N, 9)
                    rot_scores = np.sum((1.0 - dists[:, valid] / MAX_DIST), axis=1)
                    np.maximum(best, rot_scores, out=best)

            # Only count observations that match well for at least one candidate
            mask = best >= MIN_OBS_SCORE
            scores[mask] += best[mask]
    return scores


def assign_faces_hungarian(obs_list, current_state, classifier):
    """Assign observations to faces using Hungarian scoring.

    For each observation, scores against all 6 faces (no rotation needed).
    Returns list of (face_name, obs_labs) tuples.
    """
    aligned = []
    for obs_labs in obs_list:
        best_face = None
        best_score = -999

        for face_name in ALL_FACES:
            s, _ = _score_face_hungarian(
                obs_labs, current_state[face_name], classifier)
            if s > best_score:
                best_score = s
                best_face = face_name

        if best_face is not None and best_score >= -1.0:
            aligned.append((best_face, obs_labs))

    return aligned


def score_state_hungarian(aligned_faces, candidate_state, classifier):
    """Score a candidate state against face-assigned observations.

    Uses Hungarian matching per face. For duplicate face detections,
    keeps the MAX score per face.

    Returns total score.
    """
    face_obs = {}
    for face_name, obs_labs in aligned_faces:
        if face_name not in face_obs:
            face_obs[face_name] = []
        face_obs[face_name].append(obs_labs)

    total = 0.0
    for face_name, obs_list_for_face in face_obs.items():
        best_score = -999.0
        for obs_labs in obs_list_for_face:
            s, _ = _score_face_hungarian(
                obs_labs, candidate_state[face_name], classifier)
            if s > best_score:
                best_score = s
        total += best_score

    return total


# Keep old functions as aliases for calibrator compatibility
def assign_faces_by_scoring(obs_list, current_state, classifier):
    """Alias — delegates to Hungarian-based assignment."""
    return assign_faces_hungarian(obs_list, current_state, classifier)


def score_state_aligned(aligned_faces, candidate_state, classifier):
    """Alias — delegates to Hungarian-based scoring."""
    return score_state_hungarian(aligned_faces, candidate_state, classifier)


class MoveDetector:
    """
    Per-frame voting + streak confirmation move detection.

    Each aligned frame independently scores all 19 candidates (current state
    + 18 single moves). Quality and margin gates filter out noisy frames.
    A move is confirmed when the same non-current candidate wins N consecutive
    high-quality frames.

    Tolerant of alignment classifier mistakes:
    - Empty/unaligned frames are skipped (never reset the streak)
    - Only contradicting evidence resets the streak (different candidate
      or no-move winning)
    - Bad frames that slip through alignment are caught by quality/margin gates
    """

    def __init__(self, cube, calibrator,
                 confirm_streak=3, min_frame_score=2.0,
                 min_frame_margin=0.8, cooldown_frames=3):
        self.cube = copy.deepcopy(cube)
        self.calibrator = calibrator
        self.move_log = []

        self.confirm_streak = confirm_streak
        self.min_frame_score = min_frame_score
        self.min_frame_margin = min_frame_margin
        self.cooldown_frames = cooldown_frames

        self._streak_candidate = None  # notation of current streak leader
        self._streak_count = 0         # consecutive quality frames this candidate won
        self._streak_cube = None       # Cube object for the streak candidate
        self._cooldown = 0

    def _score_single(self, obs_list, candidate_state):
        """Score one candidate state against one frame's observations.

        Hybrid: 50% position-pinned (discriminative) + 50% Hungarian (robust).
        Face assignment is fixed to current state to prevent per-candidate shopping.
        """
        s_pin, nf = pin_and_score(
            obs_list, self.cube.state, candidate_state, self.calibrator)
        aligned = assign_faces_hungarian(
            obs_list, self.cube.state, self.calibrator)
        s_hung = score_state_hungarian(
            aligned, candidate_state, self.calibrator) if aligned else 0.0
        if nf > 0:
            return 0.5 * s_pin + 0.5 * s_hung
        elif aligned:
            return s_hung
        return -999.0

    def _score_frame(self, obs_list):
        """Score all 19 candidates against a single frame's observations.

        Returns sorted list of (score, notation, cube), best first.
        """
        scored = []
        # No-move (current state)
        s = self._score_single(obs_list, self.cube.state)
        scored.append((s, "", self.cube))
        # All 18 single moves
        for notation, cand_cube in generate_candidates(self.cube):
            s = self._score_single(obs_list, cand_cube.state)
            scored.append((s, notation, cand_cube))
        scored.sort(reverse=True, key=lambda x: x[0])
        return scored

    def process_frame(self, face_observations, is_aligned, timestamp):
        """Process one frame through the voting + streak pipeline.

        Args:
            face_observations: list of {"lab_colors": [9 LAB values]}
            is_aligned: bool from alignment classifier
            timestamp: seconds since solve start

        Returns:
            list of detected move dicts [{"move": str, "timestamp": float}]
        """
        # Cooldown after confirmed move
        if self._cooldown > 0:
            self._cooldown -= 1
            return []

        # No observations or not aligned — skip, don't reset anything
        if not is_aligned or not face_observations:
            return []

        # Score all candidates for this single frame
        obs_list = [obs["lab_colors"] for obs in face_observations]
        scored = self._score_frame(obs_list)

        best_score, best_notation, best_cube = scored[0]
        second_score = scored[1][0]
        margin = best_score - second_score

        # Quality gate: frame too noisy to be useful
        if best_score < self.min_frame_score:
            return []

        # Margin gate: ambiguous frame (two candidates nearly tied)
        if margin < self.min_frame_margin:
            return []

        # No-move wins: current state is best match, reset any pending streak
        if best_notation == "":
            self._streak_candidate = None
            self._streak_count = 0
            self._streak_cube = None
            return []

        # A move candidate wins — update streak
        if best_notation == self._streak_candidate:
            self._streak_count += 1
        else:
            self._streak_candidate = best_notation
            self._streak_count = 1
            self._streak_cube = best_cube

        # Streak confirmation
        if self._streak_count >= self.confirm_streak:
            return self._confirm_move(best_notation, best_cube,
                                      best_score, second_score, timestamp)

        return []

    def _confirm_move(self, notation, cube, score, second, timestamp):
        """Confirm a detected move: update state and return entries."""
        self.cube = copy.deepcopy(cube)
        self._streak_candidate = None
        self._streak_count = 0
        self._streak_cube = None
        self._cooldown = self.cooldown_frames

        moves = notation.split()
        entries = []
        for m in moves:
            entry = {"move": m, "timestamp": timestamp}
            self.move_log.append(entry)
            entries.append(entry)

        print(f"    [detected] {notation} "
              f"(score={score:.1f} 2nd={second:.1f} streak={self.confirm_streak})")
        return entries

    @property
    def current_state(self):
        return self.cube.state

    @property
    def move_count(self):
        return len(self.move_log)

    @property
    def notation_string(self):
        return " ".join(entry["move"] for entry in self.move_log)


# Backwards-compatible alias — the new MoveDetector accepts is_aligned directly
AlignmentMoveDetector = MoveDetector


# ---------------------------------------------------------------------------
# Spatial constraint utilities
# ---------------------------------------------------------------------------

# Edge positions: grid positions on each side of a face
# When two faces are adjacent, their boundary stickers are at these positions.
# "right edge" of a face = positions 2, 5, 8
# "left edge" = 0, 3, 6
# "top edge" = 0, 1, 2
# "bottom edge" = 6, 7, 8
EDGE_POSITIONS = {
    "right": [2, 5, 8],
    "left": [0, 3, 6],
    "top": [0, 1, 2],
    "bottom": [6, 7, 8],
}

# FACE_ADJACENCY: for each pair of adjacent faces, which edge of each
# face meets the other, and which positions pair up (under identity rotation).
# This is derived from COORD_MAP cross-face edges.
# Format: (face_a_idx, face_b_idx) -> [(pos_a, pos_b), ...]
# Only need one direction; will check both.

FACE_NAMES = ["up", "right", "front", "down", "left", "back"]
FACE_IDX = {name: i for i, name in enumerate(FACE_NAMES)}


def _angle_diff_180(a, b):
    """Circular angle difference in [0, 90] for angles in [0, 180)."""
    d = abs(a - b)
    return min(d, 180 - d)


def find_cross_face_pairs(face_obs_list, pixel_threshold=80.0, angle_threshold=30.0):
    """Find cross-face boundary sticker pairs from a single frame.

    Two stickers from different face groups are a boundary pair if:
    1. They're pixel-close (within pixel_threshold)
    2. Their face groups have perpendicular mean angles

    Args:
        face_obs_list: list of rich observations (with stickers, mean_angles)
        pixel_threshold: max pixel distance between boundary stickers
        angle_threshold: min angle difference to consider perpendicular

    Returns:
        list of (face_idx_a, pos_a, face_idx_b, pos_b, same_color)
        where face_idx is index into face_obs_list, pos is grid position,
        same_color is bool (LAB distance < threshold → same color)
    """
    SAME_COLOR_THRESHOLD = 12.0  # CIEDE2000 distance for "same color"

    pairs = []
    n = len(face_obs_list)

    for i in range(n):
        for j in range(i + 1, n):
            obs_a = face_obs_list[i]
            obs_b = face_obs_list[j]

            if "mean_angles" not in obs_a or "mean_angles" not in obs_b:
                continue

            # Check if face groups have perpendicular angles
            # Each face has 2 edge angles. Perpendicular faces share one
            # angle direction and differ in the other.
            angles_a = obs_a["mean_angles"]
            angles_b = obs_b["mean_angles"]

            # Check if at least one pair of angles is perpendicular
            perp = False
            for aa in angles_a:
                for ab in angles_b:
                    diff = _angle_diff_180(aa, ab)
                    if 90 - angle_threshold < diff < 90 + angle_threshold:
                        perp = True
                        break
                if perp:
                    break

            if not perp:
                continue

            # Find boundary sticker pairs by proximity
            for pos_a in range(9):
                sticker_a = obs_a["stickers"][pos_a]
                if sticker_a is None:
                    continue
                lab_a = obs_a["lab_colors"][pos_a]
                if lab_a[0] < 40:  # MISSING_LAB sentinel
                    continue

                for pos_b in range(9):
                    sticker_b = obs_b["stickers"][pos_b]
                    if sticker_b is None:
                        continue
                    lab_b = obs_b["lab_colors"][pos_b]
                    if lab_b[0] < 40:
                        continue

                    # Pixel distance between centroids
                    dist = np.linalg.norm(
                        sticker_a["centroid"] - sticker_b["centroid"])
                    if dist > pixel_threshold:
                        continue

                    # These are boundary stickers — compute color similarity
                    lab_dist = ciede2000(
                        cv_lab_to_std(np.asarray(lab_a, dtype=np.float64)),
                        cv_lab_to_std(np.asarray(lab_b, dtype=np.float64)))
                    same_color = lab_dist < SAME_COLOR_THRESHOLD

                    pairs.append((i, pos_a, j, pos_b, same_color))

    return pairs


# ---------------------------------------------------------------------------
# Constraint-based move detection
# ---------------------------------------------------------------------------

def _state_key(state):
    """Hashable key for a cube state dict."""
    return tuple(
        tuple(state[f]) for f in ["up", "right", "front", "down", "left", "back"]
    )


class ConstraintMoveDetector:
    """Move detection via spatial constraint elimination.

    Uses two types of constraints:
    1. Within-face: relative color patterns (which positions share a color)
    2. Cross-face: boundary sticker color relationships (same/different)

    These are structural constraints that don't depend on absolute color
    classification — only on relative LAB similarity.

    For each candidate state, checks if there exists ANY assignment of
    observed face groups to cube faces (+ grid rotations) that satisfies
    all spatial constraints.
    """

    SAME_THRESHOLD = 10.0  # CIEDE2000 distance for "definitely same color"
    DIFF_THRESHOLD = 20.0  # CIEDE2000 distance for "definitely different color"

    def __init__(self, initial_cube):
        self.candidates = [(copy.deepcopy(initial_cube), [])]
        self.confirmed_moves = []
        self.n_clusters = 0
        self._pending_moves = 0  # accumulated gaps without constraints

    def process_cluster(self, observations, gap_frames=0):
        """Process one aligned-frame cluster.

        Args:
            observations: list of rich face obs dicts (from process_frame(rich=True))
            gap_frames: unaligned frames before this cluster

        Returns: list of newly confirmed move strings
        """
        spatial = self._extract_spatial(observations)

        if gap_frames > 0:
            self._pending_moves += 1

        has_constraints = bool(spatial["faces"])

        if has_constraints and self._pending_moves > 0:
            # Expand by accumulated pending moves, cap at 2
            self._expand(min(self._pending_moves, 2))
            self._pending_moves = 0

        before = len(self.candidates)
        if has_constraints:
            self.candidates = [
                (c, p) for c, p in self.candidates
                if self._is_consistent(c.state, spatial)
            ]

        self.n_clusters += 1
        faces = spatial["faces"]
        face_info = {}
        for f, p in faces.items():
            n_same = sum(1 for _, _, s in p if s)
            face_info[f] = f"{len(p)}({n_same}s/{len(p)-n_same}d)"
        print(f"  Cluster {self.n_clusters}: "
              f"faces={face_info}, "
              f"{before} → {len(self.candidates)} candidates")

        new_moves = []
        if len(self.candidates) == 1:
            _, path = self.candidates[0]
            new_moves = path[len(self.confirmed_moves):]
            self.confirmed_moves = path[:]
        elif len(self.candidates) == 0:
            print("  WARNING: all candidates eliminated!")

        return new_moves

    MAX_CANDIDATES = 5000

    def _expand(self, max_moves):
        """Expand candidate set by 1 or 2 moves + no-move."""
        # Cap input to prevent explosion
        if len(self.candidates) > self.MAX_CANDIDATES:
            self.candidates = self.candidates[:self.MAX_CANDIDATES]

        expanded = {}

        for cube, path in self.candidates:
            key = _state_key(cube.state)
            if key not in expanded or len(path) < len(expanded[key][1]):
                expanded[key] = (copy.deepcopy(cube), path[:])

            for notation, c1 in generate_candidates(cube):
                k1 = _state_key(c1.state)
                p1 = path + [notation]
                if k1 not in expanded or len(p1) < len(expanded[k1][1]):
                    expanded[k1] = (c1, p1)

                if max_moves >= 2:
                    for n2, c2 in generate_candidates(c1):
                        k2 = _state_key(c2.state)
                        p2 = path + [notation, n2]
                        if k2 not in expanded or len(p2) < len(expanded[k2][1]):
                            expanded[k2] = (c2, p2)

        self.candidates = list(expanded.values())

    def _extract_spatial(self, observations):
        """Extract constraints from observations.

        Hybrid approach:
        - Center sticker (pos 4) → face identity (absolute color, reliable)
        - Non-center stickers → same/different pairs (relative, no cal needed)

        Groups observations by observation index. For each group, builds
        aggregated same/different pairs via majority vote.

        Returns dict with:
            groups: [[(pos_a, pos_b, same), ...], ...]
        Each group is a list of constraint pairs from one observed face.
        """
        from collections import Counter

        # Group by observation index (not face identity)
        group_votes = {}  # obs_idx -> {(pa,pb): Counter}

        for obs_idx, obs in enumerate(observations):
            lab_colors = obs["lab_colors"]

            if obs_idx not in group_votes:
                group_votes[obs_idx] = {}

            # Filter stickers: must be visible
            valid_pos = set()
            for p in range(9):
                if lab_colors[p][0] >= 40:
                    valid_pos.add(p)

            # Same/different pairs within this face
            for pa in range(9):
                if pa not in valid_pos:
                    continue
                for pb in range(pa + 1, 9):
                    if pb not in valid_pos:
                        continue
                    # CIEDE2000 perceptual distance
                    dist = ciede2000(
                        cv_lab_to_std(np.asarray(lab_colors[pa], dtype=np.float64)),
                        cv_lab_to_std(np.asarray(lab_colors[pb], dtype=np.float64)))
                    if dist < self.SAME_THRESHOLD:
                        rel = "same"
                    elif dist > self.DIFF_THRESHOLD:
                        rel = "diff"
                    else:
                        continue

                    key = (pa, pb)
                    if key not in group_votes[obs_idx]:
                        group_votes[obs_idx][key] = Counter()
                    group_votes[obs_idx][key][rel] += 1

        # Build consensus pairs per group
        groups = []
        for obs_idx in sorted(group_votes.keys()):
            votes = group_votes[obs_idx]
            pairs = []
            for (pa, pb), counter in votes.items():
                total = sum(counter.values())
                if total < 2:
                    continue
                best, count = counter.most_common(1)[0]
                if count >= total * 0.75:
                    pairs.append((pa, pb, best == "same"))
            if pairs:
                groups.append(pairs)

        return {"groups": groups}

    def _is_consistent(self, state, spatial):
        """Check if state is consistent with spatial constraints.

        For each observation group, check if ANY of the 6 faces × 4 rotations
        satisfies >=65% of the same/different pairs.
        A candidate is eliminated if ANY group fails to match any face.
        """
        groups = spatial["groups"]
        if not groups:
            return True

        MIN_MATCH_RATE = 0.65

        for pairs in groups:
            n_pairs = len(pairs)
            min_matched = int(n_pairs * MIN_MATCH_RATE)

            found = False
            for face_name in ALL_FACES:
                if found:
                    break
                colors = state[face_name]
                for rot in GRID_ROTATIONS:
                    matched = 0
                    for pa, pb, same in pairs:
                        ca = colors[rot[pa]]
                        cb = colors[rot[pb]]
                        if same and ca == cb:
                            matched += 1
                        elif not same and ca != cb:
                            matched += 1
                    if matched >= min_matched:
                        found = True
                        break

            if not found:
                return False

        return True

    @property
    def notation_string(self):
        if self.confirmed_moves:
            return " ".join(self.confirmed_moves)
        if self.candidates:
            shortest = min(self.candidates, key=lambda x: len(x[1]))
            path_str = " ".join(shortest[1]) if shortest[1] else "(no move)"
            return f"{path_str} (? {len(self.candidates)} candidates)"
        return "(none)"

    @property
    def move_count(self):
        return len(self.confirmed_moves)


# ---------------------------------------------------------------------------
# Beam search move detection (offline)
# ---------------------------------------------------------------------------

def adaptive_still_threshold(motions, ratio=2.0, floor=8.0, ceil=25.0, default=10.0):
    """Per-video stillness threshold from the cube-region motion distribution.

    A held cube is still (motion ≈ sensor noise); a turning cube moves a whole
    layer of stickers. The two form a low/high split whose absolute scale shifts
    with lighting, cube size in frame, and fps — so a hand-tuned absolute
    threshold doesn't transfer. Instead estimate this video's still-baseline as a
    low percentile (p20, robust since holds outnumber turns) and reject anything
    beyond baseline*ratio, clamped to a sane band. Falls back to `default` when
    there are too few frames to estimate.
    """
    m = [x for x in motions if x is not None]
    if len(m) < 8:
        return default
    baseline = float(np.percentile(np.asarray(m, dtype=float), 20))
    return float(min(ceil, max(floor, baseline * ratio)))


def extract_clusters(frames_data, min_cluster_size=2, min_gap=4):
    """Group frame observations into clusters based on alignment signal.

    Uses alignment alone to find state boundaries — sticker detection
    failures within an aligned period don't split clusters. A gap in
    alignment shorter than min_gap is bridged (treated as noise/wobble,
    not a move).

    Gaps shorter than ``min_gap`` are treated as alignment noise. Longer gaps
    end the current cluster and start a new state boundary.

    Args:
        frames_data: list of (is_aligned, obs_list, timestamp)
        min_cluster_size: minimum observation frames for a cluster to be kept
        min_gap: unaligned gaps shorter than this are bridged

    Returns:
        list of dicts with keys:
            obs_frames: list of obs_list (one per frame with observations)
            gap_before: number of unaligned frames before this cluster
    """
    # Pass 1: find contiguous aligned runs
    runs = []  # (start_idx, end_idx) inclusive
    run_start = None
    for i, (is_aligned, _, _) in enumerate(frames_data):
        if is_aligned:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                runs.append((run_start, i - 1))
                run_start = None
    if run_start is not None:
        runs.append((run_start, len(frames_data) - 1))

    if not runs:
        return []

    # Pass 2: merge runs separated by short gaps
    merged_runs = [runs[0]]
    for start, end in runs[1:]:
        prev_end = merged_runs[-1][1]
        gap = start - prev_end - 1
        if gap < min_gap:
            # Bridge: extend previous run to cover this one
            merged_runs[-1] = (merged_runs[-1][0], end)
        else:
            merged_runs.append((start, end))

    # Pass 3: collect observations from within each merged run
    clusters = []
    prev_end = -1
    for start, end in merged_runs:
        obs_frames = []
        for i in range(start, end + 1):
            _, obs_list, _ = frames_data[i]
            if obs_list:
                obs_frames.append(obs_list)

        gap_before = start - prev_end - 1
        prev_end = end

        if len(obs_frames) >= min_cluster_size:
            clusters.append({
                "obs_frames": obs_frames,
                "gap_before": max(gap_before, 0),
            })

    return clusters


def _invert_move(notation):
    """Invert a single move: R↔R', R2↔R2."""
    if notation.endswith("'"):
        return notation[:-1]
    elif notation.endswith("2"):
        return notation  # double moves are self-inverse
    else:
        return notation + "'"


class BeamSearchDetector:
    """Offline beam search with forward + backward passes.

    Uses numeric state representation: (54,) int8 arrays with precomputed
    move permutations. No deepcopy, no string lookups in the hot path.

    Beam items: (state_array, path, score)
    """

    def __init__(self, classifier, beam_width=200, max_depth=1):
        self.classifier = classifier
        self.beam_width = beam_width
        self.max_depth = max_depth

    def detect(self, clusters, initial_cube, solved_state=None):
        """Run beam search on pre-extracted clusters."""
        if not clusters:
            return []

        import time
        t0 = time.perf_counter()
        precomputed_clusters = [
            precompute_cluster(c["obs_frames"], self.classifier)
            for c in clusters
        ]
        print(f"  Precomputed {len(clusters)} clusters in "
              f"{time.perf_counter() - t0:.2f}s", flush=True)

        initial_arr = state_to_array(initial_cube.state)

        # Phase 1: Independent forward + backward
        fwd_beams = self._forward(clusters, precomputed_clusters, initial_arr)

        if solved_state is not None:
            solved_arr = state_to_array(solved_state)
            bwd_beams = self._backward(clusters, precomputed_clusters, solved_arr)
            result = self._merge(fwd_beams, bwd_beams)
            if result is not None:
                return result

            # Phase 2: Score fusion — re-run forward with backward hints
            # Backward scores tell us which states are reachable from solved.
            # Adding them as a bonus during forward pruning prevents the
            # correct state from being pruned when forward scores are noisy.
            print("  Phase 1 merge failed, trying score fusion...", flush=True)
            bwd_hints = [{s.tobytes(): sc for s, _, sc in beam}
                         for beam in bwd_beams]
            fwd_beams2 = self._forward_fused(
                clusters, precomputed_clusters, initial_arr, bwd_hints)
            result = self._merge(fwd_beams2, bwd_beams)
            if result is not None:
                return result

        # Fallback: best forward candidate
        if fwd_beams and fwd_beams[-1]:
            best = max(fwd_beams[-1], key=lambda x: x[2])
            return best[1]
        return []

    def _expand(self, beam, depth):
        """Expand beam by applying all moves. Pure numpy — no deepcopy."""
        expanded = {}  # bytes key -> (state_array, path)

        for state, path, _ in beam:
            # No-move
            key = state.tobytes()
            if key not in expanded or len(path) < len(expanded[key][1]):
                expanded[key] = (state, path[:])

            # Depth 1
            for notation, perm in MOVE_LIST:
                s1 = state[perm]
                k1 = s1.tobytes()
                p1 = path + [notation]
                if k1 not in expanded or len(p1) < len(expanded[k1][1]):
                    expanded[k1] = (s1, p1)

                # Depth 2
                if depth >= 2:
                    for n2, perm2 in MOVE_LIST:
                        s2 = s1[perm2]
                        k2 = s2.tobytes()
                        p2 = path + [notation, n2]
                        if k2 not in expanded or len(p2) < len(expanded[k2][1]):
                            expanded[k2] = (s2, p2)

        return [(s, p, 0.0) for s, p in expanded.values()]

    def _score_and_prune(self, beam, precomputed):
        """Score all candidates in one batched numpy pass, keep top beam_width."""
        if not precomputed:
            return beam[:self.beam_width]

        n = len(beam)

        # Use all frames with valid observations (no sub-sampling)
        states = np.stack([s for s, _, _ in beam])  # (N, 54) int8
        face_indices = _build_face_indices_from_states(states)

        scores = score_batch(precomputed, face_indices, n)

        top_idx = np.argsort(scores)[::-1][:self.beam_width]
        return [(beam[i][0], beam[i][1], float(scores[i])) for i in top_idx]

    def _score_and_prune_fused(self, beam, precomputed, bwd_scores):
        """Score + prune with backward score bonus for sorting.

        Forward scores are stored in the beam (for correct merge scoring).
        Backward scores are added as a bonus only for pruning order — states
        the backward pass also reached are less likely to be pruned.
        """
        if not precomputed:
            return beam[:self.beam_width]

        n = len(beam)
        states = np.stack([s for s, _, _ in beam])
        face_indices = _build_face_indices_from_states(states)

        fwd_scores = score_batch(precomputed, face_indices, n)

        # Add backward bonus for pruning
        sort_scores = fwd_scores.copy()
        for j in range(n):
            key = beam[j][0].tobytes()
            bwd_sc = bwd_scores.get(key, 0.0)
            if bwd_sc > 0:
                sort_scores[j] += bwd_sc

        top_idx = np.argsort(sort_scores)[::-1][:self.beam_width]
        return [(beam[i][0], beam[i][1], float(fwd_scores[i])) for i in top_idx]

    def _forward_fused(self, clusters, precomputed_clusters, initial_arr, bwd_hints):
        """Forward pass with backward score hints biasing pruning."""
        beam = [(initial_arr, [], 0.0)]
        beams_after = []

        for i, cluster in enumerate(clusters):
            gap = cluster["gap_before"]

            if gap > 0 and i > 0:
                beam = self._expand(beam, self.max_depth)
                print(f"  Fused fwd expand: {len(beam)} candidates (gap={gap})",
                      flush=True)

            beam = self._score_and_prune_fused(
                beam, precomputed_clusters[i], bwd_hints[i])
            top_score = beam[0][2] if beam else 0
            top_path = " ".join(beam[0][1]) if beam and beam[0][1] else "(none)"
            print(f"  Fused fwd cluster {i}: {len(beam)} candidates, "
                  f"top={top_score:.1f} path=[{top_path}]")

            beams_after.append(list(beam))

        return beams_after

    def _forward(self, clusters, precomputed_clusters, initial_arr):
        beam = [(initial_arr, [], 0.0)]
        beams_after = []

        for i, cluster in enumerate(clusters):
            gap = cluster["gap_before"]

            if gap > 0 and i > 0:
                beam = self._expand(beam, self.max_depth)
                print(f"  Fwd expand: {len(beam)} candidates (gap={gap})", flush=True)

            beam = self._score_and_prune(beam, precomputed_clusters[i])
            top_score = beam[0][2] if beam else 0
            top_path = " ".join(beam[0][1]) if beam and beam[0][1] else "(none)"
            print(f"  Fwd cluster {i}: {len(beam)} candidates, "
                  f"top={top_score:.1f} path=[{top_path}]")

            beams_after.append(list(beam))

        return beams_after

    def _backward(self, clusters, precomputed_clusters, solved_arr):
        beam = [(solved_arr, [], 0.0)]
        beams_after = []

        for i in range(len(clusters) - 1, -1, -1):
            next_gap = clusters[i + 1]["gap_before"] if i + 1 < len(clusters) else 0
            if next_gap > 0:
                beam = self._expand(beam, self.max_depth)
                print(f"  Bwd expand: {len(beam)} candidates (gap={next_gap})", flush=True)

            beam = self._score_and_prune(beam, precomputed_clusters[i])
            top_score = beam[0][2] if beam else 0
            print(f"  Bwd cluster {i}: {len(beam)} candidates, top={top_score:.1f}")

            beams_after.append(list(beam))

        beams_after.reverse()
        return beams_after

    def _merge(self, fwd_beams, bwd_beams):
        best_merge = None
        best_score = -1.0

        for i in range(len(fwd_beams)):
            fwd_states = {s.tobytes(): (s, p, sc) for s, p, sc in fwd_beams[i]}
            bwd_states = {s.tobytes(): (s, p, sc) for s, p, sc in bwd_beams[i]}

            common = set(fwd_states) & set(bwd_states)
            if common:
                best_key = max(common, key=lambda k:
                    fwd_states[k][2] + bwd_states[k][2])
                combined = fwd_states[best_key][2] + bwd_states[best_key][2]

                if combined > best_score:
                    best_score = combined
                    fwd_path = fwd_states[best_key][1]
                    bwd_path = bwd_states[best_key][1]
                    bwd_forward = [_invert_move(m) for m in reversed(bwd_path)]
                    best_merge = (i, fwd_path, bwd_forward)

        if best_merge:
            i, fwd_path, bwd_forward = best_merge
            full_path = fwd_path + bwd_forward
            print(f"  Merge at cluster {i} (score={best_score:.1f}): "
                  f"fwd=[{' '.join(fwd_path)}] + "
                  f"bwd_inv=[{' '.join(bwd_forward)}]")
            return full_path

        return None

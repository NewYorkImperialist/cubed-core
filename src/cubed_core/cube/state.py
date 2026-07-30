"""Deterministic fixed-center 3x3 facelet state and move permutations.

The stable array layout is six row-major faces in ``U, R, F, D, L, B`` order.
Color indices deliberately use Cubed's established calibration order:
``white, yellow, red, orange, blue, green``.

This module validates facelet shape, color counts, and fixed centers. It does
not attempt a full cubie reachability proof; callers should not treat
structural validation as proof that an arbitrary hand-authored state is
physically solvable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .notation import AlgorithmLike, MoveLike, parse_algorithm, parse_move

FACE_ORDER = ("up", "right", "front", "down", "left", "back")
COLOR_ORDER = ("white", "yellow", "red", "orange", "blue", "green")

# Stable whole-cube rotation order. Each key names the model faces occupying
# spatial up, front, and right. Benchmark scoring treats states equivalent
# under these 24 proper orientations, so this order is a cube-runtime contract
# rather than a decoder-implementation detail.
ORIENTATION_KEYS = (
    ("up", "front", "right"),
    ("up", "right", "back"),
    ("up", "back", "left"),
    ("up", "left", "front"),
    ("front", "down", "right"),
    ("front", "right", "up"),
    ("front", "up", "left"),
    ("front", "left", "down"),
    ("down", "back", "right"),
    ("down", "right", "front"),
    ("down", "front", "left"),
    ("down", "left", "back"),
    ("back", "up", "right"),
    ("back", "right", "down"),
    ("back", "down", "left"),
    ("back", "left", "up"),
    ("left", "front", "up"),
    ("left", "up", "back"),
    ("left", "back", "down"),
    ("left", "down", "front"),
    ("right", "front", "down"),
    ("right", "down", "back"),
    ("right", "back", "up"),
    ("right", "up", "front"),
)

FACE_COLORS = MappingProxyType(
    {
        "up": "white",
        "right": "red",
        "front": "green",
        "down": "yellow",
        "left": "orange",
        "back": "blue",
    }
)
COLOR_TO_INDEX = MappingProxyType({color: index for index, color in enumerate(COLOR_ORDER)})
FACE_TO_LETTER = MappingProxyType(
    {"up": "U", "right": "R", "front": "F", "down": "D", "left": "L", "back": "B"}
)
LETTER_TO_FACE = MappingProxyType({letter: face for face, letter in FACE_TO_LETTER.items()})

FACELET_COUNT = 54
FACE_SIZE = 9
_FACE_OFFSETS = {face: index * FACE_SIZE for index, face in enumerate(FACE_ORDER)}
_CLOCKWISE_FACE_SOURCES = np.asarray((6, 3, 0, 7, 4, 1, 8, 5, 2), dtype=np.intp)


def _face_positions(face: str, positions: Sequence[int]) -> NDArray[np.intp]:
    return np.asarray([_FACE_OFFSETS[face] + position for position in positions], dtype=np.intp)


def _strip(face: str, *positions: int) -> list[int]:
    offset = _FACE_OFFSETS[face]
    return [offset + position for position in positions]


# Destination and source facelets for one clockwise turn, excluding the face itself.
# The mappings preserve the established U/R/F/D/L/B row-major orientation contract.
_ADJACENT_TRANSFERS = {
    "U": (
        _strip("left", 0, 1, 2)
        + _strip("back", 0, 1, 2)
        + _strip("right", 0, 1, 2)
        + _strip("front", 0, 1, 2),
        _strip("front", 0, 1, 2)
        + _strip("left", 0, 1, 2)
        + _strip("back", 0, 1, 2)
        + _strip("right", 0, 1, 2),
    ),
    "R": (
        _strip("front", 2, 5, 8)
        + _strip("down", 2, 5, 8)
        + _strip("back", 0, 3, 6)
        + _strip("up", 2, 5, 8),
        _strip("down", 2, 5, 8)
        + _strip("back", 6, 3, 0)
        + _strip("up", 8, 5, 2)
        + _strip("front", 2, 5, 8),
    ),
    "F": (
        _strip("left", 2, 5, 8)
        + _strip("up", 6, 7, 8)
        + _strip("right", 0, 3, 6)
        + _strip("down", 0, 1, 2),
        _strip("down", 0, 1, 2)
        + _strip("left", 8, 5, 2)
        + _strip("up", 6, 7, 8)
        + _strip("right", 6, 3, 0),
    ),
    "D": (
        _strip("left", 6, 7, 8)
        + _strip("back", 6, 7, 8)
        + _strip("right", 6, 7, 8)
        + _strip("front", 6, 7, 8),
        _strip("back", 6, 7, 8)
        + _strip("right", 6, 7, 8)
        + _strip("front", 6, 7, 8)
        + _strip("left", 6, 7, 8),
    ),
    "L": (
        _strip("front", 0, 3, 6)
        + _strip("down", 0, 3, 6)
        + _strip("back", 2, 5, 8)
        + _strip("up", 0, 3, 6),
        _strip("up", 0, 3, 6)
        + _strip("front", 0, 3, 6)
        + _strip("down", 6, 3, 0)
        + _strip("back", 8, 5, 2),
    ),
    "B": (
        _strip("left", 0, 3, 6)
        + _strip("up", 0, 1, 2)
        + _strip("right", 2, 5, 8)
        + _strip("down", 6, 7, 8),
        _strip("up", 2, 1, 0)
        + _strip("right", 2, 5, 8)
        + _strip("down", 8, 7, 6)
        + _strip("left", 0, 3, 6),
    ),
}


def _clockwise_permutation(face_letter: str) -> NDArray[np.intp]:
    permutation = np.arange(FACELET_COUNT, dtype=np.intp)
    face = LETTER_TO_FACE[face_letter]
    face_positions = _face_positions(face, range(FACE_SIZE))
    permutation[face_positions] = face_positions[_CLOCKWISE_FACE_SOURCES]

    destinations, sources = _ADJACENT_TRANSFERS[face_letter]
    permutation[np.asarray(destinations, dtype=np.intp)] = np.asarray(sources, dtype=np.intp)
    permutation.flags.writeable = False
    return permutation


def _move_permutations() -> MappingProxyType[str, NDArray[np.intp]]:
    permutations: dict[str, NDArray[np.intp]] = {}
    identity = np.arange(FACELET_COUNT, dtype=np.intp)
    for face_letter in LETTER_TO_FACE:
        quarter_turn = _clockwise_permutation(face_letter)
        for turns, suffix in ((1, ""), (2, "2"), (3, "'")):
            permutation = identity
            for _ in range(turns):
                permutation = permutation[quarter_turn]
            permutation = np.asarray(permutation, dtype=np.intp)
            permutation.flags.writeable = False
            permutations[f"{face_letter}{suffix}"] = permutation
    return MappingProxyType(permutations)


MOVE_PERMUTATIONS = _move_permutations()

_SOLVED_VALUES = [COLOR_TO_INDEX[FACE_COLORS[face]] for face in FACE_ORDER for _ in range(9)]
SOLVED_ARRAY = np.asarray(_SOLVED_VALUES, dtype=np.int8)
SOLVED_ARRAY.flags.writeable = False
_SOLVED_CENTERS = SOLVED_ARRAY[4::9]


def _validated_array(array: ArrayLike) -> NDArray[np.int8]:
    values = np.asarray(array)
    if values.shape != (FACELET_COUNT,):
        raise ValueError(f"cube array must have shape ({FACELET_COUNT},), got {values.shape}")
    if np.issubdtype(values.dtype, np.bool_) or not np.issubdtype(values.dtype, np.integer):
        raise TypeError("cube array must contain integer color indices")

    integer_values = values.astype(np.int64, copy=False)
    if np.any(integer_values < 0) or np.any(integer_values >= len(COLOR_ORDER)):
        raise ValueError(f"cube color indices must be in 0..{len(COLOR_ORDER) - 1}")

    counts = np.bincount(integer_values, minlength=len(COLOR_ORDER))
    if not np.array_equal(counts, np.full(len(COLOR_ORDER), 9, dtype=counts.dtype)):
        raise ValueError(
            f"cube array must contain exactly nine of each color index, got {counts.tolist()}"
        )
    if not np.array_equal(integer_values[4::9], _SOLVED_CENTERS):
        raise ValueError("cube centers must remain in canonical U/R/F/D/L/B orientation")

    return integer_values.astype(np.int8, copy=True)


def state_to_array(state: Mapping[str, Sequence[str]] | Cube) -> NDArray[np.int8]:
    """Convert a named-face state to a validated, independent ``(54,)`` array."""

    if isinstance(state, Cube):
        return state.to_array()
    if not isinstance(state, Mapping):
        raise TypeError("cube state must be a mapping from face name to nine colors")

    missing = set(FACE_ORDER).difference(state)
    extra = set(state).difference(FACE_ORDER)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={sorted(missing)}")
        if extra:
            details.append(f"extra={sorted(extra)}")
        raise ValueError(f"cube state must contain exactly {FACE_ORDER}: {', '.join(details)}")

    values: list[int] = []
    for face in FACE_ORDER:
        stickers = state[face]
        if isinstance(stickers, (str, bytes)) or not isinstance(stickers, Sequence):
            raise TypeError(f"face {face!r} must be a sequence of nine color names")
        if len(stickers) != FACE_SIZE:
            raise ValueError(f"face {face!r} must contain exactly nine stickers")
        for color in stickers:
            try:
                values.append(COLOR_TO_INDEX[color])
            except (KeyError, TypeError) as exc:
                raise ValueError(f"unknown color {color!r} on face {face!r}") from exc

    return _validated_array(values)


def array_to_state(array: ArrayLike) -> dict[str, list[str]]:
    """Convert a validated numeric array to an independent named-face mapping."""

    values = _validated_array(array)
    return {
        face: [COLOR_ORDER[int(value)] for value in values[index * 9 : (index + 1) * 9]]
        for index, face in enumerate(FACE_ORDER)
    }


def permutation_for_move(move: MoveLike) -> NDArray[np.intp]:
    """Return the immutable permutation for one parsed move.

    Permutations use ``new_state = old_state[permutation]``.
    """

    canonical = str(parse_move(move))
    return MOVE_PERMUTATIONS[canonical]


def apply_move_to_array(array: ArrayLike, move: MoveLike) -> NDArray[np.int8]:
    """Return a validated copy with one outer-face move applied."""

    values = _validated_array(array)
    return values[permutation_for_move(move)].copy()


def is_solved_state(state: Mapping[str, Sequence[str]] | ArrayLike | Cube) -> bool:
    """Return whether a structurally valid state is the canonical solved state."""

    if isinstance(state, Cube):
        values = state._facelets
    elif isinstance(state, Mapping):
        values = state_to_array(state)
    else:
        values = _validated_array(state)
    return bool(np.array_equal(values, SOLVED_ARRAY))


class Cube:
    """Mutable fixed-center 3x3 state with deterministic face-turn semantics."""

    __slots__ = ("_facelets",)

    def __init__(
        self,
        state: Mapping[str, Sequence[str]] | ArrayLike | Cube | None = None,
    ) -> None:
        if state is None:
            self._facelets = SOLVED_ARRAY.copy()
        elif isinstance(state, Cube):
            self._facelets = state.to_array()
        elif isinstance(state, Mapping):
            self._facelets = state_to_array(state)
        else:
            self._facelets = _validated_array(state)

    @classmethod
    def solved(cls) -> Cube:
        """Create a solved cube."""

        return cls()

    @classmethod
    def from_array(cls, array: ArrayLike) -> Cube:
        """Create a cube from the stable numeric representation."""

        return cls(array)

    @classmethod
    def from_state(cls, state: Mapping[str, Sequence[str]]) -> Cube:
        """Create a cube from named row-major faces."""

        return cls(state)

    @property
    def state(self) -> dict[str, list[str]]:
        """Return an independent named-face snapshot."""

        return array_to_state(self._facelets)

    def to_array(self) -> NDArray[np.int8]:
        """Return an independent array snapshot."""

        return self._facelets.copy()

    def copy(self) -> Cube:
        return Cube(self)

    def face(self, face: str) -> tuple[str, ...]:
        """Return one row-major face by its long name."""

        if face not in FACE_ORDER:
            raise ValueError(f"face must be one of {FACE_ORDER}, got {face!r}")
        offset = _FACE_OFFSETS[face]
        return tuple(COLOR_ORDER[int(value)] for value in self._facelets[offset : offset + 9])

    def is_solved(self) -> bool:
        return bool(np.array_equal(self._facelets, SOLVED_ARRAY))

    def apply_move(self, move: MoveLike) -> Cube:
        """Apply one move in place and return ``self`` for chaining."""

        self._facelets = self._facelets[permutation_for_move(move)].copy()
        return self

    def apply_algorithm(self, algorithm: AlgorithmLike) -> Cube:
        """Apply an algorithm in order and return ``self`` for chaining."""

        for move in parse_algorithm(algorithm):
            self._facelets = self._facelets[MOVE_PERMUTATIONS[str(move)]].copy()
        return self

    def moved(self, move: MoveLike) -> Cube:
        """Return an independent cube after one move."""

        return self.copy().apply_move(move)

    def transformed(self, algorithm: AlgorithmLike) -> Cube:
        """Return an independent cube after an algorithm."""

        return self.copy().apply_algorithm(algorithm)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Cube):
            return NotImplemented
        return bool(np.array_equal(self._facelets, other._facelets))

    def __repr__(self) -> str:
        return f"Cube(state={self.state!r})"

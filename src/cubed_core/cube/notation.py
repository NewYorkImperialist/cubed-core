"""Strict parsing for standard 3x3 outer-face move notation.

The core model intentionally supports only the 18 fixed-center face turns:
``U R F D L B`` with an optional prime or double suffix. Slice moves, wide
moves, and whole-cube rotations change the reference frame and belong in a
separate orientation layer.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

FACE_LETTERS = ("U", "R", "F", "D", "L", "B")
_MOVE_PATTERN = re.compile(r"([URFDLB])([2']?)\Z")
_SUFFIX_TO_TURNS = {"": 1, "2": 2, "'": 3}
_TURNS_TO_SUFFIX = {1: "", 2: "2", 3: "'"}


@dataclass(frozen=True, slots=True)
class Move:
    """One clockwise, double, or counter-clockwise outer-face turn.

    ``turns`` is the number of clockwise quarter turns in ``{1, 2, 3}``.
    Three clockwise quarter turns are rendered as the conventional prime move.
    """

    face: str
    turns: int = 1

    def __post_init__(self) -> None:
        if self.face not in FACE_LETTERS:
            raise ValueError(f"face must be one of {FACE_LETTERS}, got {self.face!r}")
        if type(self.turns) is not int or self.turns not in _TURNS_TO_SUFFIX:
            raise ValueError("turns must be one of 1, 2, or 3")

    def inverse(self) -> Move:
        """Return the move that exactly undoes this move."""

        return Move(self.face, 2 if self.turns == 2 else 4 - self.turns)

    def __str__(self) -> str:
        return f"{self.face}{_TURNS_TO_SUFFIX[self.turns]}"


MoveLike = str | Move
AlgorithmLike = str | Iterable[MoveLike]


def parse_move(token: MoveLike) -> Move:
    """Parse one canonical outer-face token.

    Surrounding whitespace is accepted. Embedded whitespace, lower-case moves,
    wide/slice moves, rotations, and stacked suffixes are rejected.
    """

    if isinstance(token, Move):
        return token
    if not isinstance(token, str):
        raise TypeError(f"move token must be str or Move, got {type(token).__name__}")

    match = _MOVE_PATTERN.fullmatch(token.strip())
    if match is None:
        raise ValueError(f"invalid outer-face move token: {token!r}")
    face, suffix = match.groups()
    return Move(face, _SUFFIX_TO_TURNS[suffix])


def parse_algorithm(algorithm: AlgorithmLike) -> tuple[Move, ...]:
    """Parse a whitespace-delimited algorithm or an iterable of move tokens."""

    if isinstance(algorithm, str):
        tokens: Iterable[MoveLike] = algorithm.split()
    else:
        try:
            tokens = iter(algorithm)
        except TypeError as exc:
            raise TypeError("algorithm must be a string or iterable of move tokens") from exc

    return tuple(parse_move(token) for token in tokens)


def format_algorithm(algorithm: AlgorithmLike) -> str:
    """Return one stable, single-space canonical representation."""

    return " ".join(str(move) for move in parse_algorithm(algorithm))


def invert_algorithm(algorithm: AlgorithmLike) -> tuple[Move, ...]:
    """Return the inverse algorithm in application order."""

    moves = parse_algorithm(algorithm)
    return tuple(move.inverse() for move in reversed(moves))

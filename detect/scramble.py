"""
Parse standard cube notation and apply scrambles to a Cube.

Supports: face moves (R/U/F/D/L/B), primes ('), doubles (2),
slices (M/S/E), wide moves (r/Rw), rotations (x/y/z).
"""

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.cube import Cube

FACE_MAP = {
    "R": "right", "U": "up", "F": "front",
    "D": "down", "L": "left", "B": "back",
}

SLICE_MAP = {"M": "m", "S": "s", "E": "e"}

WIDE_MAP = {
    "r": "right", "u": "up", "f": "front",
    "d": "down", "l": "left", "b": "back",
}

ROTATION_SET = {"x", "y", "z"}


def parse_scramble(scramble_str):
    """
    Tokenize a scramble string into individual move tokens.

    Args:
        scramble_str: e.g. "R U' F2 B L2 D' M x2"

    Returns:
        List of move strings: ["R", "U'", "F2", "B", "L2", "D'", "M", "x2"]
    """
    return scramble_str.strip().split()


def _token_base(token):
    """Extract the base move from a token (strip modifier and normalize wide notation)."""
    # Normalize wide: "Rw" -> "r", "Uw'" -> "u'"
    if len(token) >= 2 and token[1] == "w":
        token = token[0].lower() + token[2:]
    if token[-1] in ("'", "2"):
        return token[:-1]
    return token


def validate_scramble(scramble_str):
    """Validate a scramble string. Returns list of tokens. Raises ValueError if invalid."""
    s = scramble_str.strip()
    if not s:
        return []
    if len(s) > 500:
        raise ValueError("Scramble too long (max 500 characters)")
    tokens = s.split()
    for token in tokens:
        base = _token_base(token)
        if base not in FACE_MAP and base not in SLICE_MAP and base not in WIDE_MAP and base not in ROTATION_SET:
            raise ValueError(f"Unknown move: {token!r}")
    return tokens


def apply_move(cube, token):
    """
    Apply a single move token to a Cube instance (mutates in place).

    Args:
        cube: Cube instance
        token: e.g. "R", "U'", "F2", "M'", "r", "Rw'", "x", "y2"
    """
    # Normalize wide move notation: "Rw" -> "r", "Uw'" -> "u'"
    if len(token) >= 2 and token[1] == "w":
        token = token[0].lower() + token[2:]

    # Determine base move and modifier
    if token[-1] == "'":
        base = token[:-1]
        modifier = "prime"
    elif token[-1] == "2":
        base = token[:-1]
        modifier = "double"
    else:
        base = token
        modifier = "single"

    # Rotations (x, y, z)
    if base in ROTATION_SET:
        if modifier == "prime":
            cube.rotate(f"{base}'")
        elif modifier == "double":
            cube.rotate(base)
            cube.rotate(base)
        else:
            cube.rotate(base)

    # Slice moves (M, S, E)
    elif base in SLICE_MAP:
        s = SLICE_MAP[base]
        if modifier == "prime":
            cube.move_slice_prime(s)
        elif modifier == "double":
            cube.move_slice_two(s)
        else:
            cube.move_slice(s)

    # Wide moves (r, u, f, d, l, b)
    elif base in WIDE_MAP:
        side = WIDE_MAP[base]
        if modifier == "prime":
            cube.wide_move_prime(side)
        elif modifier == "double":
            cube.wide_move_two(side)
        else:
            cube.wide_move(side)

    # Face moves (R, U, F, D, L, B)
    elif base in FACE_MAP:
        side = FACE_MAP[base]
        if modifier == "prime":
            cube.move_prime(side)
        elif modifier == "double":
            cube.move_two(side)
        else:
            cube.move(side)

    else:
        raise ValueError(f"Unknown move token: {token!r}")


def apply_scramble(cube, scramble_str):
    """
    Apply a full scramble string to a Cube instance (mutates in place).

    Args:
        cube: Cube instance
        scramble_str: e.g. "R U' F2 B L2 D'"
    """
    for token in parse_scramble(scramble_str):
        apply_move(cube, token)


def scrambled_cube(scramble_str):
    """
    Create a new Cube with a scramble applied.

    Args:
        scramble_str: e.g. "R U' F2 B L2 D'"

    Returns:
        Cube instance in scrambled state
    """
    cube = Cube()
    apply_scramble(cube, scramble_str)
    return cube


def move_to_notation(method, arg):
    """
    Convert a Cube method call back to standard notation.

    Args:
        method: "move", "move_prime", "move_two", "move_slice", etc.
        arg: "right", "up", "m", etc.

    Returns:
        Standard notation string like "R", "U'", "M2"
    """
    # Reverse maps
    face_rev = {v: k for k, v in FACE_MAP.items()}
    slice_rev = {v: k for k, v in SLICE_MAP.items()}
    wide_rev = {v: k for k, v in WIDE_MAP.items()}

    if method == "move":
        return face_rev[arg]
    elif method == "move_prime":
        return face_rev[arg] + "'"
    elif method == "move_two":
        return face_rev[arg] + "2"
    elif method == "move_slice":
        return slice_rev[arg]
    elif method == "move_slice_prime":
        return slice_rev[arg] + "'"
    elif method == "move_slice_two":
        return slice_rev[arg] + "2"
    elif method == "wide_move":
        return wide_rev[arg]
    elif method == "wide_move_prime":
        return wide_rev[arg] + "'"
    elif method == "wide_move_two":
        return wide_rev[arg] + "2"
    elif method == "rotate":
        return arg  # already in standard notation ("x", "y'", etc.)
    else:
        raise ValueError(f"Unknown method: {method!r}")

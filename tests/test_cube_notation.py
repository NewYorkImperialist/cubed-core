from __future__ import annotations

import pytest

from cubed_core.cube import (
    Move,
    format_algorithm,
    invert_algorithm,
    parse_algorithm,
    parse_move,
)


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("U", Move("U", 1)),
        (" R2 ", Move("R", 2)),
        ("F'", Move("F", 3)),
        (Move("B", 2), Move("B", 2)),
    ],
)
def test_parse_move_accepts_canonical_outer_face_notation(token, expected):
    assert parse_move(token) == expected


@pytest.mark.parametrize(
    "token",
    [
        "",
        "r",
        "Rw",
        "M",
        "x",
        "Q",
        "R3",
        "RR",
        "R''",
        "R2'",
        "R U",
        "U’",
    ],
)
def test_parse_move_rejects_noncanonical_or_reference_frame_moves(token):
    with pytest.raises(ValueError, match="invalid outer-face"):
        parse_move(token)


def test_move_validation_and_inverse_are_exact():
    with pytest.raises(ValueError, match="face must"):
        Move("r")
    with pytest.raises(ValueError, match="turns must"):
        Move("R", 0)
    with pytest.raises(ValueError, match="turns must"):
        Move("R", True)

    assert Move("U").inverse() == Move("U", 3)
    assert Move("D", 2).inverse() == Move("D", 2)
    assert Move("L", 3).inverse() == Move("L")


def test_algorithm_parsing_formatting_and_empty_identity_are_stable():
    parsed = parse_algorithm("  R   U2\nF'  ")
    assert parsed == (Move("R"), Move("U", 2), Move("F", 3))
    assert format_algorithm(parsed) == "R U2 F'"
    assert parse_algorithm("") == ()
    assert format_algorithm([]) == ""


def test_algorithm_iterable_is_parsed_without_silently_skipping_tokens():
    assert parse_algorithm(["R", Move("U", 2), "F'"]) == (
        Move("R"),
        Move("U", 2),
        Move("F", 3),
    )
    with pytest.raises(ValueError, match="invalid outer-face"):
        parse_algorithm(["R", "bad", "U"])


def test_invert_algorithm_reverses_order_and_each_move():
    inverse = invert_algorithm("R U2 F' L")
    assert inverse == (Move("L", 3), Move("F"), Move("U", 2), Move("R", 3))
    assert format_algorithm(inverse) == "L' F U2 R'"

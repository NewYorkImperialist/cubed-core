from __future__ import annotations

import random

import numpy as np
import pytest

from cubed_core.cube import (
    COLOR_ORDER,
    FACE_ORDER,
    MOVE_PERMUTATIONS,
    SOLVED_ARRAY,
    Cube,
    apply_move_to_array,
    array_to_state,
    invert_algorithm,
    is_solved_state,
    permutation_for_move,
    state_to_array,
)

_COLOR_CODE = {
    "white": "W",
    "yellow": "Y",
    "red": "R",
    "orange": "O",
    "blue": "B",
    "green": "G",
}


def _face_codes(cube: Cube) -> dict[str, str]:
    return {face: "".join(_COLOR_CODE[color] for color in cube.face(face)) for face in FACE_ORDER}


def test_solved_layout_and_color_indices_are_stable_for_later_search():
    assert FACE_ORDER == ("up", "right", "front", "down", "left", "back")
    assert COLOR_ORDER == ("white", "yellow", "red", "orange", "blue", "green")
    assert SOLVED_ARRAY.dtype == np.int8
    assert SOLVED_ARRAY.tolist() == ([0] * 9 + [2] * 9 + [5] * 9 + [1] * 9 + [3] * 9 + [4] * 9)
    assert not SOLVED_ARRAY.flags.writeable
    assert Cube().is_solved()
    assert is_solved_state(SOLVED_ARRAY)


def test_state_array_round_trip_is_exact_and_returns_independent_snapshots():
    cube = Cube().apply_algorithm("R U F2 L' D B2")
    numeric = cube.to_array()
    named = array_to_state(numeric)

    assert np.array_equal(state_to_array(named), numeric)
    assert Cube.from_array(numeric) == cube
    assert Cube.from_state(named) == cube

    numeric[0] = numeric[1]
    named["up"][0] = named["up"][1]
    assert Cube.from_array(cube.to_array()) == cube
    assert cube.state == array_to_state(cube.to_array())


@pytest.mark.parametrize(
    "array,error",
    [
        (np.zeros(53, dtype=np.int8), "shape"),
        (np.zeros(54, dtype=np.int8), "exactly nine"),
        (np.linspace(0, 5, 54), "integer"),
        (np.asarray([0] * 53 + [6], dtype=np.int8), "0..5"),
    ],
)
def test_array_conversion_rejects_structurally_invalid_states(array, error):
    with pytest.raises((TypeError, ValueError), match=error):
        array_to_state(array)


def test_array_conversion_rejects_noncanonical_centers():
    invalid = SOLVED_ARRAY.copy()
    invalid[4], invalid[13] = invalid[13], invalid[4]
    with pytest.raises(ValueError, match="centers"):
        Cube.from_array(invalid)


def test_named_state_conversion_rejects_missing_faces_lengths_and_colors():
    solved = Cube().state

    missing = dict(solved)
    missing.pop("back")
    with pytest.raises(ValueError, match="missing"):
        state_to_array(missing)

    short = Cube().state
    short["front"].pop()
    with pytest.raises(ValueError, match="exactly nine"):
        state_to_array(short)

    unknown = Cube().state
    unknown["front"][0] = "purple"
    with pytest.raises(ValueError, match="unknown color"):
        state_to_array(unknown)


def test_empty_algorithm_and_independent_copy_are_identity_operations():
    cube = Cube()
    copied = cube.copy()
    returned = cube.apply_algorithm("")

    assert returned is cube
    assert cube == copied
    copied.apply_move("R")
    assert cube.is_solved()
    assert not copied.is_solved()


@pytest.mark.parametrize("face", tuple("URFDLB"))
def test_four_quarter_turns_and_move_inverse_are_identity(face):
    assert Cube().apply_algorithm(" ".join([face] * 4)).is_solved()
    assert Cube().apply_algorithm(f"{face} {face}'").is_solved()
    assert Cube().apply_algorithm(f"{face}' {face}").is_solved()
    assert Cube().apply_algorithm(f"{face}2 {face}2").is_solved()


@pytest.mark.parametrize("face", tuple("URFDLB"))
def test_double_and_prime_turns_match_repeated_clockwise_turns(face):
    assert Cube().apply_move(f"{face}2") == Cube().apply_algorithm(f"{face} {face}")
    assert Cube().apply_move(f"{face}'") == Cube().apply_algorithm(f"{face} {face} {face}")


def test_known_r_turn_matches_canonical_face_orientation():
    cube = Cube().apply_move("R")
    assert _face_codes(cube) == {
        "up": "WWGWWGWWG",
        "right": "RRRRRRRRR",
        "front": "GGYGGYGGY",
        "down": "YYBYYBYYB",
        "left": "OOOOOOOOO",
        "back": "WBBWBBWBB",
    }


def test_known_sexy_move_sequence_matches_reference_state():
    cube = Cube().apply_algorithm("R U R' U'")
    assert _face_codes(cube) == {
        "up": "WWOWWGWWG",
        "right": "RRWBRRWRR",
        "front": "GGYGGWGGG",
        "down": "YYRYYYYYY",
        "left": "BOOOOOOOO",
        "back": "BRRBBBBBB",
    }
    assert not cube.is_solved()


def test_inverse_of_known_sequence_restores_the_start_state():
    algorithm = "F R U R' U' F' D2 L B' U2"
    cube = Cube().apply_algorithm(algorithm)
    cube.apply_algorithm(invert_algorithm(algorithm))
    assert cube.is_solved()


def test_seeded_synthetic_sequences_round_trip_through_their_inverse():
    random_source = random.Random(20260723)
    move_tokens = tuple(MOVE_PERMUTATIONS)

    for _ in range(100):
        algorithm = [random_source.choice(move_tokens) for _ in range(random_source.randint(1, 80))]
        cube = Cube().apply_algorithm(algorithm)
        assert not np.shares_memory(cube.to_array(), SOLVED_ARRAY)
        cube.apply_algorithm(invert_algorithm(algorithm))
        assert cube.is_solved()


def test_all_move_permutations_are_bijections_and_preserve_cube_invariants():
    expected_positions = np.arange(54, dtype=np.intp)
    for token, permutation in MOVE_PERMUTATIONS.items():
        assert permutation.dtype == np.intp
        assert not permutation.flags.writeable
        assert np.array_equal(np.sort(permutation), expected_positions), token

        moved = apply_move_to_array(SOLVED_ARRAY, token)
        assert moved.dtype == np.int8
        assert sorted(np.bincount(moved, minlength=6).tolist()) == [9] * 6
        assert moved[4::9].tolist() == SOLVED_ARRAY[4::9].tolist()
        assert np.array_equal(moved, Cube().apply_move(token).to_array())
        assert permutation_for_move(token) is permutation

from __future__ import annotations

import pytest

from cubed_core.decode_diagnostics import (
    DIAGNOSTIC_SCHEMA,
    DIAGNOSTIC_SCHEMA_VERSION,
    MAX_SEQUENCE_MOVES,
    DecodeDiagnosticsError,
    build_editdist_diagnostic,
)


def test_identical_sequences_report_zero_distance_and_all_equal_ops() -> None:
    moves = ["U", "R", "U'", "R'"]
    diagnostic = build_editdist_diagnostic(moves, moves)
    assert diagnostic["schema"] == DIAGNOSTIC_SCHEMA
    assert diagnostic["schema_version"] == DIAGNOSTIC_SCHEMA_VERSION
    assert diagnostic["diagnostic_only"] is True
    assert diagnostic["reference"] == "ble-teacher"
    assert diagnostic["distance"] == 0
    assert diagnostic["decoded_length"] == 4
    assert diagnostic["reference_length"] == 4
    assert [op["op"] for op in diagnostic["ops"]] == ["equal"] * 4
    for index, op in enumerate(diagnostic["ops"]):
        assert op["decoded"] == moves[index]
        assert op["reference"] == moves[index]
        assert op["index_decoded"] == index
        assert op["index_reference"] == index


def test_single_substitution_reports_distance_one() -> None:
    decoded = ["U", "R", "U'", "R'"]
    reference = ["U", "F", "U'", "R'"]
    diagnostic = build_editdist_diagnostic(decoded, reference)
    assert diagnostic["distance"] == 1
    ops = diagnostic["ops"]
    assert [op["op"] for op in ops] == ["equal", "substitute", "equal", "equal"]
    substitution = ops[1]
    assert substitution["decoded"] == "R"
    assert substitution["reference"] == "F"
    assert substitution["index_decoded"] == 1
    assert substitution["index_reference"] == 1


def test_insert_and_delete_mix() -> None:
    # A leading token in decoded with no reference counterpart (insert) and a
    # trailing reference token decoded never emits (delete) is strictly
    # cheaper here than any substitution-based alignment: "U R F" -> "R F D"
    # has edit distance 2 (drop the leading U, add the trailing D), not 3.
    decoded = ["U", "R", "F"]
    reference = ["R", "F", "D"]
    diagnostic = build_editdist_diagnostic(decoded, reference)
    ops = diagnostic["ops"]
    kinds = [op["op"] for op in ops]
    assert "insert" in kinds
    assert "delete" in kinds
    assert diagnostic["distance"] == sum(1 for k in kinds if k != "equal")

    insert_ops = [op for op in ops if op["op"] == "insert"]
    for op in insert_ops:
        assert op["reference"] is None
        assert op["index_reference"] is None
        assert op["decoded"] is not None

    delete_ops = [op for op in ops if op["op"] == "delete"]
    for op in delete_ops:
        assert op["decoded"] is None
        assert op["index_decoded"] is None
        assert op["reference"] is not None

    # Ops must reconstruct both sequences in order when replayed.
    reconstructed_decoded = [op["decoded"] for op in ops if op["decoded"] is not None]
    reconstructed_reference = [op["reference"] for op in ops if op["reference"] is not None]
    assert reconstructed_decoded == decoded
    assert reconstructed_reference == reference


def test_empty_reference_marks_every_decoded_move_as_insert() -> None:
    decoded = ["U", "R", "U'"]
    diagnostic = build_editdist_diagnostic(decoded, [])
    assert diagnostic["distance"] == 3
    assert diagnostic["reference_length"] == 0
    assert diagnostic["decoded_length"] == 3
    assert [op["op"] for op in diagnostic["ops"]] == ["insert"] * 3


def test_empty_decoded_marks_every_reference_move_as_delete() -> None:
    reference = ["U", "R", "U'"]
    diagnostic = build_editdist_diagnostic([], reference)
    assert diagnostic["distance"] == 3
    assert diagnostic["decoded_length"] == 0
    assert diagnostic["reference_length"] == 3
    assert [op["op"] for op in diagnostic["ops"]] == ["delete"] * 3


def test_both_empty_is_a_legitimate_zero_distance_document() -> None:
    diagnostic = build_editdist_diagnostic([], [])
    assert diagnostic["distance"] == 0
    assert diagnostic["ops"] == []


def test_cap_exceeded_on_decoded_side_raises_a_clear_error() -> None:
    decoded = ["U"] * (MAX_SEQUENCE_MOVES + 1)
    with pytest.raises(DecodeDiagnosticsError, match="diagnostic cap"):
        build_editdist_diagnostic(decoded, ["U"])


def test_cap_exceeded_on_reference_side_raises_a_clear_error() -> None:
    reference = ["U"] * (MAX_SEQUENCE_MOVES + 1)
    with pytest.raises(DecodeDiagnosticsError, match="diagnostic cap"):
        build_editdist_diagnostic(["U"], reference)


def test_malformed_move_token_is_rejected_with_its_index() -> None:
    with pytest.raises(DecodeDiagnosticsError, match=r"decoded moves\[1\]"):
        build_editdist_diagnostic(["U", "u"], ["U"])


def test_wide_and_slice_and_rotation_tokens_are_rejected_not_silently_dropped() -> None:
    for token in ("u", "M", "x", "Rw", "U3"):
        with pytest.raises(DecodeDiagnosticsError):
            build_editdist_diagnostic([token], ["U"])


def test_non_list_inputs_are_rejected() -> None:
    with pytest.raises(DecodeDiagnosticsError):
        build_editdist_diagnostic("U R", ["U", "R"])
    with pytest.raises(DecodeDiagnosticsError):
        build_editdist_diagnostic(["U", "R"], "U R")

"""Focused contracts for decoder-owned reconstruction checkpoints."""

from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pytest

# The base OSS test environment intentionally excludes the research-GPU
# SciPy extra. Scrub's import closure references the assignment helper, while
# these pure checkpoint tests never execute it.
try:
    import scipy.optimize  # type: ignore[import-not-found]  # noqa: F401
except ModuleNotFoundError:
    scipy_was_stubbed = True
    scipy_stub = types.ModuleType("scipy")
    optimize_stub = types.ModuleType("scipy.optimize")

    def _unused_linear_sum_assignment(*_args, **_kwargs):
        raise AssertionError("checkpoint tests must not run SciPy assignment")

    optimize_stub.linear_sum_assignment = _unused_linear_sum_assignment
    scipy_stub.optimize = optimize_stub
    sys.modules["scipy"] = scipy_stub
    sys.modules["scipy.optimize"] = optimize_stub
else:
    scipy_was_stubbed = False

CM = importlib.import_module("analysis.cubemodel")
LLC = importlib.import_module("detect.ll_completion_live")
scrub = importlib.import_module("detect.scrub_decode")
trellis_gt = importlib.import_module("scripts.trellis_gt")

if scipy_was_stubbed:
    sys.modules.pop("scipy.optimize", None)
    sys.modules.pop("scipy", None)


def test_checkpoint_groups_keep_windows_atomic_and_merge_terminal_bridge() -> None:
    groups = scrub._reconstruction_checkpoint_groups(
        [
            {
                "tokens": ["R", "U"],
                "land": 0,
                "oms": [object(), object()],
                "checkpoint_frame": 18,
            },
            {
                "tokens": ["F"],
                "land": 2,
                "oms": [object()],
                "checkpoint_frame": 39,
            },
        ],
        ["D", "L"],
        final_checkpoint_frame=39,
    )

    assert groups == [
        {"frame": 18, "move_count": 2},
        {"frame": 39, "move_count": 3},
    ]


def test_checkpoint_groups_reject_nonchronological_windows() -> None:
    with pytest.raises(ValueError, match="not chronological"):
        scrub._reconstruction_checkpoint_groups(
            [
                {"tokens": ["R"], "checkpoint_frame": 20},
                {"tokens": ["U"], "checkpoint_frame": 10},
            ],
            [],
            final_checkpoint_frame=30,
        )


def test_ll_replacement_uses_one_final_checkpoint_without_changing_land() -> None:
    selected = LLC.LLLiveHypothesis(
        source="table",
        literal_moves=("R", "U"),
        canonical_moves=("R", "U"),
        action_frames=(12, 20),
        move_frames=(12, 20),
        move_om_indices=(0, 0),
        score=0.0,
    )
    decision = LLC.LLLiveDecision(
        status="selected",
        reason="test",
        enumerated_candidates=1,
        timed_candidates=1,
        scored_hypotheses=1,
        retained=(selected,),
        selected=selected,
    )
    initial = CM.apply_seq(CM.SOLVED, ("U'", "R'"))

    _, windows, word, replay = scrub._validated_ll_selected_mutation(
        decision=decision,
        action_slots=(
            LLC.LLActionSlot("move", 12, 11, 13),
            LLC.LLActionSlot("move", 20, 19, 21),
        ),
        orientations=[object()],
        onset_frame=5,
        final_frame=30,
        span_ends=[10, 20, 30],
        prefix_windows=[],
        expected_prefix_moves=(),
        init_arr=initial,
        ll_target=CM.SOLVED,
    )

    assert word == ("R", "U")
    assert np.array_equal(replay, CM.SOLVED)
    assert [window["land"] for window in windows] == [1, 1]
    assert [window["checkpoint_frame"] for window in windows] == [30, 30]


def test_bad_checkpoint_metadata_does_not_change_ll_selection() -> None:
    selected = LLC.LLLiveHypothesis(
        source="table",
        literal_moves=("R", "U"),
        canonical_moves=("R", "U"),
        action_frames=(12, 20),
        move_frames=(12, 20),
        move_om_indices=(0, 0),
        score=0.0,
    )
    decision = LLC.LLLiveDecision(
        status="selected",
        reason="test",
        enumerated_candidates=1,
        timed_candidates=1,
        scored_hypotheses=1,
        retained=(selected,),
        selected=selected,
    )
    initial = CM.apply_seq(CM.SOLVED, ("U'", "R'", "F'"))

    _, windows, word, replay = scrub._validated_ll_selected_mutation(
        decision=decision,
        action_slots=(
            LLC.LLActionSlot("move", 12, 11, 13),
            LLC.LLActionSlot("move", 20, 19, 21),
        ),
        orientations=[object()],
        onset_frame=5,
        final_frame=30,
        span_ends=[10, 20, 30],
        prefix_windows=[
            {
                "tokens": ["F"],
                "land": 0,
                "oms": [object()],
                "checkpoint_frame": "not-a-frame",
            }
        ],
        expected_prefix_moves=("F",),
        init_arr=initial,
        ll_target=CM.SOLVED,
    )

    assert word == ("F", "R", "U")
    assert np.array_equal(replay, CM.SOLVED)
    with pytest.raises(ValueError, match="checkpoint frame is malformed"):
        scrub._reconstruction_checkpoint_groups(
            windows,
            [],
            final_checkpoint_frame=30,
        )


def test_backward_decode_suppresses_forward_clock_checkpoints() -> None:
    assert trellis_gt._reconstruction_checkpoints_valid(
        reach=True,
        backward=False,
        raw_moves=["R", "U"],
        canonical_moves=["R", "U"],
    )
    assert not trellis_gt._reconstruction_checkpoints_valid(
        reach=True,
        backward=True,
        raw_moves=["R", "U"],
        canonical_moves=["R", "U"],
    )

from __future__ import annotations

from cubed_core.job_runtime import parse_stage_markers


def test_stage_markers_return_empty_state_without_a_marker() -> None:
    info = parse_stage_markers("ordinary runner output\n")

    assert info.stage is None
    assert info.stages_seen == ()
    assert info.stage_progress is None


def test_stage_markers_track_latest_stage_and_progress() -> None:
    info = parse_stage_markers(
        "[cubed-core:stage] upload\n"
        "[cubed-core:stage] pose 5/200\n"
        "[cubed-core:stage] pose 10/200\n"
        "[cubed-core:stage] dump\n"
    )

    assert info.stage == "dump"
    assert info.stages_seen == ("upload", "pose", "dump")
    assert info.stage_progress == (10, 200)


def test_stage_markers_deduplicate_first_seen_order() -> None:
    info = parse_stage_markers(
        "[cubed-core:stage] upload\n[cubed-core:stage] model-load\n[cubed-core:stage] upload\n"
    )

    assert info.stage == "upload"
    assert info.stages_seen == ("upload", "model-load")


def test_stage_markers_preserve_unknown_valid_tokens() -> None:
    info = parse_stage_markers("[cubed-core:stage] some-custom-phase\n")

    assert info.stage == "some-custom-phase"
    assert info.stages_seen == ("some-custom-phase",)


def test_stage_markers_ignore_malformed_or_embedded_lines() -> None:
    info = parse_stage_markers(
        "[cubed-core:stage]\n"
        "[cubed-core:stage] pose 5\n"
        "prefixed [cubed-core:stage] pose\n"
        "[cubed-core:stage] pose suffixed\n"
    )

    assert info.stage is None
    assert info.stages_seen == ()
    assert info.stage_progress is None

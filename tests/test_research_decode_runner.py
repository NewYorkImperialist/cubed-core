"""Tests for the public research decode runner + from-video reads bridge.

Dry-run / arg-parsing only — never a functional decode or a video read (those
run on the GPU box). They pin:

* the CFG self-stamp token stream: the runner's ``cfg=<name>[cksum:<HASH>]``
  reproduces run_decode.sh's mechanism exactly — a POSIX ``cksum`` over the
  canonical flag tokens plus the ``ENV:MICROREST=1,CLUSTERFB=1``
  behaviour toggles, with user passthrough flags folded into their own
  ``+extras[cksum:...]`` term; and
* the from-video bridge assembles the three camera-input generator invocations
  (geo_read / gen_motion_events / extract_alignfeat) in the exact shape.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DECODE_RUNNER = REPOSITORY_ROOT / "scripts" / "run_research_decode.sh"
READS_RUNNER = REPOSITORY_ROOT / "scripts" / "run_research_reads.sh"
RUNTIME_CONFIG = REPOSITORY_ROOT / "config" / "decode-runtime-v1.json"


def _profiles() -> dict[str, dict[str, object]]:
    return json.loads(RUNTIME_CONFIG.read_text(encoding="utf-8"))["profiles"]


def _cksum(text: str) -> str:
    out = subprocess.run(
        ["cksum"], input=text.encode(), capture_output=True, check=True
    ).stdout.decode()
    return out.split()[0]


def _dry_run(runner: Path, args: list[str], tmp_path: Path) -> str:
    reads = tmp_path / "reads.pkl"
    events = tmp_path / "events.json"
    reads.write_bytes(b"")
    events.write_bytes(b"")
    result = subprocess.run(
        ["bash", str(runner), *args],
        capture_output=True,
        text=True,
        cwd=REPOSITORY_ROOT,
        env={
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "CUBED_READS": str(reads),
            "CUBED_EVENTS": str(events),
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def _cfg_stamp(stdout: str) -> str:
    for line in stdout.splitlines():
        if " cfg=" in line:
            return line.split(" cfg=", 1)[1].strip()
    raise AssertionError(f"no cfg= stamp line in:\n{stdout}")


# --- Stamp token stream ------------------------------------------------------


def test_stamp_token_stream_matches_checked_manifest() -> None:
    """The base cksum equals the checked local-camera profile."""
    profiles = _profiles()
    local = profiles["local_camera_v1"]
    evaluation = profiles["canonical_eval_reference"]
    assert "--final-from-gt" not in local["flags"]
    assert "--final-from-gt" in evaluation["flags"]
    assert _cksum(local["cfg_hash_input"]) == local["cfg_hash"]
    assert _cksum(evaluation["cfg_hash_input"]) == evaluation["cfg_hash"]
    assert local["cfg_hash"] != evaluation["cfg_hash"]


def test_bare_run_stamps_base_cksum(tmp_path: Path) -> None:
    stamp = _cfg_stamp(_dry_run(DECODE_RUNNER, ["gtD1s"], tmp_path))
    profiles = _profiles()
    assert stamp == (
        f"{profiles['local_camera_v1']['name']}[cksum:{profiles['local_camera_v1']['cfg_hash']}]"
    )


def test_scramble_override_is_an_input_not_a_config_extra(tmp_path: Path) -> None:
    reads = tmp_path / "reads.pkl"
    events = tmp_path / "events.json"
    reads.write_bytes(b"")
    events.write_bytes(b"")
    result = subprocess.run(
        ["bash", str(DECODE_RUNNER), "capture"],
        capture_output=True,
        text=True,
        cwd=REPOSITORY_ROOT,
        env={
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "CUBED_READS": str(reads),
            "CUBED_EVENTS": str(events),
            "CUBED_SCRAMBLE": "R U R' U'",
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--scramble R U R' U'" in result.stdout
    profiles = _profiles()
    assert _cfg_stamp(result.stdout) == (
        f"{profiles['local_camera_v1']['name']}[cksum:{profiles['local_camera_v1']['cfg_hash']}]"
    )


def test_extra_flags_fold_into_extras_cksum(tmp_path: Path) -> None:
    stamp = _cfg_stamp(_dry_run(DECODE_RUNNER, ["gtD1s", "--final-from-gt"], tmp_path))
    extras = _cksum("--final-from-gt")
    profiles = _profiles()
    assert stamp == (
        f"{profiles['local_camera_v1']['name']}"
        f"[cksum:{profiles['local_camera_v1']['cfg_hash']}]"
        f"+extras[cksum:{extras}]"
    )
    assert profiles["canonical_eval_reference"]["cfg_hash"] not in stamp


def test_footer_restamp_matches_header(tmp_path: Path) -> None:
    # Dry run stops before the timed footer; the header stamp is the pinned one.
    # The footer reuses the same $CFG_STAMP variable, verified by static presence.
    runner = DECODE_RUNNER.read_text(encoding="utf-8")
    assert "cfg=$CFG_STAMP wall=${_decode_wall}s" in runner


# --- From-video bridge shape -------------------------------------------------


def test_from_video_bridge_assembles_e2e_shapes(tmp_path: Path) -> None:
    result = subprocess.run(
        ["bash", str(READS_RUNNER), "gtD1s", "--video", "/captures/gtD1s.mp4"],
        capture_output=True,
        text=True,
        cwd=REPOSITORY_ROOT,
        env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    out = result.stdout
    # Dry-run shape: GPU reads/warp stay enabled; exact-video NVDEC selection
    # happens only when execution is requested.
    assert "geo_read.py --tag gtD1s --video /captures/gtD1s.mp4" in out
    assert "--centroids-json calibration_gan12.json" in out
    assert "--gpu-reads --gpu-warp --out /tmp/reads_gtD1s_occaware.pkl" in out
    assert "--gpu-decode" not in out
    assert "video-decode=probe-on-execute policy=auto" in out
    # motion events + alignfeat generators, staged to the decode-resolved names.
    assert "gen_motion_events.py gtD1s /tmp/motion_events_gtD1s.json" in out
    assert "CUBED_ALIGNED_MODEL=" in out
    assert (
        "extract_alignfeat.py --tag gtD1s --video /captures/gtD1s.mp4 "
        "--out /tmp/alignfeat_gtD1s_new.npz"
    ) in out
    # ends by handing the three artifacts to the decode runner.
    assert "run_research_decode.sh gtD1s" in out
    # dry run by default (no execution on this host).
    assert "DRY RUN" in result.stderr

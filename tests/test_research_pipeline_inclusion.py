"""Tests for the research decode pipeline as first-class repo code.

Import/smoke only — never a functional decode (that runs on a GPU box). They
verify: the decode + video->reads import closures resolve from the repository
layout (repo root + ``scripts`` on ``sys.path``), the local-camera profile is
the evaluation profile without its teacher-only endpoint, the shell runner
matches the checked runtime manifest, the research code carries no secrets, and
the rights-cleared alg-set data assets ship.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
import textwrap
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_CONFIG = REPOSITORY_ROOT / "config" / "decode-runtime-v1.json"

# The research decode/reads code, now living at the repository root.
RESEARCH_CODE_DIRS = ("detect", "core", "analysis")
RESEARCH_SCRIPTS = (
    "calib_util.py",
    "cell_common.py",
    "check_nvdec.py",
    "extract_alignfeat.py",
    "gen_motion_events.py",
    "geo_read.py",
    "gpu_decode.py",
    "gpu_reads.py",
    "occ_mask.py",
    "onnx_dynamic_batch.py",
    "trellis_gt.py",
)


def _research_code_files() -> list[Path]:
    files: list[Path] = []
    for name in RESEARCH_CODE_DIRS:
        files.extend(sorted((REPOSITORY_ROOT / name).rglob("*.py")))
    files.extend(REPOSITORY_ROOT / "scripts" / name for name in RESEARCH_SCRIPTS)
    return files


# --- Layout: the anchor files are present where the runner expects them. -----


def test_anchor_files_present_at_repo_root() -> None:
    for rel in (
        "detect/scrub_span_view.py",
        "detect/trellis_tracker.py",
        "detect/scrub_decode.py",
        "scripts/trellis_gt.py",
    ):
        assert (REPOSITORY_ROOT / rel).is_file(), rel


# --- Import closures resolve from the repository layout. ---------------------

_STUB_PREAMBLE = """
import sys, types
sys.path[:0] = [{root!r}, {scripts!r}]
class _Any(types.ModuleType):
    def __getattr__(self, n):
        if n.startswith("__") and n.endswith("__"):
            raise AttributeError(n)
        return _Any(n)
    def __call__(self, *a, **k):
        return _Any("x")
for name in {stubs!r}:
    sys.modules.setdefault(name, _Any(name))
"""


def _run_closure(program: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        cwd=REPOSITORY_ROOT,
    )


def test_decode_import_closure_resolves() -> None:
    """Import every decode anchor with optional backends stubbed.

    Success proves the decode closure (anchors + their detect/core/analysis
    dependencies + the bare ``scripts`` helpers) resolves from the repo layout.
    """
    preamble = _STUB_PREAMBLE.format(
        root=str(REPOSITORY_ROOT),
        scripts=str(REPOSITORY_ROOT / "scripts"),
        stubs=[
            "onnxruntime",
            "torch",
            "torch.nn",
            "torch.nn.functional",
            "scipy",
            "scipy.optimize",
            "scipy.spatial",
            "scipy.spatial.transform",
            "scipy.ndimage",
        ],
    )
    program = preamble + textwrap.dedent(
        """
        import importlib
        for m in ["detect.scrub_span_view", "detect.trellis_tracker",
                  "detect.scrub_decode", "trellis_gt"]:
            importlib.import_module(m)
        print("COMPLETE")
        """
    )
    result = _run_closure(program)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "COMPLETE" in result.stdout


def test_video_bridge_import_closure_resolves() -> None:
    """Import the video->reads bridge entrypoints with backends stubbed.

    ``gen_motion_events`` parses ``sys.argv`` at import (a CLI-only tool), so it
    is not imported here; the GPU helpers import lazily under ``--gpu-*`` in
    ``geo_read``, so they are imported explicitly to verify their own closure.
    """
    preamble = _STUB_PREAMBLE.format(
        root=str(REPOSITORY_ROOT),
        scripts=str(REPOSITORY_ROOT / "scripts"),
        stubs=[
            "onnxruntime",
            "torch",
            "torch.nn",
            "torch.nn.functional",
            "cv2",
            "onnx",
            "onnx.numpy_helper",
            "scipy",
            "scipy.optimize",
            "scipy.spatial",
            "scipy.spatial.transform",
            "scipy.ndimage",
        ],
    )
    program = preamble + textwrap.dedent(
        """
        import importlib
        for m in ["geo_read", "extract_alignfeat", "gpu_reads", "gpu_decode",
                  "onnx_dynamic_batch"]:
            importlib.import_module(m)
        print("COMPLETE")
        """
    )
    result = _run_closure(program)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "COMPLETE" in result.stdout


# --- Checked runtime profile == shell runner. --------------------------------


def _profiles() -> dict[str, dict[str, object]]:
    return json.loads(RUNTIME_CONFIG.read_text(encoding="utf-8"))["profiles"]


def test_local_profile_is_eval_without_teacher_terminal() -> None:
    profiles = _profiles()
    evaluation = list(profiles["canonical_eval_reference"]["flags"])
    local = list(profiles["local_camera_v1"]["flags"])
    assert "--final-from-gt" in evaluation
    assert local == [token for token in evaluation if token != "--final-from-gt"]


def test_shell_runner_matches_checked_local_profile(tmp_path: Path) -> None:
    reads = tmp_path / "reads.pkl"
    events = tmp_path / "events.json"
    reads.write_bytes(b"")
    events.write_bytes(b"")
    result = subprocess.run(
        ["bash", str(REPOSITORY_ROOT / "scripts" / "run_research_decode.sh"), "capture"],
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
    flags_line = next(
        line
        for line in result.stdout.splitlines()
        if line.startswith("[run_research_decode] FLAGS: ")
    )
    actual = shlex.split(flags_line.split("FLAGS: ", 1)[1])
    assert actual == _profiles()["local_camera_v1"]["flags"]


def test_video_bridge_probes_nvdec_before_enabling_it() -> None:
    runner = (REPOSITORY_ROOT / "scripts" / "run_research_reads.sh").read_text(encoding="utf-8")
    geo_command = runner.split("GEO_CMD=(", 1)[1].split("EVENTS_CMD=(", 1)[0]
    assert "--gpu-reads" in geo_command
    assert "--gpu-warp" in geo_command
    assert "--gpu-decode" not in geo_command
    assert 'python3 "$NVDEC_CHECK" --video "$VIDEO"' in runner
    assert "GEO_CMD+=(--gpu-decode)" in runner
    assert "host (NVDEC unavailable for this video)" in runner


def test_nvdec_probe_rejects_a_missing_video(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "check_nvdec.py"),
            "--video",
            str(tmp_path / "missing.mp4"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "video is not a file" in result.stderr


def test_nvdec_probe_uses_gpu_decoder_frame_api() -> None:
    source = (REPOSITORY_ROOT / "scripts" / "check_nvdec.py").read_text()

    assert "next(decoder.frames())" in source
    assert "next(iter(decoder))" not in source


def test_gpu_decoder_uses_packet_api_instead_of_fragile_simple_decoder() -> None:
    source = (REPOSITORY_ROOT / "scripts" / "gpu_decode.py").read_text()

    assert "nvc.CreateDemuxer(filename=path)" in source
    assert "nvc.CreateDecoder(" in source
    assert "nvc.SimpleDecoder(" not in source


def test_camera_only_decode_keeps_teacher_metrics_optional() -> None:
    decoder = (REPOSITORY_ROOT / "scripts" / "trellis_gt.py").read_text(encoding="utf-8")
    assert '_teacher_moves = sess.get("moves")' in decoder
    assert "teacher_reference=not-provided" in decoder
    assert "evaluation=reach-ll-onset-with-correct-pre-ll-state" in decoder
    assert "status=unavailable reason=teacher-reference-not-provided" in decoder
    assert decoder.index("diagnostic_similarity = None") < decoder.rindex(
        "return diagnostic_similarity"
    )


# --- Secrets guard + data-asset rights. -------------------------------------

_EMAIL_RE = re.compile(rb"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def test_research_code_has_no_secrets() -> None:
    for path in _research_code_files():
        raw = path.read_bytes()
        rel = path.relative_to(REPOSITORY_ROOT).as_posix()
        assert not _EMAIL_RE.search(raw), f"email in {rel}"
        assert b"ssh-rsa" not in raw, rel
        assert b"BEGIN PRIVATE KEY" not in raw, rel
        assert b"BEGIN OPENSSH PRIVATE KEY" not in raw, rel
        assert b"/Users/" not in raw, rel


def test_rights_clean_data_assets_shipped() -> None:
    algsets = REPOSITORY_ROOT / "analysis" / "data" / "algsets"
    shipped = {p.name for p in algsets.glob("*.json")}
    for expected in (
        "OLL.json",
        "PLL.json",
        "COLL.json",
        "CMLL.json",
        "2-Look-OLL.json",
        "2-Look-PLL.json",
        "2-Look-CMLL.json",
    ):
        assert expected in shipped, expected
    assert (REPOSITORY_ROOT / "analysis" / "data" / "oll_cases.json").is_file()
    assert (REPOSITORY_ROOT / "analysis" / "data" / "pll_cases.json").is_file()


def test_contested_algtrainer_flat_is_not_shipped() -> None:
    assert not (REPOSITORY_ROOT / "analysis" / "data" / "algsets" / "algtrainer_flat.json").exists()

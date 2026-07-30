#!/bin/bash
# ============================================================================
# Research decode runner — the camera-to-moves decode entrypoint.
#
# Assembles the canonical production flag set and invokes the research runner
# (scripts/trellis_gt.py). Production receives the starting scramble from the
# app and uses the runner's fixed solved task target, so the eval-only
# --final-from-gt terminal is never assembled here.
#
# The checked runtime manifest defines the profiles and hashes. This runner
# assembles the local-camera profile; tests compare its emitted flags and stamp
# directly with config/decode-runtime-v1.json.
#
# DRY RUN by default. A real run needs a CUDA box, the downloaded release assets
# (tracker models, read-trust artifact, calibration), and the per-tag decode
# inputs staged below. Set CUBED_RESEARCH_DECODE_EXECUTE=1 to actually invoke it.
# Heavy compute must run on a GPU box, never a laptop.
# ============================================================================
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: $0 <tag> [extra trellis_gt.py flags...]" >&2
  exit 2
fi
tag="$1"; shift

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TG="$REPO_ROOT/scripts/trellis_gt.py"

if [ ! -f "$TG" ]; then
  echo "run_research_decode: runner missing: $TG" >&2
  exit 1
fi

# Distilled numpy read-trust model (gitignored research artifact, NOT shipped).
TRUST_NPZ="${CUBED_TRUST_NPZ:-datasets/read_trust/trust_v1_numpy.npz}"

# Per-tag input artifacts (reads pickle + motion-events JSON staged under /tmp).
# Override with CUBED_READS / CUBED_EVENTS for explicit artifact paths.
RD="${CUBED_READS:-/tmp/reads_${tag}_occaware.pkl}"
[ -f "$RD" ] || RD="/tmp/reads_${tag}_OCC.pkl"
[ -f "$RD" ] || { echo "run_research_decode: no reads artifact for $tag (set CUBED_READS)" >&2; exit 1; }
EV="${CUBED_EVENTS:-/tmp/motion_events_${tag}.json}"
[ -f "$EV" ] || { echo "run_research_decode: no events artifact for $tag (set CUBED_EVENTS)" >&2; exit 1; }

# --- Canonical production flags (the canonical CFG MINUS --final-from-gt). ---
BASE="--om-pf-reads --dfinal 5 --deep-count-bound --om-rescue-deep 3 --om-rescue-topk 2 --ll-prior 3"
MOVEGATE="--move-gate --move-gate-allframes --move-gate-margin 1 --align-gate --align-gate-mode or --align-transition-sample"
DGAP_TRUST="--om-pf-ball dgap --trust-soft $TRUST_NPZ --misfit-thr 0"
SCRUB="--scrub-decode --scrub-om-stateful"
ROBUST="--znorm-gates"
CFG="$BASE $MOVEGATE $DGAP_TRUST $SCRUB $ROBUST"

# --- CFG self-stamp — every run records the exact config that produced it. ---
# The config cksum covers the flag tokens AND the behaviour-env toggles
# (ROBUST_ENV), tag-independent (per-tag paths are appended AFTER the stamp,
# never folded in). The production CFG omits the eval-only --final-from-gt.
CFG_NAME="cubed-core-local-camera-v1"
ROBUST_ENV="MICROREST=1,CLUSTERFB=1"
CFG_HASH_INPUT="$CFG ENV:$ROBUST_ENV"
CFG_HASH=$(printf '%s' "$CFG_HASH_INPUT" | cksum | cut -d' ' -f1)
# Extras stamp: user passthrough flags are NOT covered by CFG_HASH, so a flagged
# run folds them into their OWN cksum and appends it — a bare run with no extras
# is byte-identical to before. A user who passes --final-from-gt as an extra
# therefore gets ...[cksum:$CFG_HASH]+extras[cksum:<extras>].
PASSTHROUGH=("$@")
EXTRAS_HASH=""
if [ ${#PASSTHROUGH[@]} -gt 0 ]; then
  EXTRAS_HASH=$(printf '%s' "${PASSTHROUGH[*]}" | cksum | cut -d' ' -f1)
  CFG_STAMP="${CFG_NAME}[cksum:$CFG_HASH]+extras[cksum:$EXTRAS_HASH]"
else
  CFG_STAMP="${CFG_NAME}[cksum:$CFG_HASH]"
fi

# Structured result lane (opt-in, env only, no new decode flags): when the
# caller sets CUBED_RESULT_JSON to an absolute path, the run also writes a
# decode-result-v1 document there. The runner is the authority on config
# stamps, so it exports its own stamp for the in-process build and overwrites
# the config block after the run. Nothing here fires for an unset variable.
if [ -n "${CUBED_RESULT_JSON:-}" ]; then
  export CUBED_CFG_NAME="$CFG_NAME"
  export CUBED_CFG_HASH="$CFG_HASH"
  export CUBED_CFG_HASH_INPUT="$CFG_HASH_INPUT"
  export CUBED_EXTRAS_HASH="$EXTRAS_HASH"
fi

# Behaviour env (folded into the CFG_HASH above) + pure-speed env (CFG-neutral,
# auto-falls-back to CPU when CUDA is absent).
export CUBED_SCRUB_MICROREST_CANDIDACY=1 CUBED_SCRUB_CLUSTER_FALLBACK=1
export CUBED_GPU_SCORING="${CUBED_GPU_SCORING:-1}" CUBED_GPU_MIN_STATES="${CUBED_GPU_MIN_STATES:-128}"
export CUBED_GPU_SCRUB_SCORE=1 CUBED_SCRUB_FAST_OM=1 CUBED_GPU_SCRUB_BATCH=1
export CUBED_GPU_SCRUB_FUSED_READS=1 CUBED_GPU_SCRUB_DEVICE_BEAM=1 CUBED_GPU_SCRUB_STATEFUL_DP=1
export CUBED_ABS_SEGMENT_CACHE=1 CUBED_GPU_TRUST_BATCH=1

# Import environment: repo root (so detect/core/analysis resolve) + scripts/ (so
# the bare trellis_gt/calib_util/cell_common/occ_mask imports resolve).
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"

# Six-color centroids for the capture's cube (the canonical runner passes its
# cube's calibration unconditionally; supply your capture's own calibration via
# CUBED_CENTROIDS).
CENTROIDS="${CUBED_CENTROIDS:-calibration_gan12.json}"

# A public capture can supply its app-given scramble directly instead of
# recreating the research-only walkthrough/session_<tag>.json layout.
INPUT_ARGS=()
if [ -n "${CUBED_SCRAMBLE:-}" ]; then
  INPUT_ARGS=(--scramble "$CUBED_SCRAMBLE")
fi

# Invocation shape:
#   python3 -u trellis_gt.py --tag <tag> --reads <pkl> --events-json <json> \
#     --centroids-json <calib> <CFG> <capture inputs> <extras> --full-seq
# ${arr[@]+"${arr[@]}"} = the set -u-safe empty-array expansion (bash 3.2, macOS).
CMD=(python3 -u "$TG" --tag "$tag" --reads "$RD" --events-json "$EV" \
  --centroids-json "$CENTROIDS" $CFG \
  ${INPUT_ARGS[@]+"${INPUT_ARGS[@]}"} \
  ${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"} --full-seq)

# Every run self-stamps its config so a number can NEVER be quoted config-blind.
# The `cfg=<name>[cksum:<HASH>]` substring is the stable, grep-able config stamp.
echo "[run_research_decode] profile=$CFG_NAME (camera-only; evaluation endpoint excluded)"
echo "[run_research_decode] PYTHONPATH=$PYTHONPATH"
# Input-identity stamps: the config cksum covers flag TOKENS (an env-overridden
# artifact path changes the stamp by design); these lines pin the artifact
# CONTENT so two runs are comparable regardless of where the files live.
for _art in "$RD" "$EV" "$TRUST_NPZ" "$CENTROIDS"; do
  [ -f "$_art" ] && echo "[run_research_decode] artifact $(basename "$_art") cksum=$(cksum < "$_art" | cut -d' ' -f1) bytes=$(wc -c < "$_art" | tr -d ' ')"
done
echo "[run_research_decode] tag=$tag reads=$(basename "$RD") events=$(basename "$EV") cfg=$CFG_STAMP"
echo "[run_research_decode] FLAGS: $CFG${PASSTHROUGH[*]:+ ${PASSTHROUGH[*]}}"
echo "[run_research_decode] command: ${CMD[*]}"

if [ "${CUBED_RESEARCH_DECODE_EXECUTE:-0}" != "1" ]; then
  echo "[run_research_decode] DRY RUN (set CUBED_RESEARCH_DECODE_EXECUTE=1 to execute on a CUDA box with release assets present)." >&2
  exit 0
fi

# Run (not exec) so the footer can re-stamp the config with the wall time:
#   `^ ... cfg=$CFG_STAMP wall=${_decode_wall}s`.
_decode_start=$SECONDS
if "${CMD[@]}"; then _decode_status=0; else _decode_status=$?; fi
_decode_wall=$((SECONDS - _decode_start))
echo "[run_research_decode] ^ tag=$tag cfg=$CFG_STAMP wall=${_decode_wall}s — config stamp for the result above"

# Config injection: the decoder cannot know the stamp this script assembled, so
# a successful armed run has its config block overwritten from the values above
# and the document re-validated. A failure here is loud, because a consumer must
# never read a result whose config block is not the runner's.
if [ -n "${CUBED_RESULT_JSON:-}" ] && [ "$_decode_status" -eq 0 ]; then
  if python3 -c 'import sys; import decode_result_emit as d; d.inject_config(sys.argv[1], name=sys.argv[2], cfg_hash=sys.argv[3], cfg_hash_input=sys.argv[4], extras_hash=sys.argv[5] or None)' \
      "$CUBED_RESULT_JSON" "$CFG_NAME" "$CFG_HASH" "$CFG_HASH_INPUT" "$EXTRAS_HASH"; then
    echo "[run_research_decode] result json stamped: $CUBED_RESULT_JSON (cfg=$CFG_STAMP)"
  else
    echo "[run_research_decode] result json config injection FAILED: $CUBED_RESULT_JSON" >&2
    _decode_status=1
  fi
fi

exit $_decode_status

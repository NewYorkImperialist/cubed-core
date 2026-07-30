#!/bin/bash
# ============================================================================
# From-video reads bridge — turns a capture into the three decode inputs
# (reads pkl + motion-events JSON + alignfeat npz), then hands off to
# scripts/run_research_decode.sh. It drives the video->reads generators —
#   $GR  = scripts/geo_read.py          (video -> reads)
#   $GME = scripts/gen_motion_events.py (reads -> events)
#   $EAF = scripts/extract_alignfeat.py (video -> alignfeat)
# and stages their outputs under the /tmp per-tag filenames the decode runner
# resolves.
#
# Invocation shapes:
#   geo_read:     python3 -u geo_read.py --tag <tag> --video <v> \
#                   --centroids-json <calib> --gpu-reads --gpu-warp \
#                   --out /tmp/reads_<tag>_occaware.pkl
#   gen_events:   python3 -u gen_motion_events.py <tag> /tmp/motion_events_<tag>.json
#   extract_af:   CUBED_ALIGNED_MODEL=<onnx> python3 -u extract_alignfeat.py \
#                   --tag <tag> --out /tmp/alignfeat_<tag>_new.npz
#   decode:       scripts/run_research_decode.sh <tag>
#
# HEAVY-COMPUTE = GPU BOX ONLY. geo_read's torch import is optional (CPU-only
# hosts fall back), but a REAL run belongs on a CUDA machine (see
# docs/CLOUD_GPU.md) — never a laptop. DRY RUN by default: prints the assembled
# commands. Set CUBED_RESEARCH_DECODE_EXECUTE=1 to actually invoke them (needs
# the release-asset models and a video resolvable by its tag).
#
# Video resolution: the required --video path is passed to both geo_read and
# extract_alignfeat. gen_motion_events reads /tmp/reads_<tag>_v2.pkl; this script
# stages that name from the produced occaware reads so the no-GT segmentation
# consumes the same read stream.
# ============================================================================
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: $0 <tag> --video <path> [extra geo_read flags...]" >&2
  exit 2
fi
tag="$1"; shift
VIDEO=""
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
  if [ "${args[$i]}" = "--video" ]; then
    next=$((i + 1))
    [ "$next" -lt "${#args[@]}" ] || {
      echo "run_research_reads: --video requires a path" >&2
      exit 2
    }
    VIDEO="${args[$next]}"
  fi
done
[ -n "$VIDEO" ] || {
  echo "run_research_reads: --video is required" >&2
  exit 2
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GR="$REPO_ROOT/scripts/geo_read.py"
GME="$REPO_ROOT/scripts/gen_motion_events.py"
EAF="$REPO_ROOT/scripts/extract_alignfeat.py"
DECODE="$REPO_ROOT/scripts/run_research_decode.sh"
NVDEC_CHECK="$REPO_ROOT/scripts/check_nvdec.py"

for f in "$GR" "$GME" "$EAF" "$DECODE" "$NVDEC_CHECK"; do
  [ -f "$f" ] || { echo "run_research_reads: missing component: $f" >&2; exit 1; }
done

# Import environment: repo root (so detect/core/analysis resolve) + scripts/ (so
# the bare geo_read/calib_util/cell_common/occ_mask imports resolve).
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"

# Six-color centroids for the capture's cube. Supply your capture's own
# calibration via CUBED_CENTROIDS.
CENTROIDS="${CUBED_CENTROIDS:-calibration_gan12.json}"

# Alignment classifier fed to extract_alignfeat via CUBED_ALIGNED_MODEL. Bring
# your own aligned ONNX; override with CUBED_ALIGNED_MODEL.
ALIGNED_MODEL="${CUBED_ALIGNED_MODEL:-weights/cube_aligned_userlabeled.onnx}"

# Per-tag staging paths — identical to the names run_research_decode.sh resolves.
RD="/tmp/reads_${tag}_occaware.pkl"
RD_V2="/tmp/reads_${tag}_v2.pkl"        # gen_motion_events hardcoded input name
EV="/tmp/motion_events_${tag}.json"
AF="/tmp/alignfeat_${tag}_new.npz"

# geo_read invocation. Extra "$@" (e.g. --video <path>, --motion-gate,
# --align-thresh <v>) pass through to geo_read unchanged.
GEO_CMD=(python3 -u "$GR" --tag "$tag" "$@" \
  --centroids-json "$CENTROIDS" --gpu-reads --gpu-warp --out "$RD")
EVENTS_CMD=(python3 -u "$GME" "$tag" "$EV")
ALIGN_CMD=(env "CUBED_ALIGNED_MODEL=$ALIGNED_MODEL" python3 -u "$EAF" \
  --tag "$tag" --video "$VIDEO" --out "$AF")

NVDEC_POLICY="${CUBED_NVDEC:-auto}"
case "$NVDEC_POLICY" in
  auto|off|require) ;;
  *)
    echo "run_research_reads: CUBED_NVDEC must be auto, off, or require" >&2
    exit 2
    ;;
esac

NVDEC_SELECTED="probe-on-execute"
if [ "${CUBED_RESEARCH_DECODE_EXECUTE:-0}" = "1" ]; then
  if [ "$NVDEC_POLICY" = "off" ]; then
    NVDEC_SELECTED="host (disabled)"
    unset CUBED_GPU_DECODE
  elif python3 "$NVDEC_CHECK" --video "$VIDEO"; then
    GEO_CMD+=(--gpu-decode)
    NVDEC_SELECTED="nvdec"
  elif [ "$NVDEC_POLICY" = "require" ]; then
    echo "run_research_reads: NVDEC is required but failed its exact-video probe" >&2
    exit 1
  else
    NVDEC_SELECTED="host (NVDEC unavailable for this video)"
    unset CUBED_GPU_DECODE
  fi
fi

echo "[run_research_reads] tag=$tag"
echo "[run_research_reads] PYTHONPATH=$PYTHONPATH"
echo "[run_research_reads] video-decode=$NVDEC_SELECTED policy=$NVDEC_POLICY"
echo "[run_research_reads] 1/4 reads:   ${GEO_CMD[*]}"
echo "[run_research_reads] 2/4 events:  ${EVENTS_CMD[*]}   (reads $RD_V2)"
echo "[run_research_reads] 3/4 align:   ${ALIGN_CMD[*]}"
echo "[run_research_reads] 4/4 decode:  $DECODE $tag"

if [ "${CUBED_RESEARCH_DECODE_EXECUTE:-0}" != "1" ]; then
  echo "[run_research_reads] DRY RUN (set CUBED_RESEARCH_DECODE_EXECUTE=1 to execute on a CUDA box with release assets and a resolvable video)." >&2
  exit 0
fi

"${GEO_CMD[@]}"
cp -f "$RD" "$RD_V2"            # feed the no-GT segmentation the same read stream
"${EVENTS_CMD[@]}"
"${ALIGN_CMD[@]}"
exec "$DECODE" "$tag"

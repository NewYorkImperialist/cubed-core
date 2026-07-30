#!/bin/bash
# Readiness probe for a remote Decode GPU host.
#
# This never changes durable box state and never runs Decode. The NVDEC check
# stages and removes one temporary probe clip. The script only answers "is the
# box ready for scripts/remote_decode_runner.sh". Each
# named check prints one line as it finishes and the script always runs every
# check to completion, so one failure does not hide the rest of the picture.
#
# Environment (same host contract as scripts/remote_decode_runner.sh):
#   CUBED_REMOTE_SSH_DEST       e.g. root@203.0.113.7            (required)
#   CUBED_REMOTE_SSH_PORT       sshd port on the remote          (default 22)
#   CUBED_REMOTE_ROOT           remote cubed-core checkout       (default /workspace/cubed-core)
#   CUBED_REMOTE_DECODE_VENV    venv holding the Decode GPU extras, relative to
#                               CUBED_REMOTE_ROOT                (default .venv-decode-gpu)
#   CUBED_REMOTE_NVDEC          off, auto, or require            (default auto)
#
# Exit status is nonzero if any check reports FAIL. WARN and SKIP never affect
# the exit status.
set -uo pipefail

DEST=${CUBED_REMOTE_SSH_DEST:-}
PORT=${CUBED_REMOTE_SSH_PORT:-22}
REMOTE_ROOT=${CUBED_REMOTE_ROOT:-/workspace/cubed-core}
REMOTE_VENV=${CUBED_REMOTE_DECODE_VENV:-.venv-decode-gpu}
NVDEC_POLICY=${CUBED_REMOTE_NVDEC:-auto}
REMOTE_PYTHON="$REMOTE_ROOT/$REMOTE_VENV/bin/python"
REMOTE_MODEL_MANIFEST="$REMOTE_ROOT/workspace/release-assets/camera-tracker-v1-runtime/manifest.json"
# A short ConnectTimeout keeps this doctor fast (seconds, not minutes) even
# when the host is unreachable; the real Decode runner has no
# timeout because a real job is expected to take much longer.
SSH_OPTS=(-p "$PORT" -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10)

FAILURES=0

case "$NVDEC_POLICY" in
  off|auto|require) ;;
  *)
    printf '[remote-doctor] config.nvdec FAIL CUBED_REMOTE_NVDEC must be off, auto, or require; got %s\n' \
      "$NVDEC_POLICY" >&2
    exit 2
    ;;
esac

report() {
  local id="$1" status="$2" detail="$3"
  detail=${detail//$'\n'/ }
  printf '[remote-doctor] %s %s %s\n' "$id" "$status" "$detail"
  if [ "$status" = "FAIL" ]; then
    FAILURES=$((FAILURES + 1))
  fi
}

report_nvdec_problem() {
  local detail="$1"
  if [ "$NVDEC_POLICY" = "require" ]; then
    report box.nvdec FAIL "$detail"
  else
    report box.nvdec WARN "$detail"
  fi
}

ssh_capture() {
  # Runs one remote command with BatchMode ssh, capturing stdout only via a
  # global OUT variable and returning the remote exit status. stderr is
  # dropped because provider SSH banners (for example Vast's welcome text)
  # arrive there and would pollute single-value captures.
  OUT=$(ssh "${SSH_OPTS[@]}" "$DEST" "$@" 2>/dev/null)
}

if [ -z "$DEST" ]; then
  report ssh.reachable FAIL "CUBED_REMOTE_SSH_DEST is required"
  SSH_OK=0
elif ssh_capture true; then
  report ssh.reachable PASS "connected to $DEST"
  SSH_OK=1
else
  report ssh.reachable FAIL "${OUT:-ssh connection failed}"
  SSH_OK=0
fi

if [ "$SSH_OK" = 1 ] && ssh_capture "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"; then
  report box.nvidia-smi PASS "$OUT"
elif [ "$SSH_OK" = 1 ]; then
  report box.nvidia-smi FAIL "${OUT:-nvidia-smi failed or is not installed}"
else
  report box.nvidia-smi SKIP "ssh.reachable failed"
fi

# box.venv gates box.nvdec (the probe needs a Python to run check_nvdec.py
# with), so resolve venv presence once and reuse it for both checks; box.venv
# is still reported on its own line later, in the order the caller asked for.
VENV_OK=0
if [ "$SSH_OK" = 1 ] && ssh_capture "test -x '$REMOTE_PYTHON'"; then
  VENV_OK=1
fi

if [ "$NVDEC_POLICY" = "off" ]; then
  report box.nvdec SKIP "disabled by CUBED_REMOTE_NVDEC=off"
elif [ "$SSH_OK" != 1 ]; then
  report box.nvdec SKIP "ssh.reachable failed"
elif [ "$VENV_OK" != 1 ]; then
  report box.nvdec SKIP "no Decode venv python at $REMOTE_PYTHON"
elif ! command -v ffmpeg >/dev/null 2>&1; then
  report_nvdec_problem "local ffmpeg is required to build the NVDEC probe clip"
else
  PROBE_DIR=$(mktemp -d)
  PROBE_LOCAL="$PROBE_DIR/nvdec-probe.mp4"
  if ! ffmpeg -hide_banner -loglevel error -f lavfi -i 'color=c=gray:size=64x64:rate=30' \
      -frames:v 2 -an -c:v libx264 -profile:v baseline -pix_fmt yuv420p -y "$PROBE_LOCAL" \
      >/dev/null 2>&1; then
    report_nvdec_problem "local ffmpeg could not encode the NVDEC probe clip"
  else
    REMOTE_SCRATCH="$REMOTE_ROOT/workspace/remote-doctor-nvdec-probe-$$"
    if ! ssh_capture "mkdir -p '$REMOTE_SCRATCH'"; then
      report_nvdec_problem "could not create a remote scratch directory: ${OUT:-unknown error}"
    elif ! scp -P "$PORT" -o BatchMode=yes -q "$PROBE_LOCAL" "$DEST:$REMOTE_SCRATCH/probe.mp4" 2>/dev/null; then
      report_nvdec_problem "could not upload the NVDEC probe clip"
      ssh "${SSH_OPTS[@]}" "$DEST" "rm -rf '$REMOTE_SCRATCH'" >/dev/null 2>&1 || true
    else
      if ssh_capture "cd '$REMOTE_ROOT' && PYTHONPATH='$REMOTE_ROOT:$REMOTE_ROOT/scripts' '$REMOTE_PYTHON' scripts/check_nvdec.py --video '$REMOTE_SCRATCH/probe.mp4'"; then
        report box.nvdec PASS "$OUT"
      else
        report_nvdec_problem "${OUT:-NVDEC probe failed}; hint: fractional Vast rentals (gpu_frac < 1.0) have no NVDEC"
      fi
      ssh "${SSH_OPTS[@]}" "$DEST" "rm -rf '$REMOTE_SCRATCH'" >/dev/null 2>&1 || true
    fi
  fi
  rm -rf "$PROBE_DIR"
fi

if [ "$SSH_OK" != 1 ]; then
  report box.venv SKIP "ssh.reachable failed"
elif [ "$VENV_OK" = 1 ]; then
  report box.venv PASS "$REMOTE_PYTHON exists"
else
  report box.venv FAIL "$REMOTE_PYTHON is missing"
fi

if [ "$SSH_OK" != 1 ]; then
  report box.onnx-cuda SKIP "ssh.reachable failed"
elif [ "$VENV_OK" != 1 ]; then
  report box.onnx-cuda SKIP "no Decode venv python at $REMOTE_PYTHON"
elif ssh_capture "cd '$REMOTE_ROOT' && CUBED_CORE_TRACKER_MODEL_MANIFEST='$REMOTE_MODEL_MANIFEST' make doctor-decode-gpu >/dev/null && echo active"; then
  report box.onnx-cuda PASS "alignment model inference used an active CUDA session"
else
  report box.onnx-cuda FAIL "CUDA model-session doctor failed"
fi

if [ "$SSH_OK" != 1 ]; then
  report repo.commit-match SKIP "ssh.reachable failed"
else
  LOCAL_HEAD=$(git rev-parse HEAD 2>/dev/null || true)
  # rsync-preserved ownership makes root's git refuse the repo without an
  # explicit safe.directory override (git "dubious ownership" protection).
  if ssh_capture "git -c safe.directory='$REMOTE_ROOT' -C '$REMOTE_ROOT' rev-parse HEAD"; then
    REMOTE_HEAD="$OUT"
  else
    REMOTE_HEAD=""
  fi
  if [ -z "$LOCAL_HEAD" ] || [ -z "$REMOTE_HEAD" ]; then
    report repo.commit-match WARN "could not read one or both commit hashes (local='$LOCAL_HEAD' remote='$REMOTE_HEAD')"
  elif [ "$LOCAL_HEAD" = "$REMOTE_HEAD" ]; then
    report repo.commit-match PASS "local and remote are both at $LOCAL_HEAD"
  else
    report repo.commit-match WARN "local $LOCAL_HEAD differs from remote $REMOTE_HEAD"
  fi
fi

if [ "$FAILURES" -gt 0 ]; then
  echo "[remote-doctor] $FAILURES check(s) failed" >&2
  exit 1
fi
echo "[remote-doctor] all checks passed"

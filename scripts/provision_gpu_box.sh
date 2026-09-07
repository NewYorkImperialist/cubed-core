#!/usr/bin/env bash
#
# provision_gpu_box.sh - bootstrap a remote GPU box for the per-job Decode
# bridge (scripts/remote_decode_runner.sh). Today that setup is prose in
# docs/CLOUD_GPU.md; this script automates it.
#
# It has two modes:
#
#   Workbench-host mode (default, run this on your Mac/workbench):
#     1. verifies non-interactive SSH to the box
#     2. verifies the local runtime and decode-support assets
#     3. rsyncs this checkout to the box
#     4. re-invokes itself on the box with --on-box
#     5. copies the resulting receipt back
#     6. prints the environment block for the workbench host, and optionally
#        appends it to .env.hub
#
#   --on-box mode (runs ON the remote box, normally only via step 4 above):
#     installs system packages and uv, runs `make bootstrap-compute-gpu`, then
#     verifies GPU/NVDEC/CUDA-provider/Decode readiness and writes
#     workspace/provision-receipt.json.
#
# Usage:
#   ./scripts/provision_gpu_box.sh [--dest user@host] [--port N] [--append-env]
#   ./scripts/provision_gpu_box.sh --on-box            (invoked over ssh, not
#                                                        normally run by hand)
#
# Environment (workbench-host mode):
#   CUBED_REMOTE_SSH_DEST   e.g. root@203.0.113.7            (required unless --dest)
#   CUBED_REMOTE_SSH_PORT   sshd port on the remote           (default 22, or --port)
#   CUBED_REMOTE_ROOT       remote cubed-core checkout        (default /workspace/cubed-core)
#
# It is idempotent: re-running it re-syncs the checkout, re-runs the same
# on-box steps (each of which is itself a no-op when already satisfied), and
# refreshes the same .env.hub block instead of duplicating it. It never uses
# sudo (Vast and similar GPU rentals run the SSH session as root already) and
# never inlines pkill/pgrep -f.
set -euo pipefail

UV_VERSION="0.11.16"
PY_MIN_MINOR="10"
PY_MAX_MINOR="12"
APT_PACKAGES=(ca-certificates curl ffmpeg git make python3 python3-venv tmux xz-utils)
DEMO_VIDEO_NAME="gtD1s_decoder-demo-clip_f3919-f9722.mp4"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---------------------------------------------------------------------------
# Output helpers (same shape as setup.sh)
# ---------------------------------------------------------------------------
info()  { printf '==> %s\n' "$1"; }
warn()  { printf 'WARNING: %s\n' "$1" >&2; }
fail()  { printf 'ERROR: %s\n' "$1" >&2; exit 1; }

on_error() {
  status=$?
  printf '\n' >&2
  printf 'provision_gpu_box.sh stopped (exit %s). The message above says what failed.\n' "$status" >&2
  printf 'Fix that item, then re-run - it is safe to run again.\n' >&2
  exit "$status"
}
trap on_error ERR

usage() {
  sed -n '3,37p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
MODE="host"
DEST="${CUBED_REMOTE_SSH_DEST:-}"
PORT="${CUBED_REMOTE_SSH_PORT:-22}"
REMOTE_ROOT="${CUBED_REMOTE_ROOT:-/workspace/cubed-core}"
APPEND_ENV=0

while [ $# -gt 0 ]; do
  case "$1" in
    --dest) DEST=$2; shift 2 ;;
    --port) PORT=$2; shift 2 ;;
    --append-env) APPEND_ENV=1; shift ;;
    --on-box) MODE="on-box"; shift ;;
    -h|--help) usage ;;
    *) fail "unknown argument: $1 (try --help)" ;;
  esac
done

# ===========================================================================
# --on-box mode
# ===========================================================================
run_on_box() {
  cd "$REPO_ROOT"

  CHECKS_TSV="$(mktemp)"
  cleanup_checks() { rm -f "$CHECKS_TSV"; }
  trap cleanup_checks EXIT

  # record <id> <required:0|1> <ok:0|1> <detail...>
  # The receipt is one check per TSV line, so a detail string (e.g. captured
  # command output) must not contain a literal tab or newline.
  record() {
    local id="$1" required="$2" ok="$3"
    shift 3
    local detail="$*"
    detail="${detail//$'\n'/ }"
    detail="${detail//$'\t'/ }"
    printf '%s\t%s\t%s\t%s\n' "$id" "$required" "$ok" "$detail" >>"$CHECKS_TSV"
    if [ "$ok" = "1" ]; then
      info "check $id: ok ($detail)"
    else
      warn "check $id: FAILED ($detail)"
    fi
  }

  # --- (a) system packages, only if missing ---------------------------------
  info "Phase A: system packages"
  missing_packages=()
  for pkg in "${APT_PACKAGES[@]}"; do
    if ! dpkg -s "$pkg" >/dev/null 2>&1; then
      missing_packages+=("$pkg")
    fi
  done
  if [ "${#missing_packages[@]}" -gt 0 ]; then
    info "installing: ${missing_packages[*]}"
    apt-get update
    apt-get install -y "${missing_packages[@]}"
  else
    info "all required packages already present"
  fi

  # --- (b) pinned uv, the way setup.sh installs it ---------------------------
  info "Phase B: uv $UV_VERSION"
  uv_current_version() {
    command -v uv >/dev/null 2>&1 || return 1
    uv --version 2>/dev/null | awk '{print $2}'
  }
  if [ "$(uv_current_version || true)" != "$UV_VERSION" ]; then
    curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh
    if [ -f "$HOME/.local/bin/env" ]; then
      # shellcheck disable=SC1091
      . "$HOME/.local/bin/env"
    fi
    case ":$PATH:" in
      *":$HOME/.local/bin:"*) : ;;
      *) PATH="$HOME/.local/bin:$PATH"; export PATH ;;
    esac
    [ "$(uv_current_version || true)" = "$UV_VERSION" ] \
      || fail "uv $UV_VERSION not on PATH after install"
  else
    info "uv $UV_VERSION already present"
  fi

  # Find a supported interpreter (mirrors setup.sh's find_supported_python:
  # newer distros may ship a python3 outside 3.10-3.12).
  python_ok() {
    command -v "$1" >/dev/null 2>&1 || return 1
    "$1" - "$PY_MIN_MINOR" "$PY_MAX_MINOR" <<'PY'
import sys
lo, hi = int(sys.argv[1]), int(sys.argv[2])
major, minor = sys.version_info[:2]
sys.exit(0 if major == 3 and lo <= minor <= hi else 1)
PY
  }
  BOOTSTRAP_PYTHON=""
  for candidate in python3 python3.12 python3.11 python3.10; do
    if python_ok "$candidate"; then
      BOOTSTRAP_PYTHON="$candidate"
      break
    fi
  done
  [ -n "$BOOTSTRAP_PYTHON" ] || fail "no Python 3.${PY_MIN_MINOR}-3.${PY_MAX_MINOR} interpreter found"

  # --- (c) bootstrap-compute-gpu --------------------------------------------
  info "Phase C: make bootstrap-compute-gpu (BOOTSTRAP_PYTHON=$BOOTSTRAP_PYTHON)"
  make bootstrap-compute-gpu BOOTSTRAP_PYTHON="$BOOTSTRAP_PYTHON"

  MANIFEST="$REPO_ROOT/workspace/release-assets/camera-tracker-v1-runtime/manifest.json"

  # --- (d) verification chain ------------------------------------------------
  info "Phase D: verification"

  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    record nvidia-smi 1 1 "GPU visible"
  else
    record nvidia-smi 1 0 "nvidia-smi missing or failed"
  fi

  # NVDEC is advisory: fractional Vast rentals (gpu_frac < 1.0) have no NVDEC,
  # and the decode pipeline already falls back safely (CUBED_NVDEC=auto), so a
  # failure here is recorded but does not fail provisioning.
  # check_nvdec.py's child process does `from gpu_decode import ...`, a bare
  # import that needs scripts/ on PYTHONPATH, and needs the Decode GPU venv
  # for onnxruntime/torch/PyNvVideoCodec - mirrors how
  # native_decode_runner.py invokes the same script.
  demo_video="$REPO_ROOT/workspace/release-assets/$DEMO_VIDEO_NAME"
  if [ ! -f "$demo_video" ]; then
    record nvdec 0 0 "skipped: no demo video at $demo_video (run make download-demo)"
  elif PYTHONPATH="$REPO_ROOT:$REPO_ROOT/scripts" .venv-decode-gpu/bin/python \
      scripts/check_nvdec.py --video "$demo_video" >/dev/null 2>&1; then
    record nvdec 0 1 "NVDEC decoded the demo clip"
  else
    warn "Vast note: fractional GPU rentals (gpu_frac < 1.0) have no NVDEC; rent a whole GPU (gpu_frac 1.0)"
    record nvdec 0 0 "NVDEC probe failed"
  fi

  if CUBED_CORE_TRACKER_MODEL_MANIFEST="$MANIFEST" make doctor-decode-gpu >/dev/null 2>&1; then
    record doctor-decode-gpu 1 1 "alignment model inference used an active CUDA session"
  else
    record doctor-decode-gpu 1 0 "CUDA model-session doctor failed"
  fi

  # --- (e) receipt -------------------------------------------------------
  info "Phase E: writing workspace/provision-receipt.json"
  mkdir -p "$REPO_ROOT/workspace"
  # rsync preserves the workbench checkout's numeric owner. On a rented box
  # the SSH user is commonly root, so Git otherwise rejects the synced
  # checkout as a "dubious ownership" repository even though this script just
  # created it. Scope the trust exception to these read-only receipt probes.
  GIT_SAFE=(-c "safe.directory=$REPO_ROOT")
  COMMIT="$(git "${GIT_SAFE[@]}" rev-parse HEAD 2>/dev/null || echo unknown)"
  DIRTY=0
  [ -n "$(git "${GIT_SAFE[@]}" status --porcelain 2>/dev/null || true)" ] && DIRTY=1
  TIMESTAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  export CUBED_PROV_OS_RELEASE
  CUBED_PROV_OS_RELEASE="$(cat /etc/os-release 2>/dev/null || true)"
  export CUBED_PROV_NVIDIA_DRIVER
  CUBED_PROV_NVIDIA_DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 || true)"
  export CUBED_PROV_NVIDIA_GPU_NAME
  CUBED_PROV_NVIDIA_GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || true)"
  export CUBED_PROV_CUDA_VERSION
  CUBED_PROV_CUDA_VERSION="$(nvidia-smi 2>/dev/null | grep -o 'CUDA Version: [0-9.]*' | head -n1 | awk '{print $3}' || true)"

  receipt_status=0
  "$BOOTSTRAP_PYTHON" - "$REPO_ROOT/workspace/provision-receipt.json" "$COMMIT" "$DIRTY" "$TIMESTAMP" "$CHECKS_TSV" <<'PY' || receipt_status=$?
import json
import os
import sys

receipt_path, commit, dirty, timestamp, checks_tsv = sys.argv[1:6]

checks = []
required_failed = False
with open(checks_tsv, encoding="utf-8") as handle:
    for line in handle:
        line = line.rstrip("\n")
        if not line:
            continue
        check_id, required, ok, detail = line.split("\t", 3)
        required = required == "1"
        ok = ok == "1"
        checks.append({"id": check_id, "required": required, "ok": ok, "detail": detail})
        if required and not ok:
            required_failed = True

receipt = {
    "schema": "cubed-core/provision-receipt-v1",
    "commit": commit,
    "dirty": dirty == "1",
    "timestamp": timestamp,
    "image": {
        "os_release": os.environ.get("CUBED_PROV_OS_RELEASE", ""),
        "nvidia_driver_version": os.environ.get("CUBED_PROV_NVIDIA_DRIVER", ""),
        "nvidia_gpu_name": os.environ.get("CUBED_PROV_NVIDIA_GPU_NAME", ""),
        "cuda_version": os.environ.get("CUBED_PROV_CUDA_VERSION", ""),
    },
    "checks": checks,
    "ok": not required_failed,
}
with open(receipt_path, "w", encoding="utf-8") as handle:
    json.dump(receipt, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(receipt, indent=2, sort_keys=True))
sys.exit(0 if not required_failed else 1)
PY

  if [ "$receipt_status" != "0" ]; then
    fail "one or more required checks failed; see workspace/provision-receipt.json"
  fi
  info "all required checks passed"
}

if [ "$MODE" = "on-box" ]; then
  run_on_box
  exit 0
fi

# ===========================================================================
# Workbench-host mode
# ===========================================================================
[ -n "$DEST" ] || fail "CUBED_REMOTE_SSH_DEST is required (or pass --dest user@host)"

SSH_OPTS=(-p "$PORT" -o BatchMode=yes -o StrictHostKeyChecking=accept-new)

# --- (a) non-interactive SSH -------------------------------------------------
info "Phase 1: verifying non-interactive SSH to $DEST (port $PORT)"
if ! ssh "${SSH_OPTS[@]}" "$DEST" true 2>/dev/null; then
  cat >&2 <<REMEDIATION

Non-interactive SSH to $DEST (port $PORT) failed.

Set up a dedicated key and register it with the provider, then retry:
  ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""
  vastai create ssh-key ~/.ssh/id_ed25519.pub
Vast injects registered keys into a new instance's authorized_keys at boot.
An already-running instance needs the key attached explicitly instead:
  vastai attach ssh <instance-id> ~/.ssh/id_ed25519.pub
Then verify by hand:
  ssh -p $PORT -o BatchMode=yes $DEST true
REMEDIATION
  fail "cannot continue without non-interactive SSH to the box"
fi
info "SSH OK"

# --- (b) local release assets ------------------------------------------------
info "Phase 2: verifying local release assets"
LOCAL_ASSET_DIR="$REPO_ROOT/workspace/release-assets"
LOCAL_MANIFEST="$LOCAL_ASSET_DIR/camera-tracker-v1-runtime/manifest.json"
LOCAL_TRUST_MODEL="$LOCAL_ASSET_DIR/trust_v1_numpy.npz"
LOCAL_CALIBRATION="$LOCAL_ASSET_DIR/calibration_gan12.json"
for asset_path in "$LOCAL_MANIFEST" "$LOCAL_TRUST_MODEL" "$LOCAL_CALIBRATION"; do
  if [ ! -f "$asset_path" ]; then
    fail "required release asset not found at $asset_path; run 'make download-assets' first (see docs/CLOUD_GPU.md)"
  fi
done
info "runtime and decode-support assets OK"

# --- (c) rsync the checkout ---------------------------------------------------
info "Phase 3: syncing checkout to $DEST:$REMOTE_ROOT"
RSYNC_SSH=(-e "ssh -p $PORT -o BatchMode=yes -o StrictHostKeyChecking=accept-new")
ssh "${SSH_OPTS[@]}" "$DEST" "mkdir -p '$REMOTE_ROOT'"

# .git IS synced (not excluded): the on-box side reads its own `git rev-parse
# HEAD` / `git status --porcelain` for the provision receipt's commit/dirty
# fields, so a maintainer can check commit parity between the workbench host
# and the box without a separate probe. The repository has no large tracked
# objects (models/video are release assets, excluded below via workspace/),
# so the extra transfer is small.
#
# workspace/ is synced separately below (release-assets only, without
# --delete) because it also holds live job state
# (workspace/decode-jobs/, workspace/captures/) that the Decode runner stages
# and cleans up on the box itself; --delete here
# would race a job in flight.
# Ignored data and artifact roots are also excluded explicitly. rsync does not
# honor .gitignore, and provisioning must never upload a contributor's local
# datasets, captures, model bytes, scratch output, or temporary files merely
# because they happen to live beneath the checkout.
# Local tooling, caches, ignored artifacts, and every `.env*` file except the
# tracked `.env.example` are excluded. The on-box side is driven entirely by
# explicit SSH assignments, so host secrets never need to reach the rented box.
rsync -az --delete "${RSYNC_SSH[@]}" \
  --exclude '.venv*' \
  --exclude 'node_modules' \
  --exclude 'graphify-out' \
  --exclude '/.agents/' \
  --exclude '/.claude/' \
  --exclude '/.codex/' \
  --exclude '/AGENTS.md' \
  --exclude '/CLAUDE.md' \
  --exclude '/docs/AGENT_BRIEF.md' \
  --exclude '.DS_Store' \
  --exclude '.pytest_cache/' \
  --exclude '.ruff_cache/' \
  --exclude '__pycache__/' \
  --exclude '*.py[cod]' \
  --exclude '*.egg-info/' \
  --exclude '.coverage' \
  --exclude 'htmlcov/' \
  --exclude '/build/' \
  --exclude '/data/' \
  --exclude '/datasets/' \
  --include '/models/local/.gitkeep' \
  --exclude '/models/local/*' \
  --exclude '/output/' \
  --exclude '/tmp/' \
  --exclude '/weights/' \
  --exclude '/workspace/' \
  --include '/.env.example' \
  --exclude '.env*' \
  --exclude '*.pem' \
  --exclude '*.key' \
  --exclude '*.onnx' \
  --exclude '*.pt' \
  --exclude '*.pth' \
  --exclude '*.engine' \
  --exclude '*.ckpt' \
  --exclude '*.safetensors' \
  --exclude '*.tflite' \
  --exclude '*.mlmodel' \
  --exclude '*.mlmodelc/' \
  --exclude '*.mlpackage/' \
  --exclude '*.npz' \
  --exclude '*.npy' \
  --exclude '*.pkl' \
  --exclude '*.pickle' \
  --exclude '*.joblib' \
  --exclude '*.bin' \
  --exclude '*.avi' \
  --exclude '*.m4v' \
  --exclude '*.mkv' \
  --exclude '*.mov' \
  --exclude '*.mp4' \
  --exclude '*.webm' \
  --exclude '*.zip' \
  --exclude '*.tar' \
  --exclude '*.tar.gz' \
  --exclude '*.tgz' \
  "$REPO_ROOT/" "$DEST:$REMOTE_ROOT/"

if [ -d "$REPO_ROOT/workspace/release-assets" ]; then
  # The checkout rsync above excludes workspace/, so the remote parent
  # directory does not exist yet and rsync will not create intermediate
  # directories on its own.
  ssh "${SSH_OPTS[@]}" "$DEST" "mkdir -p '$REMOTE_ROOT/workspace'"
  rsync -az "${RSYNC_SSH[@]}" \
    "$REPO_ROOT/workspace/release-assets/" "$DEST:$REMOTE_ROOT/workspace/release-assets/"
fi

# --- (d) re-invoke on the box --------------------------------------------------
info "Phase 4: running on-box provisioning"
on_box_status=0
ssh "${SSH_OPTS[@]}" "$DEST" "cd '$REMOTE_ROOT' && bash scripts/provision_gpu_box.sh --on-box" || on_box_status=$?

# --- (e) copy the receipt back regardless, so a partial failure is still visible
info "Phase 5: fetching workspace/provision-receipt.json"
mkdir -p "$REPO_ROOT/workspace"
if ! scp -P "$PORT" -o BatchMode=yes -q \
  "$DEST:$REMOTE_ROOT/workspace/provision-receipt.json" \
  "$REPO_ROOT/workspace/provision-receipt.json" 2>/dev/null; then
  warn "could not fetch workspace/provision-receipt.json from the box"
fi

if [ "$on_box_status" != "0" ]; then
  fail "on-box provisioning failed (see workspace/provision-receipt.json if it was fetched)"
fi

# --- (f) env block for the workbench host --------------------------------------
ENV_BLOCK_FILE="$(mktemp)"
cleanup_env_block() { rm -f "$ENV_BLOCK_FILE"; }
trap cleanup_env_block EXIT

cat >"$ENV_BLOCK_FILE" <<ENVBLOCK
# BEGIN cubed-core-provision (dest=$DEST)
CUBED_REMOTE_SSH_DEST=$DEST
CUBED_REMOTE_SSH_PORT=$PORT
CUBED_REMOTE_ROOT=$REMOTE_ROOT
CUBED_REMOTE_DECODE_VENV=.venv-decode-gpu
# Decode defaults to the API host; choosing this provisioned host for a job
# makes native mode select the bundled remote decode bridge automatically.
CUBED_CORE_DECODE_MODE=native
# END cubed-core-provision
ENVBLOCK

printf '\n%s\n\n' "Provisioning complete. Add this to the workbench host's environment:"
cat "$ENV_BLOCK_FILE"
printf '\n'

if [ "$APPEND_ENV" = "1" ]; then
  # .env.hub is deliberately separate from .env: .env holds the workbench
  # API server's own hand-edited settings (see .env.example), while this
  # block is machine-generated by this script and safe to refresh on every
  # re-run. Replacing only the marked block (instead of appending blindly)
  # keeps repeated runs idempotent.
  ENV_HUB="$REPO_ROOT/.env.hub"
  info "Phase 6: writing $ENV_HUB"
  python3 - "$ENV_HUB" "$ENV_BLOCK_FILE" <<'PY'
import sys

hub_path, block_path = sys.argv[1], sys.argv[2]
begin, end = "# BEGIN cubed-core-provision", "# END cubed-core-provision"

with open(block_path, encoding="utf-8") as handle:
    block = handle.read().rstrip("\n") + "\n"

try:
    with open(hub_path, encoding="utf-8") as handle:
        existing = handle.read()
except FileNotFoundError:
    existing = ""

begin_idx = existing.find(begin)
end_idx = existing.find(end)
if begin_idx != -1 and end_idx != -1 and end_idx > begin_idx:
    existing = existing[:begin_idx] + existing[end_idx + len(end) :]
existing = existing.rstrip("\n")
if existing:
    existing += "\n\n"
existing += block

with open(hub_path, "w", encoding="utf-8") as handle:
    handle.write(existing)
PY
  info "wrote $ENV_HUB (source it alongside .env before 'make dev' / 'make workbench')"
fi

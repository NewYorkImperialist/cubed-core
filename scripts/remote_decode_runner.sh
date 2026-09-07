#!/bin/bash
# Run a cubed-core decode job on a remote GPU host over SSH.
#
# Recommended workbench configuration:
#   CUBED_CORE_DECODE_MODE=native
#   CUBED_CORE_DECODE_COMMAND is unset
# A decode job with an explicit remote_host selection makes the server invoke
# this bundled bridge for that job as:
#   remote_decode_runner.sh --request <job-request.json> --output <decode-result.json>
# Jobs without remote_host continue to use the API host's native decode runner.
#
# External mode remains supported for operator-supplied integrations. To use
# this bridge as the global external command (the pre-selector contract), set:
#   CUBED_CORE_DECODE_MODE=external
#   CUBED_CORE_DECODE_COMMAND=/path/to/scripts/remote_decode_runner.sh
#
# The capture video and calibration are copied to the remote host, the remote
# native decode runner executes the local_camera_v1 pipeline on its GPU, and the
# finished decode result document is copied back to --output.
#
# Models are NOT uploaded. The remote host already holds the pose, alignment,
# and read-trust artifacts under its own workspace/release-assets, so the
# rewritten request references the remote runtime-asset paths in place.
#
# The result this produces is reconstruction evidence for one recording. This
# script makes no accuracy claim; the server replays the emitted moves itself.
#
# Environment:
#   CUBED_REMOTE_SSH_DEST     e.g. root@203.0.113.7          (required)
#   CUBED_REMOTE_SSH_PORT     sshd port on the remote        (default 22)
#   CUBED_REMOTE_ROOT         remote cubed-core checkout     (default /workspace/cubed-core)
#   CUBED_REMOTE_DECODE_VENV  venv holding the research-gpu extras, relative to
#                             CUBED_REMOTE_ROOT              (default .venv-decode-gpu)
#   CUBED_REMOTE_NVDEC        NVDEC policy for the remote run: auto|off|require
#                                                            (default auto)
set -euo pipefail

REQUEST=""
OUTPUT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --request) REQUEST=$2; shift 2 ;;
    --output) OUTPUT=$2; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [ -z "$REQUEST" ] || [ -z "$OUTPUT" ]; then
  echo "--request and --output are required" >&2
  exit 2
fi

DEST=${CUBED_REMOTE_SSH_DEST:?CUBED_REMOTE_SSH_DEST is required}
PORT=${CUBED_REMOTE_SSH_PORT:-22}
REMOTE_ROOT=${CUBED_REMOTE_ROOT:-/workspace/cubed-core}
# `make bootstrap-research-gpu` installs the research-gpu extras here.
REMOTE_VENV=${CUBED_REMOTE_DECODE_VENV:-.venv-decode-gpu}
REMOTE_NVDEC=${CUBED_REMOTE_NVDEC:-auto}
case "$REMOTE_NVDEC" in
  auto|off|require) ;;
  *) echo "CUBED_REMOTE_NVDEC must be auto, off, or require" >&2; exit 2 ;;
esac
# ConnectTimeout bounds the initial handshake and the ServerAlive* pair bounds
# a connection that goes quiet mid-session (dead box, network partition), so a
# lost remote fails in about a minute instead of wedging the multi-hour job
# timeout on a connection that will never come back.
SSH_OPTS=(
  -p "$PORT"
  -o BatchMode=yes
  -o StrictHostKeyChecking=accept-new
  -o ConnectTimeout=10
  -o ServerAliveInterval=15
  -o ServerAliveCountMax=4
)
# scp uses -P (capital) for port instead of ssh's -p, so it needs its own
# option array; SCP_OPTS still shares the same liveness options instead of
# repeating them inline at each call site.
SCP_OPTS=(
  -P "$PORT"
  -o BatchMode=yes
  -o ConnectTimeout=10
  -o ServerAliveInterval=15
  -o ServerAliveCountMax=4
  -q
)

JOB_ID=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["job_id"])' "$REQUEST")
CAPTURE_ID=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["capture_id"])' "$REQUEST")
case "$JOB_ID" in
  *[!a-f0-9]*|"") echo "refusing unsafe job id: $JOB_ID" >&2; exit 2 ;;
esac
case "$CAPTURE_ID" in
  *[!a-f0-9]*|"") echo "refusing unsafe capture id: $CAPTURE_ID" >&2; exit 2 ;;
esac
# The native decode runner requires the request and the result to be siblings
# inside its own job directory, so stage the job exactly there.
REMOTE_JOB="$REMOTE_ROOT/workspace/decode-jobs/$JOB_ID"
# Namespaced per job so concurrent Decode attempts never share a staged
# capture directory. The request inputs are rewritten below to point here.
REMOTE_CAPTURE="$REMOTE_ROOT/workspace/captures/$CAPTURE_ID-$JOB_ID"

cleanup() {
  # Kill nothing by pattern here: a self-matching `pkill -f` would match this
  # very script. The remote process ends with its ssh session.
  ssh "${SSH_OPTS[@]}" "$DEST" "rm -rf '$REMOTE_JOB' '$REMOTE_CAPTURE'" || true
  rm -f "$REQUEST.remote"
}
trap cleanup EXIT

echo "[cubed-core:stage] upload"
echo "remote-decode: staging inputs for job $JOB_ID on $DEST"
ssh "${SSH_OPTS[@]}" "$DEST" "mkdir -p '$REMOTE_JOB' '$REMOTE_CAPTURE'"

# Rewrite the request's local paths for the remote host and emit the copy plan.
# Only the capture media moves. Runtime assets are repository-relative and are
# resolved by the remote runner against its own checkout, so they are left
# untouched. Basenames only, never a directory component from the local host.
PLAN=$(python3 - "$REQUEST" "$REMOTE_CAPTURE" <<'PY'
import json
import os
import sys

request_path, remote_capture = sys.argv[1], sys.argv[2]
with open(request_path, encoding="utf-8") as stream:
    request = json.load(stream)
plan = []
for key in ("video", "calibration"):
    local = request["inputs"][key]
    remote = f"{remote_capture}/{os.path.basename(local)}"
    plan.append((local, remote))
    request["inputs"][key] = remote
with open(request_path + ".remote", "w", encoding="utf-8") as stream:
    json.dump(request, stream, indent=2, sort_keys=True)
    stream.write("\n")
for local, remote in plan:
    print(f"{local}\t{remote}")
PY
)

while IFS=$'\t' read -r local remote; do
  [ -n "$local" ] || continue
  echo "remote-decode: uploading $(basename "$local")"
  scp "${SCP_OPTS[@]}" "$local" "$DEST:$remote"
done <<<"$PLAN"
scp "${SCP_OPTS[@]}" "$REQUEST.remote" "$DEST:$REMOTE_JOB/job-request.json"

# The remote native runner (src/cubed_core/native_decode_runner.py) prints its
# own flushed [cubed-core:stage] markers (reads, events, alignfeat, decode) on
# stdout, and ssh streams that straight through below, so the server sees the
# fine-grained pipeline stages. Do not pipe this ssh command through anything
# that buffers.
echo "remote-decode: running the decode pipeline on the remote GPU"
ssh "${SSH_OPTS[@]}" "$DEST" "cd '$REMOTE_ROOT' && \
  CUBED_CORE_REPO_ROOT='$REMOTE_ROOT' \
  CUBED_NVDEC='$REMOTE_NVDEC' \
  '$REMOTE_VENV/bin/python' -u -m cubed_core.native_decode_runner \
    --request '$REMOTE_JOB/job-request.json' \
    --output '$REMOTE_JOB/decode-result.json'"

echo "[cubed-core:stage] download"
echo "remote-decode: downloading the decode result"
scp "${SCP_OPTS[@]}" "$DEST:$REMOTE_JOB/decode-result.json" "$OUTPUT"
if [ ! -s "$OUTPUT" ]; then
  echo "remote-decode: downloaded decode result is missing or empty" >&2
  exit 1
fi
echo "[cubed-core:stage] validate"
python3 - "$OUTPUT" "$CAPTURE_ID" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    result = json.load(stream)
if not isinstance(result, dict):
    sys.exit("remote-decode: decode result is not a JSON object")
if result.get("schema") != "cubed-core/decode-result" or result.get("schema_version") != 1:
    sys.exit("remote-decode: decode result does not declare cubed-core/decode-result version 1")
if result.get("recording_id") != sys.argv[2]:
    sys.exit("remote-decode: decode result identity does not match the requested capture")
PY
echo "remote-decode: job $JOB_ID complete"

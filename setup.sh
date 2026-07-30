#!/usr/bin/env bash
#
# setup.sh - one-command POSIX setup for the Cubed Core workbench.
#
# It checks system prerequisites, ensures uv 0.11.16 is available, and then
# drives the maintained Make targets (bootstrap, web-build, and optionally the
# release assets). Environment bootstrap is safe to re-run. Release asset
# downloads intentionally do not overwrite existing paths, so rerun without
# --with-assets when those assets are already present.
#
# Usage:
#   ./setup.sh                      # preflight + bootstrap + web build
#   ./setup.sh --install-uv         # also install the pinned uv if missing
#   ./setup.sh --with-assets        # also download runtime/demo/decode assets
#   ./setup.sh --with-assets URL    # download assets from a local/base URL
#   ./setup.sh --help
#
# Environment toggles:
#   CUBED_SETUP_INSTALL_UV=1        # same as passing --install-uv
#
# The script never uses sudo and never runs video processing.

set -euo pipefail

UV_VERSION="0.11.16"
NODE_MAJOR="22"
NODE_MIN_MINOR="3"
PY_MIN_MINOR="10"
PY_MAX_MINOR="12"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

INSTALL_UV="${CUBED_SETUP_INSTALL_UV:-0}"
WITH_ASSETS=0
ASSETS_BASE_URL=""

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
info()  { printf '==> %s\n' "$1"; }
warn()  { printf 'WARNING: %s\n' "$1" >&2; }
fail()  { printf 'ERROR: %s\n' "$1" >&2; exit 1; }

on_error() {
  status=$?
  printf '\n' >&2
  printf 'setup.sh stopped (exit %s). The message above says what failed.\n' "$status" >&2
  printf 'Fix that item, then re-run ./setup.sh. If assets were already downloaded, omit --with-assets.\n' >&2
  exit "$status"
}
trap on_error ERR

usage() {
  sed -n '3,19p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

# ---------------------------------------------------------------------------
# Platform-aware install hints
# ---------------------------------------------------------------------------
PLATFORM="other"
case "$(uname -s)" in
  Darwin) PLATFORM="macos" ;;
  Linux)  PLATFORM="linux" ;;
esac

hint() {
  # hint <tool>  -> prints one platform-aware install line for that tool
  case "$1:$PLATFORM" in
    python3:macos)  echo "brew install python@3.12" ;;
    python3:linux)  echo "sudo apt-get install -y python3.12 python3.12-venv   # or 3.10/3.11; on Ubuntu add the deadsnakes PPA if apt lacks it" ;;
    node:macos|npm:macos) echo 'brew install node@22; export PATH="$(brew --prefix node@22)/bin:$PATH"' ;;
    node:linux|npm:linux) echo "install Node 22 from https://nodejs.org/en/download (distro package may be too old)" ;;
    ffmpeg:macos|ffprobe:macos) echo "brew install ffmpeg" ;;
    ffmpeg:linux|ffprobe:linux) echo "sudo apt-get install -y ffmpeg" ;;
    git:macos|make:macos) echo "xcode-select --install   # provides git and make" ;;
    git:linux)      echo "sudo apt-get install -y git" ;;
    make:linux)     echo "sudo apt-get install -y make" ;;
    curl:macos)     echo "curl ships with macOS; restore the system command-line tools" ;;
    curl:linux)     echo "sudo apt-get install -y curl" ;;
    *)              echo "install $1 (see README.md)" ;;
  esac
}

# ---------------------------------------------------------------------------
# Version checks (return 0 when acceptable)
# ---------------------------------------------------------------------------
python_ok() {
  # python_ok <command>  -> 0 when that interpreter is inside 3.MIN-3.MAX
  command -v "$1" >/dev/null 2>&1 || return 1
  "$1" - "$PY_MIN_MINOR" "$PY_MAX_MINOR" <<'PY'
import sys
lo, hi = int(sys.argv[1]), int(sys.argv[2])
major, minor = sys.version_info[:2]
sys.exit(0 if major == 3 and lo <= minor <= hi else 1)
PY
}

find_supported_python() {
  # Prefer the default python3; fall back to versioned binaries so a distro
  # whose python3 is newer than 3.12 (e.g. 3.13/3.14) still works when a
  # supported interpreter is installed alongside it.
  for candidate in python3 python3.12 python3.11 python3.10; do
    if python_ok "$candidate"; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

node_ok() {
  command -v node >/dev/null 2>&1 || return 1
  v="$(node --version 2>/dev/null)"        # e.g. v22.23.1
  v="${v#v}"
  major="${v%%.*}"
  rest="${v#*.}"
  minor="${rest%%.*}"
  case "$major" in ''|*[!0-9]*) return 1 ;; esac
  case "$minor" in ''|*[!0-9]*) return 1 ;; esac
  [ "$major" -eq "$NODE_MAJOR" ] && [ "$minor" -ge "$NODE_MIN_MINOR" ]
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [ $# -gt 0 ]; do
  case "$1" in
    --install-uv) INSTALL_UV=1 ;;
    --with-assets)
      WITH_ASSETS=1
      if [ $# -gt 1 ]; then
        case "$2" in
          -*) : ;;                       # next token is another flag
          *)  ASSETS_BASE_URL="$2"; shift ;;
        esac
      fi
      ;;
    -h|--help) usage ;;
    *) fail "unknown argument: $1 (try ./setup.sh --help)" ;;
  esac
  shift
done

cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Phase 0: preflight
# ---------------------------------------------------------------------------
info "Phase 0: checking system prerequisites"
missing=""

PYTHON_BIN=""
if PYTHON_BIN="$(find_supported_python)"; then
  info "$PYTHON_BIN $("$PYTHON_BIN" --version 2>&1 | awk '{print $2}') OK"
else
  if command -v python3 >/dev/null 2>&1; then
    warn "python3 $(python3 --version 2>&1 | awk '{print $2}') is outside 3.${PY_MIN_MINOR}-3.${PY_MAX_MINOR}, and no python3.${PY_MIN_MINOR}-python3.${PY_MAX_MINOR} binary was found on PATH"
    warn "newer distros ship Python 3.13+; install a supported interpreter alongside it (it does not need to be the default python3)"
  else
    warn "python3 not found"
  fi
  missing="$missing python3"
fi

if node_ok; then
  info "node $(node --version) OK"
else
  if command -v node >/dev/null 2>&1; then
    warn "node $(node --version) is not on the ${NODE_MAJOR}.${NODE_MIN_MINOR}+ / <${NODE_MAJOR}+1 line"
  else
    warn "node not found (needed to build the web frontend)"
  fi
  missing="$missing node"
fi

if command -v npm >/dev/null 2>&1; then
  info "npm $(npm --version) OK"
else
  warn "npm not found"
  missing="$missing npm"
fi

for tool in ffmpeg ffprobe git make; do
  if command -v "$tool" >/dev/null 2>&1; then
    info "$tool OK"
  else
    warn "$tool not found"
    missing="$missing $tool"
  fi
done

if [ "$INSTALL_UV" = "1" ]; then
  current_uv_version="$(uv --version 2>/dev/null | awk '{print $2}' || true)"
  if [ "$current_uv_version" != "$UV_VERSION" ]; then
    if command -v curl >/dev/null 2>&1; then
      info "curl OK"
    else
      warn "curl not found (needed by --install-uv)"
      missing="$missing curl"
    fi
  fi
fi

if [ -n "$missing" ]; then
  printf '\n' >&2
  printf 'Missing or unsupported prerequisites:%s\n' "$missing" >&2
  printf 'Install them, then re-run ./setup.sh:\n' >&2
  printed=""
  for tool in $missing; do
    h="$(hint "$tool")"
    case "$printed" in
      *"|$h|"*) : ;;                     # de-duplicate identical hints
      *) printf '  - %s\n' "$h" >&2; printed="$printed|$h|" ;;
    esac
  done
  exit 1
fi

# ---------------------------------------------------------------------------
# Phase 1: uv 0.11.16
# ---------------------------------------------------------------------------
info "Phase 1: ensuring uv $UV_VERSION"

uv_installer_line="curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | sh"

uv_current_version() {
  command -v uv >/dev/null 2>&1 || return 1
  uv --version 2>/dev/null | awk '{print $2}'
}

uv_present_version="$(uv_current_version || true)"

if [ "$uv_present_version" = "$UV_VERSION" ]; then
  info "uv $UV_VERSION OK"
else
  if [ "$INSTALL_UV" = "1" ]; then
    if [ -n "$uv_present_version" ]; then
      info "found uv $uv_present_version; installing pinned $UV_VERSION"
    else
      info "uv not found; installing $UV_VERSION with the official installer"
    fi
    curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh
    # The installer drops uv in ~/.local/bin (or $XDG_BIN_HOME); make it usable now.
    if [ -f "$HOME/.local/bin/env" ]; then
      # shellcheck disable=SC1091
      . "$HOME/.local/bin/env"
    fi
    case ":$PATH:" in
      *":$HOME/.local/bin:"*) : ;;
      *) PATH="$HOME/.local/bin:$PATH"; export PATH ;;
    esac
    got="$(uv_current_version || true)"
    [ "$got" = "$UV_VERSION" ] || fail "uv $UV_VERSION not on PATH after install (found '${got:-none}'). Open a new shell or add \$HOME/.local/bin to PATH, then re-run."
    info "uv $UV_VERSION installed"
  else
    printf '\n' >&2
    if [ -n "$uv_present_version" ]; then
      printf 'uv %s is present but Cubed Core pins uv %s.\n' "$uv_present_version" "$UV_VERSION" >&2
    else
      printf 'uv is required (pinned to %s) and was not found.\n' "$UV_VERSION" >&2
    fi
    printf 'Install it with:\n  %s\n  source "$HOME/.local/bin/env"\n' "$uv_installer_line" >&2
    printf 'Or let this script do it:\n  ./setup.sh --install-uv\n' >&2
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# Phase 2: bootstrap (venv + uv sync + npm ci) and build the frontend
# ---------------------------------------------------------------------------
info "Phase 2: bootstrapping the project (venv, uv sync, npm ci)"
make bootstrap BOOTSTRAP_PYTHON="$PYTHON_BIN"

info "Building the web frontend"
make web-build

# ---------------------------------------------------------------------------
# Phase 3 (optional): release assets
# ---------------------------------------------------------------------------
if [ "$WITH_ASSETS" = "1" ]; then
  info "Phase 3: downloading release assets"
  if [ -n "$ASSETS_BASE_URL" ]; then
    make download-assets RELEASE_ASSET_BASE_URL="$ASSETS_BASE_URL"
  else
    make download-assets
  fi
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
trap - ERR
cat <<'DONE'

============================================================
You're ready.

Start the workbench (loopback port CUBED_CORE_PORT, default 8000):
    make workbench

Open the exact printed URL. It starts on Demo, the zero-compute replay. Demo
needs no manual asset download, model setup, or GPU.

Or hot-reload development (API on the same port, Vite on 5173):
    make dev

Verify the environment at any time:
    source .venv/bin/activate
    make doctor

If npm printed an advisory, read SECURITY.md before changing the lock. Do not
run a forced audit fix that silently changes supported dependency majors.

Maintained guides:
    README.md                    - setup and project overview
    docs/tutorials/DECODE.md    - Decode and Runs
    docs/tutorials/LABEL.md     - optional Label tool
    docs/CLOUD_GPU.md           - local and remote CUDA
    docs/DATASET.md             - public Hugging Face corpus
============================================================
DONE

if [ "$WITH_ASSETS" != "1" ]; then
  cat <<'ASSETS'
No release assets are needed to view the published demo replay.

Optional: fetch the tracker runtime, gtD1s video, read-trust model, and demo
calibration only when you want to import the video or run the pipeline:
    make download-assets
If a download returns 404, see docs/tutorials/DECODE.md and confirm that the
selected release and manifest are published.

To use the camera model in Label, the base workbench is not enough.
Prepare the optional Label environment and launch target:
    make bootstrap-label-cpu
    make workbench-label-cpu
ASSETS
fi

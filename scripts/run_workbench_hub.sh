#!/usr/bin/env bash
# Load the ignored .env.hub configuration, then run the ordinary workbench.
#
# This is useful when the local UI should discover a configured remote GPU
# runner without exporting the same variables in every shell. Values already
# exported by the caller take precedence over values in .env.hub.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd)"
ENV_FILE="$REPO_ROOT/.env.hub"
DEV=0

usage() {
	cat <<'EOF'
Usage: scripts/run_workbench_hub.sh [--dev]

  --dev   Run `make dev` instead of the built `make workbench`.

The script loads ignored settings from .env.hub. Variables already exported
by the caller take precedence.
EOF
}

while [ $# -gt 0 ]; do
	case "$1" in
		--dev) DEV=1 ;;
		-h | --help)
			usage
			exit 0
			;;
		*)
			echo "run_workbench_hub.sh: unknown argument: $1" >&2
			usage >&2
			exit 2
			;;
	esac
	shift
done

if [ ! -f "$ENV_FILE" ]; then
	echo "run_workbench_hub.sh: creating $ENV_FILE"
	{
		echo "# Cubed Core workbench settings. Local only; never commit this file."
		echo "# Add remote GPU runner variables here when needed."
	} >"$ENV_FILE"
fi

# Keep the internal server token stable across hub restarts. The loopback UI
# obtains it automatically; users do not need to copy or configure it.
if ! grep -q '^CUBED_CORE_ADMIN_TOKEN=' "$ENV_FILE" 2>/dev/null; then
	minted_token="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
	printf 'CUBED_CORE_ADMIN_TOKEN=%s\n' "$minted_token" >>"$ENV_FILE"
fi

# Snapshot caller-supplied values for keys that .env.hub would otherwise
# overwrite. Indexed arrays keep this compatible with macOS's bash 3.2.
env_keys="$(grep -Eo '^[A-Za-z_][A-Za-z0-9_]*=' "$ENV_FILE" 2>/dev/null | sed 's/=$//')"
prior_keys=()
prior_vals=()
for key in $env_keys; do
	if [ -n "${!key+x}" ]; then
		prior_keys+=("$key")
		prior_vals+=("${!key}")
	fi
done

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

i=0
for key in ${prior_keys[@]+"${prior_keys[@]}"}; do
	export "${key}=${prior_vals[$i]}"
	i=$((i + 1))
done

cd "$REPO_ROOT"
if [ "$DEV" -eq 1 ]; then
	exec make dev
fi
exec make workbench

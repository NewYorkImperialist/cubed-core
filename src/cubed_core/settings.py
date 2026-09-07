from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from pathlib import Path

from .tracker_runtime import parse_onnx_provider_list

DEFAULT_MAX_UPLOAD_BYTES = 1024**3
MIN_ADMIN_TOKEN_LENGTH = 32
MAX_ADMIN_TOKEN_LENGTH = 512
DECODE_MODE_ENV = "CUBED_CORE_DECODE_MODE"
DECODE_COMMAND_ENV = "CUBED_CORE_DECODE_COMMAND"
DECODE_RUNNER_LABEL_ENV = "CUBED_CORE_DECODE_RUNNER_LABEL"
LOCAL_BROWSER_AUTH_ENV = "CUBED_CORE_LOCAL_BROWSER_AUTH"
DECODE_MODES = frozenset({"disabled", "native", "external"})
REQUIRED_REPOSITORY_PATHS = (
    "pyproject.toml",
    "config/tooling.json",
    "config/decode-runtime-v1.json",
    "schemas/ble-session-v1.schema.json",
    "schemas/capture-bundle-v1.schema.json",
    "schemas/capture-derivation-v1.schema.json",
    "schemas/model-artifact-manifest-v1.schema.json",
    "schemas/decode-job-request-v1.schema.json",
    "schemas/decode-job-receipt-v1.schema.json",
    "schemas/decode-ground-truth-diagnostic-v1.schema.json",
    "schemas/decode-result-v1.schema.json",
)


def _admin_token(value: str) -> str:
    if not value:
        return ""
    if not MIN_ADMIN_TOKEN_LENGTH <= len(value) <= MAX_ADMIN_TOKEN_LENGTH:
        raise ValueError(
            "CUBED_CORE_ADMIN_TOKEN must contain between "
            f"{MIN_ADMIN_TOKEN_LENGTH} and {MAX_ADMIN_TOKEN_LENGTH} characters"
        )
    if any(ord(character) < 0x21 or ord(character) > 0x7E for character in value):
        raise ValueError("CUBED_CORE_ADMIN_TOKEN must contain printable ASCII without whitespace")
    return value


def _boolean_env(value: str, *, variable: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"", "0", "false", "no", "off"}:
        return False
    if normalized in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(f"{variable} must be a boolean (1/0, true/false, yes/no, or on/off)")


def _argv(
    value: str,
    *,
    variable: str,
    reserved: frozenset[str] = frozenset({"--request", "--output"}),
    reserved_message: str = "--request or --output",
) -> tuple[str, ...]:
    if not value.strip():
        return ()
    try:
        command = tuple(shlex.split(value, posix=os.name != "nt"))
    except ValueError as exc:
        raise ValueError(f"{variable} must be valid shell-style argv") from exc
    if not command:
        return ()
    if len(command) > 128 or any(
        not token or len(token) > 4096 or "\0" in token for token in command
    ):
        raise ValueError(f"{variable} is too large or contains an invalid argument")
    if any(
        token in reserved or any(token.startswith(f"{flag}=") for flag in reserved)
        for token in command
    ):
        raise ValueError(f"{variable} must not include reserved {reserved_message} arguments")
    return command


def _decode_mode(value: str) -> str:
    # Decode runs the heavy local_camera_v1 pipeline, so a blank value is always
    # disabled. Configuring a command alone never turns execution on.
    normalized = value.strip()
    if not normalized:
        return "disabled"
    if normalized not in DECODE_MODES:
        raise ValueError(f"{DECODE_MODE_ENV} must be disabled, native, or external")
    return normalized


def _label_predict_command(value: str) -> tuple[str, ...]:
    return _argv(
        value,
        variable="CUBED_CORE_LABEL_PREDICT_COMMAND",
        reserved=frozenset({"--request", "--output", "--model"}),
        reserved_message="--request, --output, or --model",
    )


def _absolute_path_without_resolving(value: str, *, base: Path) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = base / candidate
    return Path(os.path.abspath(candidate))


@dataclass(frozen=True)
class Settings:
    repo_root: Path
    workspace: Path
    max_upload_bytes: int
    admin_token: str = ""
    # Allows one same-origin browser on a loopback-only serve to bootstrap the
    # existing admin token. The CLI derives this from its bind/public settings;
    # direct app construction and TestClient remain opt-in and token-protected.
    local_browser_auth: bool = False
    tracker_onnx_providers: tuple[str, ...] = ()
    # Decode execution is separately opt-in from tracking: it runs the full
    # local_camera_v1 reads-to-moves pipeline rather than a single model pass.
    decode_mode: str = "disabled"
    decode_command: tuple[str, ...] = ()
    decode_runner_label: str = ""
    label_predict_command: tuple[str, ...] = ()
    label_model_path: Path | None = None
    tracker_model_manifest: Path | None = None

    def missing_repository_paths(self) -> tuple[str, ...]:
        return tuple(
            relative
            for relative in REQUIRED_REPOSITORY_PATHS
            if not (self.repo_root / relative).is_file()
        )

    def require_repository_layout(self) -> None:
        missing = self.missing_repository_paths()
        if not missing:
            return
        rendered = ", ".join(missing)
        raise ValueError(
            "Cubed Core requires a complete source checkout or the supplied "
            "container image; a standalone Python wheel is not a supported runtime. "
            f"Repository root {self.repo_root} is missing: {rendered}. Clone the "
            "repository and use `pip install -e .`, or set CUBED_CORE_REPO_ROOT to "
            "that checkout."
        )

    @classmethod
    def from_env(cls, *, repo_root: Path | None = None) -> Settings:
        root_value = os.environ.get("CUBED_CORE_REPO_ROOT", "")
        discovered_root = Path(__file__).resolve().parents[2]
        if not root_value and not (discovered_root / "pyproject.toml").is_file():
            working_directory = Path.cwd()
            if (working_directory / "pyproject.toml").is_file():
                discovered_root = working_directory
        if root_value:
            discovered_root = Path(root_value)
        root = (repo_root or discovered_root).resolve()
        workspace_value = os.environ.get("CUBED_CORE_WORKSPACE", "")
        workspace = Path(workspace_value) if workspace_value else root / "workspace"
        if not workspace.is_absolute():
            workspace = root / workspace
        try:
            max_upload_bytes = int(
                os.environ.get("CUBED_CORE_MAX_UPLOAD_BYTES", DEFAULT_MAX_UPLOAD_BYTES)
            )
        except ValueError as exc:
            raise ValueError("CUBED_CORE_MAX_UPLOAD_BYTES must be an integer") from exc
        if max_upload_bytes < 1024:
            raise ValueError("CUBED_CORE_MAX_UPLOAD_BYTES must be at least 1024")
        decode_command = _argv(
            os.environ.get(DECODE_COMMAND_ENV, ""),
            variable=DECODE_COMMAND_ENV,
        )
        decode_mode = _decode_mode(os.environ.get(DECODE_MODE_ENV, ""))
        if decode_mode == "native" and decode_command:
            raise ValueError(
                f"{DECODE_MODE_ENV}=native rejects {DECODE_COMMAND_ENV}; "
                "the current-Python module runner is selected automatically"
            )
        settings = cls(
            repo_root=root,
            workspace=workspace.resolve(),
            max_upload_bytes=max_upload_bytes,
            admin_token=_admin_token(os.environ.get("CUBED_CORE_ADMIN_TOKEN", "")),
            local_browser_auth=_boolean_env(
                os.environ.get(LOCAL_BROWSER_AUTH_ENV, ""),
                variable=LOCAL_BROWSER_AUTH_ENV,
            ),
            tracker_onnx_providers=parse_onnx_provider_list(
                os.environ.get("CUBED_CORE_TRACKER_ONNX_PROVIDERS", "")
            ),
            decode_mode=decode_mode,
            decode_command=decode_command,
            decode_runner_label=os.environ.get(DECODE_RUNNER_LABEL_ENV, "").strip()[:120],
            label_predict_command=_label_predict_command(
                os.environ.get("CUBED_CORE_LABEL_PREDICT_COMMAND", "")
            ),
            label_model_path=(
                _absolute_path_without_resolving(
                    os.environ["CUBED_CORE_LABEL_MODEL_PATH"],
                    base=root,
                )
                if os.environ.get("CUBED_CORE_LABEL_MODEL_PATH", "")
                else None
            ),
            tracker_model_manifest=(
                _absolute_path_without_resolving(
                    os.environ["CUBED_CORE_TRACKER_MODEL_MANIFEST"],
                    base=root,
                )
                if os.environ.get("CUBED_CORE_TRACKER_MODEL_MANIFEST", "")
                else None
            ),
        )
        if repo_root is None:
            settings.require_repository_layout()
        return settings

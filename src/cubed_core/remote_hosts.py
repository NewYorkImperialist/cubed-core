from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .settings import Settings

REMOTE_HOSTS_SCHEMA = "cubed-core/remote-hosts-v1"
REMOTE_HOSTS_FILE_NAME = "remote-hosts.json"
REMOTE_HOSTS_MAX_BYTES = 256 * 1024
REMOTE_HOSTS_MAX_COUNT = 64
# The server's own CUBED_REMOTE_* environment always shows up under this id, so
# a host id declared in remote-hosts.json may not claim it.
ENVIRONMENT_HOST_ID = "environment"
_ID = re.compile(r"[a-z0-9][a-z0-9_-]*")


class RemoteHostsError(ValueError):
    pass


@dataclass(frozen=True)
class RemoteHost:
    """One remote GPU host a Decode job can run against.

    ``root``, ``decode_venv``, and ``nvdec`` are optional overrides for the
    matching CUBED_REMOTE_* runner variable. ``None`` keeps the runner's
    default.
    """

    id: str
    label: str
    ssh_dest: str
    ssh_port: int
    root: str | None = None
    decode_venv: str | None = None
    nvdec: str | None = None

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
        }

    def environment_overrides(self) -> dict[str, str]:
        """The CUBED_REMOTE_* values this host contributes to a runner subprocess."""

        overrides = {
            "CUBED_REMOTE_SSH_DEST": self.ssh_dest,
            "CUBED_REMOTE_SSH_PORT": str(self.ssh_port),
        }
        if self.root is not None:
            overrides["CUBED_REMOTE_ROOT"] = self.root
        if self.decode_venv is not None:
            overrides["CUBED_REMOTE_DECODE_VENV"] = self.decode_venv
        if self.nvdec is not None:
            overrides["CUBED_REMOTE_NVDEC"] = self.nvdec
        return overrides


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value} is not allowed")


def _object(value: Any, *, field: str, required: set[str], allowed: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RemoteHostsError(f"{field} must be an object")
    missing = required - value.keys()
    if missing:
        raise RemoteHostsError(f"{field} is missing {sorted(missing)[0]}")
    extra = value.keys() - allowed
    if extra:
        raise RemoteHostsError(f"{field} contains unsupported field {sorted(extra)[0]}")
    return value


def _string(value: Any, *, field: str, minimum: int = 1, maximum: int) -> str:
    if not isinstance(value, str) or not minimum <= len(value) <= maximum:
        raise RemoteHostsError(
            f"{field} must be a string between {minimum} and {maximum} characters"
        )
    if "\0" in value:
        raise RemoteHostsError(f"{field} may not contain a NUL byte")
    return value


def _optional_string(value: Any, *, field: str, maximum: int) -> str | None:
    if value is None:
        return None
    return _string(value, field=field, maximum=maximum)


def _port(value: Any, *, field: str) -> int:
    if type(value) is not int or not 1 <= value <= 65535:
        raise RemoteHostsError(f"{field} must be an integer between 1 and 65535")
    return value


def _host_id(value: Any, *, field: str) -> str:
    text = _string(value, field=field, maximum=64)
    if not _ID.fullmatch(text):
        raise RemoteHostsError(
            f"{field} must start with a lowercase letter or digit and contain only "
            "lowercase letters, digits, '-', or '_'"
        )
    if text == ENVIRONMENT_HOST_ID:
        raise RemoteHostsError(
            f"{field} may not use the reserved id {ENVIRONMENT_HOST_ID!r} "
            "(that id names the server's own environment configuration)"
        )
    return text


def _parse_host(raw: Any, *, index: int) -> RemoteHost:
    field = f"remote-hosts.json hosts[{index}]"
    item = _object(
        raw,
        field=field,
        required={"id", "label", "ssh_dest"},
        allowed={
            "id",
            "label",
            "ssh_dest",
            "ssh_port",
            "root",
            "decode_venv",
            "nvdec",
        },
    )
    return RemoteHost(
        id=_host_id(item["id"], field=f"{field}.id"),
        label=_string(item["label"], field=f"{field}.label", maximum=200),
        ssh_dest=_string(item["ssh_dest"], field=f"{field}.ssh_dest", maximum=300),
        ssh_port=_port(item.get("ssh_port", 22), field=f"{field}.ssh_port"),
        root=_optional_string(item.get("root"), field=f"{field}.root", maximum=1024),
        decode_venv=_optional_string(
            item.get("decode_venv"), field=f"{field}.decode_venv", maximum=1024
        ),
        nvdec=_optional_string(item.get("nvdec"), field=f"{field}.nvdec", maximum=32),
    )


def load_remote_hosts(workspace_root: Path) -> tuple[RemoteHost, ...]:
    """Load workspace/remote-hosts.json.

    A missing file is the normal, expected default state for most installs
    and returns an empty tuple rather than raising. A present-but-malformed
    file (bad schema, bad id, duplicate id, symlink escape) raises
    ``RemoteHostsError`` so the caller can turn it into a clear error instead
    of silently ignoring a broken config.
    """

    path = workspace_root / REMOTE_HOSTS_FILE_NAME
    if not path.exists() and not path.is_symlink():
        return ()
    if path.is_symlink():
        raise RemoteHostsError(f"{REMOTE_HOSTS_FILE_NAME} may not be a symlink")
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(workspace_root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise RemoteHostsError(f"{REMOTE_HOSTS_FILE_NAME} is unavailable") from exc
    try:
        size = resolved.stat().st_size
    except OSError as exc:
        raise RemoteHostsError(f"{REMOTE_HOSTS_FILE_NAME} is unavailable") from exc
    if size <= 0:
        raise RemoteHostsError(f"{REMOTE_HOSTS_FILE_NAME} is empty")
    if size > REMOTE_HOSTS_MAX_BYTES:
        raise RemoteHostsError(
            f"{REMOTE_HOSTS_FILE_NAME} exceeds the {REMOTE_HOSTS_MAX_BYTES}-byte limit"
        )
    try:
        value = json.loads(resolved.read_bytes(), parse_constant=_reject_nonfinite)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RemoteHostsError(f"{REMOTE_HOSTS_FILE_NAME} must be valid finite JSON") from exc
    value = _object(
        value,
        field=REMOTE_HOSTS_FILE_NAME,
        required={"schema", "schema_version", "hosts"},
        allowed={"schema", "schema_version", "hosts"},
    )
    if value["schema"] != REMOTE_HOSTS_SCHEMA or value["schema_version"] != 1:
        raise RemoteHostsError(
            f"{REMOTE_HOSTS_FILE_NAME} must use {REMOTE_HOSTS_SCHEMA} schema version 1"
        )
    raw_hosts = value["hosts"]
    if not isinstance(raw_hosts, list) or len(raw_hosts) > REMOTE_HOSTS_MAX_COUNT:
        raise RemoteHostsError(
            f"{REMOTE_HOSTS_FILE_NAME} hosts must be a list of at most "
            f"{REMOTE_HOSTS_MAX_COUNT} items"
        )
    hosts: list[RemoteHost] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_hosts):
        host = _parse_host(raw, index=index)
        if host.id in seen_ids:
            raise RemoteHostsError(
                f"{REMOTE_HOSTS_FILE_NAME} declares duplicate host id {host.id!r}"
            )
        seen_ids.add(host.id)
        hosts.append(host)
    return tuple(hosts)


def environment_default_host() -> RemoteHost | None:
    """Synthesize the "environment" virtual host from the server's own env.

    Whatever the server process is already configured to run against (via
    CUBED_REMOTE_SSH_DEST and friends, read directly at request time so a
    changed environment never needs a restart to show up here) is always a
    visible, selectable option, even with no hosts file at all.
    """

    ssh_dest = os.environ.get("CUBED_REMOTE_SSH_DEST", "").strip()
    if not ssh_dest:
        return None
    port_value = os.environ.get("CUBED_REMOTE_SSH_PORT", "").strip()
    try:
        ssh_port = int(port_value) if port_value else 22
    except ValueError:
        ssh_port = 22
    return RemoteHost(
        id=ENVIRONMENT_HOST_ID,
        label="Environment (server default)",
        ssh_dest=ssh_dest,
        ssh_port=ssh_port,
    )


def list_remote_hosts(workspace_root: Path) -> tuple[RemoteHost, ...]:
    """Configured hosts plus the synthesized environment default, if any."""

    hosts = load_remote_hosts(workspace_root)
    env_host = environment_default_host()
    return (env_host, *hosts) if env_host is not None else hosts


def resolve_remote_host(workspace_root: Path, host_id: str) -> RemoteHost:
    for host in list_remote_hosts(workspace_root):
        if host.id == host_id:
            return host
    raise RemoteHostsError(f"unknown remote host id {host_id!r}")


def build_remote_hosts_response(settings: Settings) -> dict[str, Any]:
    """The GET /api/remote-hosts payload: public host fields plus the env default id.

    Raises ``RemoteHostsError`` when workspace/remote-hosts.json is present but
    malformed; the route layer turns that into a clear error response instead
    of a 500.
    """

    hosts = list_remote_hosts(settings.workspace)
    return {
        "schema": REMOTE_HOSTS_SCHEMA,
        "hosts": [host.public() for host in hosts],
        "env_default_id": ENVIRONMENT_HOST_ID if environment_default_host() is not None else None,
    }

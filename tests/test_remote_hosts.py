from __future__ import annotations

import json
from pathlib import Path

import pytest

from cubed_core.remote_hosts import (
    ENVIRONMENT_HOST_ID,
    REMOTE_HOSTS_SCHEMA,
    RemoteHost,
    RemoteHostsError,
    environment_default_host,
    list_remote_hosts,
    load_remote_hosts,
    resolve_remote_host,
)


def _write_hosts(path: Path, hosts: list[dict[str, object]]) -> Path:
    payload = {
        "schema": REMOTE_HOSTS_SCHEMA,
        "schema_version": 1,
        "hosts": hosts,
    }
    hosts_path = path / "remote-hosts.json"
    hosts_path.write_text(json.dumps(payload), encoding="utf-8")
    return hosts_path


def test_missing_file_is_an_empty_list_not_an_error(tmp_path: Path) -> None:
    assert load_remote_hosts(tmp_path) == ()
    assert list_remote_hosts(tmp_path) == ()


def test_public_host_contract_only_exposes_picker_fields() -> None:
    host = RemoteHost(
        id="vast-4090",
        label="Vast RTX 4090",
        ssh_dest="root@203.0.113.7",
        ssh_port=2222,
    )
    assert host.public() == {"id": "vast-4090", "label": "Vast RTX 4090"}


def test_valid_load_returns_hosts_with_defaults_and_overrides(tmp_path: Path) -> None:
    _write_hosts(
        tmp_path,
        [
            {"id": "vast-4090", "label": "Vast RTX 4090", "ssh_dest": "root@203.0.113.7"},
            {
                "id": "office-box",
                "label": "Office GPU box",
                "ssh_dest": "cubed@10.0.0.5",
                "ssh_port": 2222,
                "root": "/workspace/cubed-core",
                "decode_venv": ".venv-decode-gpu",
                "nvdec": "require",
            },
        ],
    )

    hosts = load_remote_hosts(tmp_path)
    assert [host.id for host in hosts] == ["vast-4090", "office-box"]

    first = hosts[0]
    assert first.ssh_port == 22
    assert first.root is None
    assert first.environment_overrides() == {
        "CUBED_REMOTE_SSH_DEST": "root@203.0.113.7",
        "CUBED_REMOTE_SSH_PORT": "22",
    }

    second = hosts[1]
    assert second.ssh_port == 2222
    assert second.environment_overrides() == {
        "CUBED_REMOTE_SSH_DEST": "cubed@10.0.0.5",
        "CUBED_REMOTE_SSH_PORT": "2222",
        "CUBED_REMOTE_ROOT": "/workspace/cubed-core",
        "CUBED_REMOTE_DECODE_VENV": ".venv-decode-gpu",
        "CUBED_REMOTE_NVDEC": "require",
    }


def test_duplicate_id_is_rejected(tmp_path: Path) -> None:
    _write_hosts(
        tmp_path,
        [
            {"id": "dup", "label": "A", "ssh_dest": "a@host"},
            {"id": "dup", "label": "B", "ssh_dest": "b@host"},
        ],
    )
    with pytest.raises(RemoteHostsError, match="duplicate host id 'dup'"):
        load_remote_hosts(tmp_path)


@pytest.mark.parametrize(
    "bad_id",
    ["Bad-Caps", "-leading-dash", "_leading-underscore", "has space", ""],
)
def test_bad_id_charset_is_rejected(tmp_path: Path, bad_id: str) -> None:
    _write_hosts(tmp_path, [{"id": bad_id, "label": "Bad", "ssh_dest": "a@host"}])
    with pytest.raises(RemoteHostsError):
        load_remote_hosts(tmp_path)


def test_reserved_environment_id_is_rejected(tmp_path: Path) -> None:
    _write_hosts(
        tmp_path,
        [{"id": "environment", "label": "Sneaky", "ssh_dest": "a@host"}],
    )
    with pytest.raises(RemoteHostsError, match="reserved id"):
        load_remote_hosts(tmp_path)


def test_malformed_json_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "remote-hosts.json").write_text("not json", encoding="utf-8")
    with pytest.raises(RemoteHostsError, match="valid finite JSON"):
        load_remote_hosts(tmp_path)


def test_wrong_schema_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "remote-hosts.json").write_text(
        json.dumps({"schema": "wrong", "schema_version": 1, "hosts": []}),
        encoding="utf-8",
    )
    with pytest.raises(RemoteHostsError, match="schema version 1"):
        load_remote_hosts(tmp_path)


def test_symlinked_file_is_rejected(tmp_path: Path) -> None:
    real = tmp_path / "elsewhere.json"
    real.write_text(
        json.dumps({"schema": REMOTE_HOSTS_SCHEMA, "schema_version": 1, "hosts": []}),
        encoding="utf-8",
    )
    link = tmp_path / "remote-hosts.json"
    link.symlink_to(real)
    with pytest.raises(RemoteHostsError, match="symlink"):
        load_remote_hosts(tmp_path)


def test_environment_default_host_absent_without_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CUBED_REMOTE_SSH_DEST", raising=False)
    assert environment_default_host() is None
    assert list_remote_hosts(Path("/nonexistent-workspace-root")) == ()


def test_environment_default_host_present_when_configured(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("CUBED_REMOTE_SSH_DEST", "root@203.0.113.9")
    monkeypatch.setenv("CUBED_REMOTE_SSH_PORT", "2200")
    host = environment_default_host()
    assert host is not None
    assert host.id == ENVIRONMENT_HOST_ID
    assert host.ssh_dest == "root@203.0.113.9"
    assert host.ssh_port == 2200

    hosts = list_remote_hosts(tmp_path)
    assert hosts[0].id == ENVIRONMENT_HOST_ID


def test_resolve_remote_host_finds_configured_and_environment_hosts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("CUBED_REMOTE_SSH_DEST", "root@203.0.113.9")
    _write_hosts(tmp_path, [{"id": "vast-4090", "label": "Vast", "ssh_dest": "root@1.2.3.4"}])

    assert resolve_remote_host(tmp_path, "vast-4090").ssh_dest == "root@1.2.3.4"
    assert resolve_remote_host(tmp_path, ENVIRONMENT_HOST_ID).ssh_dest == "root@203.0.113.9"

    with pytest.raises(RemoteHostsError, match="unknown remote host id 'bogus'"):
        resolve_remote_host(tmp_path, "bogus")


def test_environment_default_host_falls_back_to_port_22_on_garbage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUBED_REMOTE_SSH_DEST", "root@203.0.113.9")
    monkeypatch.setenv("CUBED_REMOTE_SSH_PORT", "not-a-port")
    host = environment_default_host()
    assert host is not None
    assert host.ssh_port == 22

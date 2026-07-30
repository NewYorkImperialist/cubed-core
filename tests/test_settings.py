from __future__ import annotations

import pytest

from cubed_core.settings import Settings


def test_default_video_upload_limit_is_one_gib(tmp_path, monkeypatch) -> None:
    for name in ("CUBED_CORE_MAX_UPLOAD_BYTES", "CUBED_CORE_LOCAL_BROWSER_AUTH"):
        monkeypatch.delenv(name, raising=False)
    settings = Settings.from_env(repo_root=tmp_path)
    assert settings.max_upload_bytes == 1024**3
    assert settings.local_browser_auth is False


def test_video_upload_limit_is_configurable(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CUBED_CORE_MAX_UPLOAD_BYTES", str(384 * 1024**2))

    assert Settings.from_env(repo_root=tmp_path).max_upload_bytes == 384 * 1024**2


@pytest.mark.parametrize("value", ["short", "x" * 513, "x" * 31 + "\n"])
def test_configured_admin_token_rejects_weak_or_unsafe_values(
    tmp_path,
    monkeypatch,
    value: str,
) -> None:
    monkeypatch.setenv("CUBED_CORE_ADMIN_TOKEN", value)
    with pytest.raises(ValueError, match="CUBED_CORE_ADMIN_TOKEN"):
        Settings.from_env(repo_root=tmp_path)


def test_configured_admin_token_accepts_a_long_printable_secret(tmp_path, monkeypatch) -> None:
    value = "cubed-core-test-" + "a" * 32
    monkeypatch.setenv("CUBED_CORE_ADMIN_TOKEN", value)

    assert Settings.from_env(repo_root=tmp_path).admin_token == value


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", True),
        ("true", True),
        ("YES", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("No", False),
        ("off", False),
    ],
)
def test_local_browser_auth_parses_explicit_boolean_environment(
    tmp_path,
    monkeypatch,
    value: str,
    expected: bool,
) -> None:
    monkeypatch.setenv("CUBED_CORE_LOCAL_BROWSER_AUTH", value)

    assert Settings.from_env(repo_root=tmp_path).local_browser_auth is expected


def test_local_browser_auth_rejects_ambiguous_environment(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CUBED_CORE_LOCAL_BROWSER_AUTH", "automatic")

    with pytest.raises(ValueError, match="CUBED_CORE_LOCAL_BROWSER_AUTH"):
        Settings.from_env(repo_root=tmp_path)


def test_automatic_repository_root_fails_closed_for_incomplete_wheel_layout(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("CUBED_CORE_REPO_ROOT", str(tmp_path))
    with pytest.raises(ValueError, match="standalone Python wheel is not a supported runtime"):
        Settings.from_env()


def test_explicit_test_settings_can_report_missing_repository_files(tmp_path) -> None:
    settings = Settings.from_env(repo_root=tmp_path)
    assert "config/tooling.json" in settings.missing_repository_paths()
    with pytest.raises(ValueError, match="CUBED_CORE_REPO_ROOT"):
        settings.require_repository_layout()


def test_label_model_path_preserves_symlink_for_capability_check(
    tmp_path,
    monkeypatch,
) -> None:
    target = tmp_path / "weights.bin"
    target.write_bytes(b"weights")
    symlink = tmp_path / "model.bin"
    symlink.symlink_to(target)
    monkeypatch.setenv("CUBED_CORE_LABEL_MODEL_PATH", "model.bin")

    settings = Settings.from_env(repo_root=tmp_path)

    assert settings.label_model_path == symlink
    assert settings.label_model_path.is_symlink()


def test_tracker_model_manifest_path_is_resolved_without_following_symlinks(
    tmp_path,
    monkeypatch,
) -> None:
    target = tmp_path / "models" / "manifest.json"
    target.parent.mkdir()
    target.write_text("{}", encoding="utf-8")
    symlink = tmp_path / "tracker-models.json"
    symlink.symlink_to(target)
    monkeypatch.setenv("CUBED_CORE_TRACKER_MODEL_MANIFEST", "tracker-models.json")

    settings = Settings.from_env(repo_root=tmp_path)

    assert settings.tracker_model_manifest == symlink
    assert settings.tracker_model_manifest.is_symlink()


def test_tracker_provider_list_is_normalized_and_frozen(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(
        "CUBED_CORE_TRACKER_ONNX_PROVIDERS",
        " CUDAExecutionProvider, CPUExecutionProvider ",
    )
    settings = Settings.from_env(repo_root=tmp_path)

    assert settings.tracker_onnx_providers == (
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    )
    monkeypatch.setenv(
        "CUBED_CORE_TRACKER_ONNX_PROVIDERS",
        "CoreMLExecutionProvider",
    )
    assert settings.tracker_onnx_providers == (
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    )


@pytest.mark.parametrize(
    "value",
    [
        "cuda",
        "CUDAExecutionProvider,",
        "CUDAExecutionProvider,,CPUExecutionProvider",
        "CUDAExecutionProvider,CUDAExecutionProvider",
    ],
)
def test_tracker_provider_list_rejects_invalid_or_duplicate_values(
    tmp_path,
    monkeypatch,
    value: str,
) -> None:
    monkeypatch.setenv("CUBED_CORE_TRACKER_ONNX_PROVIDERS", value)
    with pytest.raises(ValueError, match="CUBED_CORE_TRACKER_ONNX_PROVIDERS"):
        Settings.from_env(repo_root=tmp_path)

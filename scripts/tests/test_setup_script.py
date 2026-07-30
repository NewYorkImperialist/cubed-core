from __future__ import annotations

import subprocess
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
SETUP_SCRIPT = REPOSITORY / "setup.sh"


def test_setup_script_is_valid_bash() -> None:
    subprocess.run(
        ["bash", "-n", str(SETUP_SCRIPT)],
        cwd=REPOSITORY,
        check=True,
    )


def test_setup_help_names_the_complete_optional_asset_group() -> None:
    result = subprocess.run(
        [str(SETUP_SCRIPT), "--help"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--with-assets" in result.stdout
    assert "runtime/demo/decode assets" in result.stdout


def test_setup_preflights_make_and_conditional_uv_installer_dependency() -> None:
    source = SETUP_SCRIPT.read_text(encoding="utf-8")

    assert "for tool in ffmpeg ffprobe git make; do" in source
    assert 'if [ "$INSTALL_UV" = "1" ]; then' in source
    assert 'warn "curl not found (needed by --install-uv)"' in source


def test_setup_macos_hint_installs_the_supported_keg_only_node_major() -> None:
    source = SETUP_SCRIPT.read_text(encoding="utf-8")

    assert "brew install node@22" in source
    assert "brew --prefix node@22" in source
    assert "brew install node   # Node 22.3+" not in source


def test_setup_downloads_the_complete_asset_group_once() -> None:
    source = SETUP_SCRIPT.read_text(encoding="utf-8")

    assert "make download-assets download-decode-support" not in source
    assert 'make download-assets RELEASE_ASSET_BASE_URL="$ASSETS_BASE_URL"' in source
    assert "\n    make download-assets\n" in source
    assert "downloads intentionally do not overwrite existing paths" in source

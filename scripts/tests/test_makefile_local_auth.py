from __future__ import annotations

from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
MAKEFILE = REPOSITORY / "Makefile"


def test_hot_reload_dev_uses_the_tokenless_loopback_bootstrap() -> None:
    source = MAKEFILE.read_text(encoding="utf-8")
    dev = source[source.index("dev:\n") : source.index("\n# Remote/GPU launcher")]

    assert "$(MAKE) api BROWSER_PORT=5173" in dev
    assert "Admin workbench:" not in dev
    assert "#admin=" not in dev

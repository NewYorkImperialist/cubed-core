from __future__ import annotations

import subprocess
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
HUB_SCRIPT = REPOSITORY / "scripts" / "run_workbench_hub.sh"


def test_hub_launcher_is_valid_bash() -> None:
    subprocess.run(["bash", "-n", str(HUB_SCRIPT)], cwd=REPOSITORY, check=True)


def test_hub_launcher_only_loads_local_env_and_runs_the_workbench() -> None:
    source = HUB_SCRIPT.read_text(encoding="utf-8")

    assert 'source "$ENV_FILE"' in source
    assert "exec make workbench" in source
    assert "localhost.run" not in source
    assert "CUBED_CORE_PUBLIC_URL" not in source
    assert "--tunnel" not in source

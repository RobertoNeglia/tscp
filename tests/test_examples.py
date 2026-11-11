"""Every example runs on its own config, so the examples and the tutorials stay in step."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
EXAMPLES = {
    "quickstart": "quickstart.yaml",
    "compose": "compose.yaml",
    "custom_component": "custom_component.yaml",
    "learned_score": "learned_score.yaml",
    "production": "production.yaml",
}


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_examples_run(name):
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), os.environ.get("PYTHONPATH", "")]),
    }
    config = ROOT / "examples" / "configs" / EXAMPLES[name]
    result = subprocess.run(
        [sys.executable, str(ROOT / "examples" / f"{name}.py"), "--config", str(config)],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()

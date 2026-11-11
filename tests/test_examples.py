"""Every example runs on its own config, so the examples and the tutorials stay in step."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tscp.imports import has_module

ROOT = Path(__file__).parents[1]
EXAMPLES = {
    "calibration": "calibration.yaml",
    "agaci": "agaci.yaml",
    "online": "online.yaml",
    "bootstrap_intervals": "enbpi.yaml",
    "quickstart": "quickstart.yaml",
    "compose": "compose.yaml",
    "custom_component": "custom_component.yaml",
    "learned_score": "learned_score.yaml",
    "production": "production.yaml",
}


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_examples_run(name):
    run_example(name, EXAMPLES[name])


@pytest.mark.parametrize(
    ("script", "config", "dependency"),
    [
        ("online", "online_pid.yaml", "statsmodels"),
        ("bootstrap_intervals", "spci.yaml", "sklearn_quantile"),
    ],
)
def test_optional_examples_run(script, config, dependency):
    if not has_module(dependency):
        pytest.skip(f"optional example needs {dependency}")
    run_example(script, config)


def run_example(name, settings):
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), os.environ.get("PYTHONPATH", "")]),
    }
    config = ROOT / "examples" / "configs" / settings
    result = subprocess.run(
        [sys.executable, str(ROOT / "examples" / f"{name}.py"), "--config", str(config)],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()

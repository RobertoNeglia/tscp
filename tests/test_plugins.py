"""Plugins add components and methods that specs and configs can name.

The example plugin in ``examples/plugin`` is made findable as if it were installed: its
source joins the import path and an entry point naming it is faked, so nothing is
installed.
"""

import importlib.metadata
import json
import sys
from pathlib import Path

import pytest
import yaml

from tscp import registry
from tscp.methods import compose, get_method
from tscp.testing import check_method
from tscp.utils import config as config_module
from tscp.utils import from_config

PLUGIN = Path(__file__).parents[1] / "examples" / "plugin"
MODULE = "tscp_example_plugin"


def forget_the_plugin():
    registry._LOADED_PLUGINS.discard("example")
    for kind in ("weights", "method"):
        registry.entries(kind).pop("linear_decay", None)
    for name in [name for name in config_module._CLASSES if name.startswith(f"{MODULE}.")]:
        del config_module._CLASSES[name]
    sys.modules.pop(MODULE, None)


def fake_entry_points(monkeypatch, value):
    entry_point = importlib.metadata.EntryPoint(
        name="example", value=value, group=registry.ENTRY_POINT_GROUP
    )
    real = importlib.metadata.entry_points

    def entry_points(**params):
        if params.get("group") == registry.ENTRY_POINT_GROUP:
            return importlib.metadata.EntryPoints([entry_point])
        return real(**params)

    monkeypatch.setattr(importlib.metadata, "entry_points", entry_points)


@pytest.fixture
def plugin(monkeypatch):
    forget_the_plugin()
    monkeypatch.syspath_prepend(str(PLUGIN / "src"))
    fake_entry_points(monkeypatch, MODULE)
    yield
    forget_the_plugin()


def test_a_spec_can_name_a_plugin_component(plugin):
    spec = yaml.safe_load((PLUGIN / "spec.yaml").read_text())
    assert MODULE not in sys.modules  # imported only once a spec names it
    method = compose(spec)
    assert type(method.weights).__name__ == "LinearDecayWeights"
    assert method.weights.window == 150
    check_method(lambda: compose(spec))


def test_plugin_methods_build_by_name_and_rebuild_from_their_config(plugin):
    method = get_method("linear_decay", window=50, alpha=0.2)
    assert "linear_decay" in registry.names("method")
    config = json.loads(json.dumps(method.to_config()))
    assert config["class"] == f"{MODULE}.LinearDecayCP"
    assert from_config(config).to_config() == config


def test_configs_load_the_plugin_that_defines_their_class(plugin):
    weights = from_config({"class": f"{MODULE}.LinearDecayWeights", "args": {"window": 10}})
    assert weights.window == 10 and MODULE in sys.modules


def test_a_broken_plugin_is_named_in_the_error(monkeypatch):
    forget_the_plugin()
    fake_entry_points(monkeypatch, "no_such_module_for_tscp")
    with pytest.raises(ImportError, match="plugin 'example'"):
        registry.lookup("weights", "linear_decay")

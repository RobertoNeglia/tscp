import importlib

import pytest

import tscp
from tscp.imports import has_module, require

MODULES = [f"tscp.{name}" for name in tscp._SUBPACKAGES] + ["tscp.typing", "tscp.imports"]


@pytest.mark.parametrize("name", MODULES)
def test_every_export_exists(name):
    module = importlib.import_module(name)
    assert module.__all__, f"{name} exports nothing"
    missing = [symbol for symbol in module.__all__ if not hasattr(module, symbol)]
    assert not missing, f"{name}.__all__ lists missing names {missing}"


def test_subpackages_load_lazily():
    assert tscp.methods is importlib.import_module("tscp.methods")
    assert "methods" in dir(tscp)
    with pytest.raises(AttributeError):
        _ = tscp.not_a_subpackage


def test_version_is_a_string():
    assert isinstance(tscp.__version__, str)


def test_require_names_the_extra():
    assert has_module("torch")
    assert require("torch").__name__ == "torch"
    with pytest.raises(ImportError, match=r"tscp\[nothing\]"):
        require("tscp_module_that_does_not_exist", extra="nothing")

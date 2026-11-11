import pytest

from tscp import registry
from tscp.methods import method_map
from tscp.scores import AbsoluteScore
from tscp.weights import ExponentialWeights


def test_builtins_are_registered_by_kind():
    assert {"absolute", "signed", "normalized", "cqr"} <= set(registry.names("score"))
    assert {"split", "nexcp", "aci"} <= set(registry.names("method"))
    assert registry.lookup("weights", "Exponential") is ExponentialWeights  # any case
    assert registry.build("weights", "exponential", decay=0.9).decay == 0.9


def test_build_checks_its_arguments():
    with pytest.raises(TypeError, match="does not take"):
        registry.build("weights", "exponential", window=3)
    assert registry.build("weights", "exponential", window=3, ignore_unknown=True).decay == 0.99
    with pytest.raises(ValueError, match="unknown score 'nope'"):
        registry.lookup("score", "nope")
    with pytest.raises(ValueError, match="unknown kind"):
        registry.lookup("region", "box")


def test_taken_names_raise_unless_replaced():
    class Other(AbsoluteScore):
        pass

    with pytest.raises(ValueError, match="already registered"):
        registry.register("score", "absolute")(Other)
    try:
        registry.register("score", "absolute", replace=True)(Other)
        assert registry.lookup("score", "absolute") is Other
    finally:
        registry.register("score", "absolute", replace=True)(AbsoluteScore)
    registry.register("score", "absolute")(AbsoluteScore)  # the same class again: no-op
    assert registry.lookup("score", "absolute") is AbsoluteScore


def test_the_method_map_is_the_live_method_table():
    assert registry.entries("method") is method_map

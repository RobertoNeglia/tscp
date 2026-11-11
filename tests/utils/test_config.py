import json

import numpy as np
import pytest
import torch

from tscp.utils import Component, array_fingerprint, config_hash, from_config


class Leaf(Component):
    def __init__(self, a: float = 1.0, names=("x",)):
        super().__init__()
        self.a = a


class Node(Component):
    def __init__(self, leaf: Leaf, scale: float = 2.0, **extra):
        super().__init__()
        self.leaf = leaf


class Child(Leaf):
    def __init__(self, b: int = 3):
        super().__init__(a=0.5)


class Fitted(Component):
    def fit(self, archive):
        return self


def test_records_constructor_arguments_with_defaults():
    assert Leaf(a=2.0).get_config() == {"a": 2.0, "names": ["x"]}


def test_nested_components_round_trip_through_json():
    node = Node(Leaf(3.0), scale=4.0, tag="t")
    config = json.loads(json.dumps(node.to_config()))
    rebuilt = from_config(config)
    assert isinstance(rebuilt.leaf, Leaf) and rebuilt.leaf.a == 3.0
    assert rebuilt.get_config() == node.get_config()
    assert node.get_config()["tag"] == "t"


def test_a_subclass_records_its_own_arguments():
    assert Child(b=5).get_config() == {"b": 5}


def test_unknown_classes_and_values_are_refused():
    with pytest.raises(ValueError, match="unknown class"):
        from_config({"class": "nowhere.Nothing", "args": {}})
    with pytest.raises(ValueError, match="exactly the keys"):
        from_config({"class": "nowhere.Nothing"})
    with pytest.raises(TypeError, match="cannot record"):
        Leaf(a=torch.tensor(1.0)).get_config()


def test_trainable_means_fit_is_overridden():
    assert not Leaf().is_trainable
    assert Fitted().is_trainable


def test_config_hash_ignores_key_order():
    assert config_hash({"b": 1, "a": [1, 2]}) == config_hash({"a": [1, 2], "b": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})
    with pytest.raises(ValueError):
        config_hash({"a": float("nan")})


def test_array_fingerprint_sees_names_dtypes_shapes_and_values():
    base = array_fingerprint({"x": np.arange(4)})
    assert base == array_fingerprint({"x": torch.arange(4)})
    for other in (
        {"y": np.arange(4)},
        {"x": np.arange(4, dtype=np.int32)},
        {"x": np.arange(4).reshape(2, 2)},
        {"x": np.arange(1, 5)},
    ):
        assert array_fingerprint(other) != base

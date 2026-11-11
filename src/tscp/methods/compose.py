"""Methods built from specs of short names, such as specs read from YAML."""

from collections.abc import Mapping
from typing import Any

from tscp import registry
from tscp.methods.base import AXES, ConformalPredictor
from tscp.utils import Component, from_config

_SETTINGS = ("alpha", "chunk_size")


def compose(spec: Mapping[str, Any]) -> ConformalPredictor:
    """Build a method from a spec.

    A spec names either a registered preset with its arguments::

        {"method": "nexcp", "decay": 0.99, "alpha": 0.1}

    or one component per axis, run by :class:`~tscp.methods.ConformalPredictor`::

        {
            "score": "signed",
            "source": "expanding",
            "weights": {"name": "exponential", "decay": 0.99},
            "update": {"name": "aci", "gamma": 0.005},
            "alpha": 0.1,
        }

    A component can be given as its short name, as ``{"name": ..., **arguments}``, as a
    full config ``{"class": ..., "args": ...}`` (what
    :meth:`~tscp.utils.ConfigMixin.to_config` returns), or as a component. Inside
    arguments, a component of another kind states its kind:
    ``{"kind": "features", "name": "phase", "period": 24}``. Names come from
    :mod:`tscp.registry`, so installed plugins can be named too.

    Raises:
        ValueError: If the spec has unknown keys, no score, or names nothing registered.
    """
    spec = dict(spec)
    if "method" in spec:
        name = spec.pop("method")
        arguments = {
            key: _component(key, value) if key in AXES else _resolve(value)
            for key, value in spec.items()
        }
        return registry.build("method", name, **arguments)
    unknown = sorted(set(spec) - set(AXES) - set(_SETTINGS))
    if unknown:
        raise ValueError(
            f"unknown keys {unknown} in a spec; use 'method' for a preset, or the axes "
            f"{AXES} and the settings {_SETTINGS}"
        )
    if "score" not in spec:
        raise ValueError("a spec needs a 'score', or a 'method' that names a preset")
    arguments = {
        key: _component(key, value) if key in AXES else value for key, value in spec.items()
    }
    return ConformalPredictor(**arguments)


def _component(kind: str, value: Any) -> Any:
    """Build a component of ``kind`` from any of the forms a spec allows."""
    if value is None or isinstance(value, Component):
        return value
    if isinstance(value, str):
        return registry.build(kind, value)
    if isinstance(value, Mapping):
        if set(value) == {"class", "args"}:
            return from_config(value)
        if "name" in value:
            arguments = {k: _resolve(v) for k, v in value.items() if k not in ("name", "kind")}
            return registry.build(value.get("kind", kind), value["name"], **arguments)
    raise ValueError(f"cannot build a {kind} from {value!r}")


def _resolve(value: Any) -> Any:
    """Build the components nested in an argument value; leave other values alone."""
    if isinstance(value, Mapping):
        if set(value) == {"class", "args"}:
            return from_config(value)
        if "kind" in value and "name" in value:
            return _component(value["kind"], value)
        return {key: _resolve(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve(item) for item in value]
    return value


__all__ = ["compose"]

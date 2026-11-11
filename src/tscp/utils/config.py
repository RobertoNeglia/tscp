"""Constructor-argument configs, so presets and components rebuild from JSON.

Every subclass of :class:`ConfigMixin` records the arguments its constructor was called
with. :meth:`ConfigMixin.to_config` turns them into a JSON-serializable dict, nesting
components as ``{"class": ..., "args": ...}``, and :func:`from_config` rebuilds the object.
Only classes that subclass :class:`ConfigMixin` can be rebuilt, so a config cannot import
arbitrary code.
"""

import functools
import inspect
from collections.abc import Mapping
from typing import Any

_CLASSES: dict[str, type] = {}


def qualified_name(cls: type) -> str:
    """Return ``module.QualifiedName``, the key a config uses for ``cls``."""
    return f"{cls.__module__}.{cls.__qualname__}"


def _record_arguments(cls: type, signature: inspect.Signature, args, kwargs) -> dict[str, Any]:
    bound = signature.bind(None, *args, **kwargs)
    bound.apply_defaults()
    arguments = dict(bound.arguments)
    arguments.pop(next(iter(signature.parameters)))  # self
    for name, parameter in signature.parameters.items():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            arguments.update(arguments.pop(name, {}))
        elif parameter.kind is inspect.Parameter.VAR_POSITIONAL and arguments.pop(name, ()):
            raise TypeError(f"{cls.__name__} was given *args, which a config cannot record")
    return arguments


class ConfigMixin:
    """Record constructor arguments and rebuild objects from them.

    The config is the constructor call, not the current state: changing an attribute after
    construction does not change it. Fitted state belongs in ``state_dict()``.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _CLASSES[qualified_name(cls)] = cls
        init = cls.__dict__.get("__init__")
        if init is None or getattr(init, "_records_config", False):
            return
        signature = inspect.signature(init)

        @functools.wraps(init)
        def __init__(self, *args, **kwargs):
            # The most derived constructor runs first; parents called through super() must
            # not overwrite its record.
            if "_config_arguments" not in self.__dict__:
                arguments = _record_arguments(cls, signature, args, kwargs)
                object.__setattr__(self, "_config_arguments", arguments)
            init(self, *args, **kwargs)

        __init__._records_config = True
        cls.__init__ = __init__

    def get_config(self) -> dict[str, Any]:
        """Return the constructor arguments as a JSON-serializable dict."""
        arguments = self.__dict__.get("_config_arguments", {})
        return {name: _encode(value, name) for name, value in arguments.items()}

    def to_config(self) -> dict[str, Any]:
        """Return ``{"class": ..., "args": ...}``, which :func:`from_config` rebuilds."""
        return {"class": qualified_name(type(self)), "args": self.get_config()}


def _encode(value: Any, name: str) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, ConfigMixin):
        return value.to_config()
    if isinstance(value, list | tuple):
        return [_encode(item, name) for item in value]
    if isinstance(value, Mapping) and all(isinstance(key, str) for key in value):
        return {key: _encode(item, f"{name}.{key}") for key, item in value.items()}
    raise TypeError(
        f"argument {name!r} has type {type(value).__name__}, which a config cannot record; "
        "pass numbers, strings, lists, dicts or tscp components"
    )


def _decode(value: Any) -> Any:
    if isinstance(value, Mapping):
        if set(value) == {"class", "args"}:
            return from_config(value)
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item) for item in value]
    return value


def from_config(config: Mapping[str, Any]) -> ConfigMixin:
    """Rebuild an object from :meth:`ConfigMixin.to_config` output.

    Raises:
        ValueError: If the class is unknown. The module that defines it must have been
            imported, which is always true for tscp's own classes once their subpackage loads.
    """
    if set(config) != {"class", "args"}:
        raise ValueError("a config must have exactly the keys 'class' and 'args'")
    name = config["class"]
    cls = _CLASSES.get(name)
    if cls is None:
        raise ValueError(f"unknown class {name!r}; import the module that defines it first")
    return cls(**{key: _decode(value) for key, value in config["args"].items()})


__all__ = ["ConfigMixin", "from_config", "qualified_name"]

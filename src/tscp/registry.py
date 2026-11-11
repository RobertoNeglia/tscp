"""Short names for components and methods, and plugins that add more.

Every built-in component and preset is registered under a kind and a short name, so that
a config or a spec can name it::

    from tscp import registry

    weights = registry.build("weights", "exponential", decay=0.99)

A package that adds components registers them the same way, with :func:`register`, and
declares an entry point in the group ``tscp.plugins`` whose value is the module that does
the registering. tscp imports those modules when a name or a class is not found, so a
spec or config that names a plugin's component works once the plugin is installed.
"""

import importlib
import importlib.metadata
import inspect
from collections.abc import Callable
from typing import Any

from tscp.utils.config import qualified_name

KINDS = ("score", "source", "weights", "quantile", "update", "features", "method")
"""Kinds of registered classes: the component axes, feature maps, and methods."""

ENTRY_POINT_GROUP = "tscp.plugins"
"""The entry-point group of plugins: each entry point names a module to import."""

_ENTRIES: dict[str, dict[str, type]] = {kind: {} for kind in KINDS}
_BUILTINS = (
    "tscp.scores",
    "tscp.sources",
    "tscp.weights",
    "tscp.quantile",
    "tscp.online",
    "tscp.features",
    "tscp.methods",
)
_LOADED_PLUGINS: set[str] = set()


def _check_kind(kind: str) -> None:
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}; the kinds are {KINDS}")


def register(kind: str, name: str, *, replace: bool = False) -> Callable[[type], type]:
    """Class decorator that makes a class buildable as ``build(kind, name, ...)``.

    Names are case-insensitive. Registering a name that another class holds raises a
    ValueError, so two packages never silently override each other or tscp's own
    components; pass ``replace=True`` to take a name on purpose. Registering the same
    class again, as when its module is reloaded, does nothing.

    Args:
        kind: One of :data:`KINDS`.
        name: The short name.
        replace: Take the name even if another class holds it.
    """
    _check_kind(kind)
    key = name.lower()

    def decorator(cls: type) -> type:
        current = _ENTRIES[kind].get(key)
        same = current is not None and qualified_name(current) == qualified_name(cls)
        if current is not None and not same and not replace:
            raise ValueError(
                f"{kind} {key!r} is already registered to {qualified_name(current)}; "
                "pass replace=True to take the name"
            )
        _ENTRIES[kind][key] = cls
        return cls

    return decorator


def entries(kind: str) -> dict[str, type]:
    """The live name-to-class table of a kind.

    It fills as modules register; unlike :func:`lookup`, reading it imports nothing.
    """
    _check_kind(kind)
    return _ENTRIES[kind]


def _load_builtins() -> None:
    for module in _BUILTINS:
        importlib.import_module(module)


def load_plugins() -> list[str]:
    """Import the module of every installed plugin not imported yet.

    Returns:
        The names of the entry points imported by this call.

    Raises:
        ImportError: If a plugin's module fails to import; the message names the plugin.
    """
    loaded = []
    for entry_point in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP):
        if entry_point.name in _LOADED_PLUGINS:
            continue
        try:
            entry_point.load()
        except Exception as error:
            raise ImportError(
                f"the tscp plugin {entry_point.name!r} ({entry_point.value}) failed to load: "
                f"{error}"
            ) from error
        _LOADED_PLUGINS.add(entry_point.name)
        loaded.append(entry_point.name)
    return loaded


def lookup(kind: str, name: str) -> type:
    """Return the class registered as ``name`` for ``kind``.

    The built-in components are imported first; plugins are imported if the name is
    still unknown.

    Raises:
        ValueError: If no built-in or installed plugin registers the name.
    """
    _check_kind(kind)
    _load_builtins()
    key = name.lower()
    if key not in _ENTRIES[kind]:
        load_plugins()
    if key not in _ENTRIES[kind]:
        raise ValueError(f"unknown {kind} {name!r}; registered: {sorted(_ENTRIES[kind])}")
    return _ENTRIES[kind][key]


def build(kind: str, name: str, *, ignore_unknown: bool = False, **kwargs: Any) -> Any:
    """Build the class registered as ``name`` for ``kind`` with keyword arguments.

    Args:
        kind: One of :data:`KINDS`.
        name: A registered name, in any case.
        ignore_unknown: Drop keyword arguments the class does not take, which helps when
            one config sweeps several classes. By default they raise.
        **kwargs: Arguments of the class.

    Raises:
        ValueError: If the name is not registered.
        TypeError: If an argument is unknown and ``ignore_unknown`` is False.
    """
    cls = lookup(kind, name)
    parameters = inspect.signature(cls.__init__).parameters
    takes_any = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    accepted = [p for p in parameters if p != "self"]
    unknown = [] if takes_any else sorted(set(kwargs) - set(accepted))
    if unknown:
        if not ignore_unknown:
            raise TypeError(f"{cls.__name__} does not take {unknown}; it takes {accepted}")
        kwargs = {key: value for key, value in kwargs.items() if key in accepted}
    return cls(**kwargs)


def names(kind: str) -> list[str]:
    """The registered names of a kind, built-ins and installed plugins included."""
    _check_kind(kind)
    _load_builtins()
    load_plugins()
    return sorted(_ENTRIES[kind])


__all__ = [
    "ENTRY_POINT_GROUP",
    "KINDS",
    "build",
    "entries",
    "load_plugins",
    "lookup",
    "names",
    "register",
]

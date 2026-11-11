"""Published methods as presets: one choice per axis, run by :class:`ConformalPredictor`.

Presets are registered in :data:`method_map` and built by name with :func:`get_method`,
as poolers are in tgp.
"""

import inspect

from .aci import ACI
from .base import AXES, ConformalPredictor
from .nexcp import NexCP
from .split import SplitCP

method_map: dict[str, type[ConformalPredictor]] = {
    "split": SplitCP,
    "nexcp": NexCP,
    "aci": ACI,
}
"""Registered presets by short name. Add an entry to make a new preset buildable by name."""


def get_method(name: str, *, ignore_unknown: bool = False, **kwargs) -> ConformalPredictor:
    """Build a registered preset by name.

    Args:
        name: A key of :data:`method_map`, in any case.
        ignore_unknown: Drop keyword arguments the preset does not take, which helps when
            one config sweeps several presets. By default they raise.
        **kwargs: Arguments of the preset.

    Raises:
        ValueError: If ``name`` is not registered.
        TypeError: If an argument is unknown and ``ignore_unknown`` is False.
    """
    key = name.lower()
    if key not in method_map:
        raise ValueError(f"unknown method {name!r}; registered methods: {sorted(method_map)}")
    cls = method_map[key]
    parameters = inspect.signature(cls.__init__).parameters
    takes_any = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    accepted = [p for p in parameters if p != "self"]
    unknown = [] if takes_any else sorted(set(kwargs) - set(accepted))
    if unknown:
        if not ignore_unknown:
            raise TypeError(f"{cls.__name__} does not take {unknown}; it takes {accepted}")
        kwargs = {key: value for key, value in kwargs.items() if key in accepted}
    return cls(**kwargs)


__all__ = ["ACI", "AXES", "ConformalPredictor", "NexCP", "SplitCP", "get_method", "method_map"]

"""Published methods as presets: one choice per axis, run by :class:`ConformalPredictor`.

Presets are registered in :mod:`tscp.registry` under the kind ``"method"``;
:data:`method_map` is that table, and :func:`get_method` builds a preset by name, as
poolers are built in tgp. :func:`compose` builds a method from a spec of short names.
"""

from tscp import registry

from .aci import ACI
from .agaci import AgACI
from .base import AXES, ConformalPredictor
from .compose import compose
from .dtaci import DtACI
from .enbpi import EnbPI
from .io import load_method, save_method
from .mondrian import MondrianCP
from .nexcp import NexCP
from .pid import ConformalPI, ConformalPID
from .spci import SPCI
from .split import SplitCP
from .windowed import WindowedCP

method_map: dict[str, type[ConformalPredictor]] = registry.entries("method")
"""Registered presets by short name: the live table of :mod:`tscp.registry`."""


def get_method(name: str, *, ignore_unknown: bool = False, **kwargs) -> ConformalPredictor:
    """Build a registered preset by name.

    Args:
        name: A registered method name, in any case; installed plugins count.
        ignore_unknown: Drop keyword arguments the preset does not take, which helps when
            one config sweeps several presets. By default they raise.
        **kwargs: Arguments of the preset.

    Raises:
        ValueError: If ``name`` is not registered.
        TypeError: If an argument is unknown and ``ignore_unknown`` is False.
    """
    return registry.build("method", name, ignore_unknown=ignore_unknown, **kwargs)


__all__ = [
    "ACI",
    "AgACI",
    "AXES",
    "ConformalPredictor",
    "ConformalPI",
    "ConformalPID",
    "DtACI",
    "EnbPI",
    "MondrianCP",
    "NexCP",
    "SplitCP",
    "SPCI",
    "WindowedCP",
    "compose",
    "get_method",
    "load_method",
    "method_map",
    "save_method",
]

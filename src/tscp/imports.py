"""Guards for optional dependencies.

Core tscp needs only torch and numpy. A component that needs another package imports it
through :func:`require`, so the error names the extra that provides it.
"""

import importlib
import importlib.util
from types import ModuleType


def has_module(name: str) -> bool:
    """Return whether ``name`` can be imported, without importing it."""
    return importlib.util.find_spec(name) is not None


def require(name: str, extra: str | None = None) -> ModuleType:
    """Import an optional dependency, or raise an ImportError that says how to install it.

    Args:
        name: Module to import, such as ``"scipy"``.
        extra: The tscp extra that provides it, if there is one.
    """
    try:
        return importlib.import_module(name)
    except ImportError as error:
        hint = f"pip install 'tscp[{extra}]'" if extra else f"pip install {name}"
        message = f"{name} is needed for this feature; install it with `{hint}`."
        raise ImportError(message) from error


__all__ = ["has_module", "require"]

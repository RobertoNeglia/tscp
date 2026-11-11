"""tscp: conformal prediction for time series, in PyTorch.

Subpackages load on first access, so ``import tscp`` stays light::

    import tscp

    archive = tscp.datasets.synthetic.gaussian_control(horizon=1)
    method = tscp.methods.get_method("split", alpha=0.1)
"""

import importlib

__version__ = "0.1.0.dev0"

_SUBPACKAGES = (
    "data",
    "datasets",
    "features",
    "methods",
    "metrics",
    "nn",
    "online",
    "quantile",
    "regions",
    "registry",
    "scores",
    "sources",
    "testing",
    "utils",
    "weights",
)

__all__ = ["__version__", *_SUBPACKAGES]


def __getattr__(name: str):
    if name in _SUBPACKAGES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)

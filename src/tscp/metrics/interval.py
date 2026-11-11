"""Coverage and size of regions; interval-specific scores."""

import torch
from torch import Tensor

from tscp.regions import Interval, Region


def _mean(values: Tensor, observed: Tensor | None) -> Tensor:
    values = values.to(torch.float64)
    if observed is None:
        return values.mean()
    return values[observed].mean()


def coverage(region: Region, residuals: Tensor, observed: Tensor | None = None) -> Tensor:
    """Fraction of (observed) targets inside their region."""
    return _mean(region.contains(residuals), observed)


def coverage_gap(
    region: Region, residuals: Tensor, alpha: float, observed: Tensor | None = None
) -> Tensor:
    """Coverage minus its target ``1 - alpha``; negative means under-coverage."""
    return coverage(region, residuals, observed) - (1 - alpha)


def log_width(interval: Interval) -> Tensor:
    """Log-width of each interval: its log-volume in one dimension."""
    return interval.log_volume()


def mean_width(interval: Interval, observed: Tensor | None = None) -> Tensor:
    """Arithmetic mean width.

    Only for comparing with published one-step tables, which report it. Report size as
    log-width otherwise: a mean of raw widths is dominated by the widest.
    """
    return _mean(interval.width(), observed)


def winkler_score(interval: Interval, residuals: Tensor, alpha: float) -> Tensor:
    """Winkler (interval) score of each interval: width plus ``2/alpha`` times the miss."""
    below = (interval.lower - residuals).clamp(min=0)
    above = (residuals - interval.upper).clamp(min=0)
    return (interval.upper - interval.lower) + (2 / alpha) * (below + above)


__all__ = ["coverage", "coverage_gap", "log_width", "mean_width", "winkler_score"]

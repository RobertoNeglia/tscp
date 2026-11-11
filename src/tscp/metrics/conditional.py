"""Coverage by group and over rolling windows."""

import torch
from torch import Tensor


def _hits_and_counts(covered: Tensor, observed: Tensor | None) -> tuple[Tensor, Tensor]:
    counts = torch.ones_like(covered, dtype=torch.float64)
    if observed is not None:
        counts = observed.to(torch.float64)
    return covered.to(torch.float64) * counts, counts


def rolling_coverage(covered: Tensor, window: int, observed: Tensor | None = None) -> Tensor:
    """Coverage over each run of ``window`` consecutive query origins, per element.

    Args:
        covered: ``[Q, ...]`` bool, as returned by ``region.contains``.
        window: Number of consecutive origins per window.
        observed: Optional mask shaped like ``covered``; unobserved targets are skipped.

    Returns:
        ``[Q - window + 1, ...]``; NaN for a window without observed targets.
    """
    if not 1 <= window <= covered.shape[0]:
        raise ValueError("window must lie between 1 and the number of origins")
    hits, counts = _hits_and_counts(covered, observed)
    zero = torch.zeros_like(hits[:1])
    hits = torch.cat([zero, hits.cumsum(0)])
    counts = torch.cat([zero, counts.cumsum(0)])
    return (hits[window:] - hits[:-window]) / (counts[window:] - counts[:-window])


def worst_window_coverage(covered: Tensor, window: int, observed: Tensor | None = None) -> Tensor:
    """Lowest rolling coverage over every window and element."""
    rolling = rolling_coverage(covered, window, observed)
    return torch.where(rolling.isnan(), torch.inf, rolling).min()


def grouped_coverage(
    covered: Tensor, groups: Tensor, observed: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    """Coverage within each group of query origins, such as the hour of day.

    Args:
        covered: ``[Q, ...]`` bool.
        groups: ``[Q]`` integer group label of each origin.
        observed: Optional mask shaped like ``covered``.

    Returns:
        The sorted group labels and the coverage in each.
    """
    if groups.shape != covered.shape[:1]:
        raise ValueError("groups must hold one label per query origin")
    hits, counts = _hits_and_counts(covered, observed)
    hits = hits.reshape(covered.shape[0], -1).sum(1)
    counts = counts.reshape(covered.shape[0], -1).sum(1)
    labels, inverse = torch.unique(groups, return_inverse=True)
    group_hits = torch.zeros(len(labels), dtype=torch.float64, device=covered.device)
    group_counts = torch.zeros_like(group_hits)
    group_hits.index_add_(0, inverse, hits)
    group_counts.index_add_(0, inverse, counts)
    return labels, group_hits / group_counts


__all__ = ["grouped_coverage", "rolling_coverage", "worst_window_coverage"]

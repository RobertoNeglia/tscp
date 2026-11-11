"""The default evaluation report."""

import math
from typing import Any

import torch
from torch import Tensor

from tscp.data import ResidualArchive
from tscp.metrics.conditional import grouped_coverage, worst_window_coverage
from tscp.metrics.interval import winkler_score
from tscp.regions import Interval, Region


def evaluate(
    region: Region,
    archive: ResidualArchive,
    alpha: float,
    *,
    window: int | None = None,
    groups: Tensor | None = None,
) -> dict[str, Any]:
    """Score regions against the residuals they were made for.

    Size is the per-context log-volume divided by the region's dimension (for an
    interval, its log-width), averaged over bounded, nonempty regions only. Unbounded and
    empty regions are counted, never averaged in.

    Args:
        region: Regions from ``predict``, one per origin of ``archive``.
        archive: The rows the regions were made for, usually ``archive.split("test")``.
        alpha: The target miscoverage level.
        window: If given, also report the worst coverage over windows of this many origins.
        groups: If given, ``[Q]`` labels; also report coverage per label.

    Returns:
        ``num_targets``, ``coverage``, ``coverage_gap``, ``log_volume_per_dim``,
        ``infinite_fraction``, ``empty_fraction``; ``winkler`` (mean over bounded,
        nonempty intervals) for intervals; ``worst_window_coverage`` and
        ``grouped_coverage`` when requested.
    """
    residuals = archive.residuals
    if region.event_dims or tuple(region.batch_shape) != tuple(residuals.shape):
        raise ValueError("evaluate() handles regions with one element per residual for now")
    covered = region.contains(residuals)
    observed = archive.observed
    valid = torch.ones_like(covered) if observed is None else observed
    log_volume = region.log_volume()
    proper = valid & torch.isfinite(log_volume)

    def mean(values: Tensor, mask: Tensor) -> float:
        return float(values[mask].to(torch.float64).mean()) if mask.any() else math.nan

    report: dict[str, Any] = {
        "num_targets": int(valid.sum()),
        "coverage": mean(covered, valid),
        "log_volume_per_dim": mean(log_volume / region.dimension, proper),
        "infinite_fraction": mean(torch.isposinf(log_volume), valid),
        "empty_fraction": mean(torch.isneginf(log_volume), valid),
    }
    report["coverage_gap"] = report["coverage"] - (1 - alpha)
    if isinstance(region, Interval):
        report["winkler"] = mean(winkler_score(region, residuals, alpha), proper)
    if window is not None:
        report["worst_window_coverage"] = float(worst_window_coverage(covered, window, observed))
    if groups is not None:
        labels, values = grouped_coverage(covered, groups, observed)
        report["grouped_coverage"] = dict(zip(labels.tolist(), values.tolist(), strict=True))
    return report


__all__ = ["evaluate"]

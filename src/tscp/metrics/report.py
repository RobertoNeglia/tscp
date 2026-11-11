"""The default evaluation report."""

import dataclasses
import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from tscp.data import ResidualArchive
from tscp.metrics.conditional import grouped_coverage, worst_window_coverage
from tscp.metrics.interval import observed_regions, winkler_score
from tscp.regions import RESIDUAL_DIMS, Interval, Region


@dataclass(frozen=True)
class Report:
    """What :func:`evaluate` measures.

    Attributes:
        alpha: The target miscoverage level.
        num_regions: Regions evaluated: those whose targets are all observed.
        coverage: Fraction of evaluated regions that cover their targets.
        coverage_gap: ``coverage - (1 - alpha)``; negative means under-coverage.
        log_volume_per_dim: Mean log-volume per dimension (for an interval, the
            log-width), over bounded, nonempty regions only.
        infinite_fraction: Fraction of evaluated regions that are unbounded.
        empty_fraction: Fraction of evaluated regions that are empty or of zero volume.
        winkler: Mean Winkler score of bounded, nonempty intervals; None for other regions.
        coverage_by_step: Coverage at each horizon step, for regions made per step.
        log_width_by_step: Mean log-width at each horizon step, for intervals made per step.
        joint_coverage: Fraction of origins (and series, unless covered jointly) whose
            targets are covered at every step of the horizon.
        worst_window_coverage: Lowest coverage over windows of consecutive origins, if a
            window was given.
        grouped_coverage: Coverage per group label, if groups were given.
    """

    alpha: float
    num_regions: int
    coverage: float
    coverage_gap: float
    log_volume_per_dim: float
    infinite_fraction: float
    empty_fraction: float
    winkler: float | None = None
    coverage_by_step: tuple[float, ...] | None = None
    log_width_by_step: tuple[float, ...] | None = None
    joint_coverage: float | None = None
    worst_window_coverage: float | None = None
    grouped_coverage: dict[int, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """The report as a JSON-serializable dict."""
        return dataclasses.asdict(self)


def _mean(values: Tensor, mask: Tensor) -> float:
    return float(values[mask].to(torch.float64).mean()) if mask.any() else math.nan


def evaluate(
    region: Region,
    archive: ResidualArchive,
    alpha: float,
    *,
    window: int | None = None,
    groups: Tensor | None = None,
) -> Report:
    """Score regions against the residuals they were made for.

    Works for any region type. A region that covers some dimensions jointly counts as
    covering only if all its targets lie inside, and it is evaluated only if all its
    targets are observed. Size is the per-context log-volume divided by the region's
    dimension, averaged over bounded, nonempty regions only; unbounded and empty regions
    are counted, never averaged in.

    Args:
        region: Regions from ``predict``, one per row of ``archive`` and element not
            covered jointly.
        archive: The rows the regions were made for, usually ``archive.split("test")``.
        alpha: The target miscoverage level.
        window: If given, also report the worst coverage over windows of this many origins.
        groups: If given, ``[Q]`` labels; also report coverage per label.
    """
    residuals = archive.residuals
    sizes = dict(zip(RESIDUAL_DIMS, residuals.shape[1:], strict=True))
    kept = [name for name in RESIDUAL_DIMS if name not in region.event_dims]
    expected = (len(archive), *(sizes[name] for name in kept))
    if tuple(region.batch_shape) != expected:
        raise ValueError(
            f"regions must have batch shape {expected}: one per row and element not covered "
            f"jointly, not {tuple(region.batch_shape)}"
        )
    covered = region.contains(residuals)
    valid = observed_regions(region, archive.observed)
    if valid is None:
        valid = torch.ones_like(covered)
    log_volume = region.log_volume()
    proper = valid & torch.isfinite(log_volume)
    coverage = _mean(covered, valid)
    fields: dict[str, Any] = {
        "alpha": float(alpha),
        "num_regions": int(valid.sum()),
        "coverage": coverage,
        "coverage_gap": coverage - (1 - alpha),
        "log_volume_per_dim": _mean(log_volume / region.dimension, proper),
        "infinite_fraction": _mean(torch.isposinf(log_volume), valid),
        "empty_fraction": _mean(torch.isneginf(log_volume), valid),
    }
    if isinstance(region, Interval):
        fields["winkler"] = _mean(winkler_score(region, residuals, alpha), proper)
    if "step" in kept:
        step = 1  # the step dimension of the batch
        fields["coverage_by_step"] = tuple(
            _mean(covered.select(step, h), valid.select(step, h)) for h in range(sizes["step"])
        )
        if isinstance(region, Interval):
            fields["log_width_by_step"] = tuple(
                _mean(log_volume.select(step, h), proper.select(step, h))
                for h in range(sizes["step"])
            )
        fields["joint_coverage"] = _mean(covered.all(dim=step), valid.all(dim=step))
    else:
        fields["joint_coverage"] = coverage
    if window is not None:
        fields["worst_window_coverage"] = float(worst_window_coverage(covered, window, valid))
    if groups is not None:
        labels, values = grouped_coverage(covered, groups, valid)
        fields["grouped_coverage"] = dict(zip(labels.tolist(), values.tolist(), strict=True))
    return Report(**fields)


__all__ = ["Report", "evaluate"]

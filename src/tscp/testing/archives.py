"""Small synthetic archives for checks and tests."""

from dataclasses import replace

import torch

from tscp.data import ResidualArchive
from tscp.datasets import variance_shift

MONDAY = 1_767_571_200_000_000_000
"""2026-01-05 00:00 UTC, a Monday, in nanoseconds: the first origin of synthetic archives."""
HOUR = 3_600_000_000_000
"""One hour in nanoseconds: the base step of synthetic archives."""


def with_missing_targets(
    archive: ResidualArchive, fraction: float, seed: int = 0
) -> ResidualArchive:
    """Return ``archive`` with a random ``fraction`` of its targets marked unobserved.

    Unobserved residuals become exactly 0, as the archive requires. Targets that were
    already unobserved stay so.
    """
    if not 0 <= fraction < 1:
        raise ValueError("fraction must lie in [0, 1)")
    generator = torch.Generator().manual_seed(seed)
    observed = (torch.rand(archive.residuals.shape, generator=generator) >= fraction).to(
        archive.device
    )
    if archive.observed is not None:
        observed = observed & archive.observed
    return replace(archive, residuals=archive.residuals * observed, observed=observed)


def synthetic_archive(
    *,
    horizon: int = 2,
    num_series: int = 1,
    num_origins: int = 300,
    missing: float = 0.0,
    seed: int = 0,
) -> ResidualArchive:
    """A small archive whose error scale jumps during the test split, with forecaster outputs.

    The residuals come from :func:`~tscp.datasets.variance_shift`, with its default splits.
    The archive also holds what tscp's own components read: the outputs ``scale``
    (positive) and a band ``lower`` and ``upper`` one unit around the forecasts; two
    random context columns, ``load`` and ``temperature``; and timestamps, one hour apart
    from :data:`MONDAY`. Every built-in score and feature map can run on it, and with
    ``missing > 0`` so can the code paths for unobserved targets.

    Args:
        horizon: Number of steps ``H``.
        num_series: Number of series ``N``.
        num_origins: Number of forecast origins ``T``.
        missing: Fraction of targets to mark unobserved.
        seed: Seed of every random draw.
    """
    archive = variance_shift(
        horizon=horizon, num_origins=num_origins, num_series=num_series, seed=seed
    )
    generator = torch.Generator().manual_seed(seed + 1)
    outputs = {
        "scale": 0.5 + torch.rand(archive.residuals.shape, generator=generator),
        "lower": archive.forecasts - 1.0,
        "upper": archive.forecasts + 1.0,
    }
    archive = replace(
        archive,
        outputs=outputs,
        context=torch.randn(num_origins, 2, generator=generator),
        context_names=("load", "temperature"),
        timestamps=MONDAY + HOUR * archive.origins,
    )
    if missing:
        archive = with_missing_targets(archive, missing, seed=seed + 2)
    return archive


__all__ = ["HOUR", "MONDAY", "synthetic_archive", "with_missing_targets"]

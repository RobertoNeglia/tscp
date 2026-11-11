"""Synthetic residual archives.

Both generators simulate a Gaussian random walk forecast by its last value, so the
residual of (origin ``s``, step ``h``) is the sum of the next ``h`` innovations. Residuals
over overlapping targets are correlated, as real multi-step errors are.

Gaussian data are controls, for checking validity, not benchmarks.
"""

from collections.abc import Mapping
from dataclasses import replace

import torch
from torch import Tensor

from tscp.data import ResidualArchive, chronological_split

DEFAULT_SIZES = {"fit": 0.2, "calibration": 0.3, "test": None}
"""Default split sizes: 20% fit, 30% calibration, the rest test."""


def _random_walk_archive(
    scales: Tensor,
    *,
    horizon: int,
    num_series: int,
    sizes: Mapping[str, int | float | None] | None,
    seed: int,
    provenance: dict,
) -> ResidualArchive:
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    num_origins = scales.numel() - horizon
    generator = torch.Generator().manual_seed(seed)
    innovations = torch.randn(scales.numel(), num_series, generator=generator)
    path = torch.cumsum(innovations * scales[:, None], dim=0)
    steps = torch.arange(1, horizon + 1)
    targets = torch.stack([path[step : num_origins + step] for step in steps.tolist()], dim=1)
    forecasts = path[:num_origins, None, :].expand(-1, horizon, -1).clone()
    origins = torch.arange(num_origins)
    return ResidualArchive(
        residuals=targets - forecasts,
        origins=origins,
        steps=steps,
        splits=chronological_split(origins, steps, sizes or DEFAULT_SIZES),
        forecasts=forecasts,
        provenance={**provenance, "horizon": horizon, "seed": seed},
    )


def gaussian_control(
    *,
    horizon: int,
    num_origins: int = 2000,
    num_series: int = 1,
    scale: float = 1.0,
    sizes: Mapping[str, int | float | None] | None = None,
    seed: int = 0,
) -> ResidualArchive:
    """A random walk with i.i.d. Gaussian innovations: an exchangeable control.

    Args:
        horizon: Number of steps ``H``; take it from the experiment config.
        num_origins: Number of forecast origins ``T``.
        num_series: Number of independent series ``N``.
        scale: Innovation standard deviation.
        sizes: Split sizes for :func:`~tscp.data.chronological_split`;
            :data:`DEFAULT_SIZES` if None.
        seed: Seed of the generator.
    """
    scales = torch.full((num_origins + horizon,), float(scale))
    provenance = {"dataset": "gaussian_control", "role": "control", "scale": scale}
    return _random_walk_archive(
        scales,
        horizon=horizon,
        num_series=num_series,
        sizes=sizes,
        seed=seed,
        provenance={**provenance, "forecaster": "last value"},
    )


def variance_shift(
    *,
    horizon: int,
    num_origins: int = 2000,
    num_series: int = 1,
    shift_at: float = 0.75,
    scale_before: float = 1.0,
    scale_after: float = 3.0,
    sizes: Mapping[str, int | float | None] | None = None,
    seed: int = 0,
) -> ResidualArchive:
    """A Gaussian random walk whose innovation scale jumps once: not exchangeable.

    With the default splits the jump falls inside the test period, so methods that
    calibrate once under-cover after it, and adaptive ones should recover.

    Args:
        horizon: Number of steps ``H``; take it from the experiment config.
        num_origins: Number of forecast origins ``T``.
        num_series: Number of independent series ``N``.
        shift_at: Fraction of the timeline after which the scale changes.
        scale_before: Innovation standard deviation before the jump.
        scale_after: Innovation standard deviation after the jump.
        sizes: Split sizes; :data:`DEFAULT_SIZES` if None.
        seed: Seed of the generator.
    """
    if not 0 < shift_at < 1:
        raise ValueError("shift_at must lie in (0, 1)")
    scales = torch.full((num_origins + horizon,), float(scale_before))
    scales[int(shift_at * num_origins) :] = float(scale_after)
    provenance = {
        "dataset": "variance_shift",
        "role": "shift",
        "shift_at": shift_at,
        "scale_before": scale_before,
        "scale_after": scale_after,
        "forecaster": "last value",
    }
    return _random_walk_archive(
        scales,
        horizon=horizon,
        num_series=num_series,
        sizes=sizes,
        seed=seed,
        provenance=provenance,
    )


def context_scale(
    *,
    horizon: int,
    num_origins: int = 2000,
    num_series: int = 1,
    strength: float = 1.0,
    sizes: Mapping[str, int | float | None] | None = None,
    seed: int = 0,
) -> ResidualArchive:
    """A Gaussian random walk whose next innovation scale is set by a known covariate.

    Each origin ``t`` has a covariate ``x_t``, a standard normal draw stored as the
    context column ``x``, and the innovation that follows it has standard deviation
    ``exp(strength * x_t)``. The one-step residual of origin ``t`` therefore has a known
    conditional scale, and a score that learns the scale from the context gives narrower
    intervals than one that ignores it. Gaussian, so a control, not a benchmark.

    Args:
        horizon: Number of steps ``H``; take it from the experiment config.
        num_origins: Number of forecast origins ``T``.
        num_series: Number of independent series ``N``.
        strength: How strongly the covariate scales the innovations.
        sizes: Split sizes; :data:`DEFAULT_SIZES` if None.
        seed: Seed of the generator.
    """
    generator = torch.Generator().manual_seed(seed + 1)
    covariate = torch.randn(num_origins + horizon, generator=generator)
    scales = torch.ones(num_origins + horizon)
    scales[1:] = torch.exp(strength * covariate[:-1])  # x_t scales the innovation at t + 1
    provenance = {
        "dataset": "context_scale",
        "role": "control",
        "strength": strength,
        "forecaster": "last value",
    }
    archive = _random_walk_archive(
        scales,
        horizon=horizon,
        num_series=num_series,
        sizes=sizes,
        seed=seed,
        provenance=provenance,
    )
    return replace(archive, context=covariate[:num_origins, None], context_names=("x",))


__all__ = ["DEFAULT_SIZES", "context_scale", "gaussian_control", "variance_shift"]

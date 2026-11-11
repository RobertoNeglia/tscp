"""Choosing the narrowest of a grid of signed-residual intervals."""

import math

import torch
from torch import Tensor

from tscp.utils import Component


class WidthOptimal(Component):
    """Choose tail probabilities that minimize the width of a residual interval.

    Following Xu & Xie (TPAMI 2023, Algorithm 1; ICML 2023, Eq. 11), search ``bins``
    equally spaced values of ``beta`` in ``[0, alpha]`` and choose the first minimizer
    of ``Q(1 - alpha + beta) - Q(beta)``. This selector is separate from a conformal
    score distribution: width-optimal intervals need not be nested as alpha changes.

    Args:
        bins: Number of candidate tail allocations, including both endpoints.
        round_percentiles: Round probabilities upward to integer percentages, matching
            the journal EnbPI reference. SPCI uses unrounded forest quantiles.
    """

    def __init__(self, bins: int = 5, round_percentiles: bool = False):
        super().__init__()
        if isinstance(bins, bool) or not isinstance(bins, int) or bins < 2:
            raise ValueError("bins must be an integer of at least 2")
        self.bins = bins
        self.round_percentiles = bool(round_percentiles)

    def probabilities(self, alpha: float, *, device=None) -> Tensor:
        """Return ``[2, bins]`` lower/upper probabilities, in float64."""
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie in (0, 1)")
        # Match a float64 linspace, including its exact final endpoint. In particular,
        # percentage rounding deliberately has no epsilon that changes reference ranks.
        step = float(alpha) / (self.bins - 1)
        beta = [i * step for i in range(self.bins - 1)] + [float(alpha)]
        probabilities = [beta, [1 - float(alpha) + value for value in beta]]
        if self.round_percentiles:
            probabilities = [
                [math.ceil(100 * value) / 100 for value in side] for side in probabilities
            ]
        return torch.tensor(probabilities, dtype=torch.float64, device=device).clamp(0, 1)

    def select(self, quantiles: Tensor) -> Tensor:
        """Return the selected ``[2]`` bounds from lower/upper quantiles ``[2, bins]``."""
        if quantiles.shape != (2, self.bins):
            raise ValueError("quantiles must have shape [2, bins]")
        if not torch.isfinite(quantiles).all():
            raise ValueError("candidate quantiles must be finite")
        widths = quantiles[1] - quantiles[0]
        if (widths < 0).any():
            raise ValueError("upper quantiles must not be below lower quantiles")
        return quantiles[:, torch.argmin(widths)]

    def forward(self, residuals: Tensor, alpha: float) -> Tensor:
        """Return ``[lower, upper]`` using linearly interpolated empirical quantiles.

        An empty residual sample gives an unbounded interval. These empirical quantiles
        do not add the test point at infinity or use the split-conformal order statistic.
        """
        if residuals.ndim != 1 or not residuals.is_floating_point():
            raise ValueError("residuals must be a floating vector")
        probabilities = self.probabilities(alpha, device=residuals.device)
        if not torch.isfinite(residuals).all():
            raise ValueError("residuals must be finite")
        if not residuals.numel():
            return residuals.new_tensor([-torch.inf, torch.inf])
        values = torch.quantile(residuals.double(), probabilities.flatten(), interpolation="linear")
        return self.select(values.reshape(2, self.bins)).to(residuals.dtype)


__all__ = ["WidthOptimal"]

"""Conformalized quantile regression."""

import torch
from torch import Tensor

from tscp.data import ResidualArchive
from tscp.regions import Interval
from tscp.scores.base import Score


class CQRScore(Score):
    """Conformalized quantile regression (Romano, Patterson & Candès, NeurIPS 2019).

    The forecaster supplies a lower and an upper quantile of each target, stored as
    outputs in target units, as the forecaster produced them. In residual coordinates
    they are ``low = q_lower - f`` and ``high = q_upper - f`` for the point forecast
    ``f``, so the archive needs its ``forecasts``. The score ``max(low - r, r - high)`` is
    how far the target falls outside the forecaster's band (negative inside it), and a
    threshold ``q`` widens the band by ``q`` on both sides.

    With ``asymmetric=True`` the two sides are channels at ``alpha / 2`` each, scored
    ``low - r`` and ``r - high``, so each side gets its own correction (Romano et al.,
    Theorem 2).

    Args:
        lower: Name of the output with the lower quantiles, ``[T, H, N]``.
        upper: Name of the output with the upper quantiles, ``[T, H, N]``.
        asymmetric: Correct each side separately.
    """

    def __init__(self, lower: str = "lower", upper: str = "upper", asymmetric: bool = False):
        super().__init__()
        self.lower = lower
        self.upper = upper
        self.asymmetric = bool(asymmetric)
        self.channels = ("lower", "upper") if self.asymmetric else ()

    def _band(self, rows: ResidualArchive) -> tuple[Tensor, Tensor]:
        """The forecaster's band in residual coordinates: ``(low, high)``, each ``[T, H, N]``."""
        if rows.forecasts is None:
            raise ValueError("CQRScore needs the archive's forecasts")
        missing = [name for name in (self.lower, self.upper) if name not in rows.outputs]
        if missing:
            raise KeyError(f"the archive has no outputs {missing}")
        bounds = (rows.outputs[self.lower], rows.outputs[self.upper])
        if any(bound.shape != rows.residuals.shape for bound in bounds):
            raise ValueError("quantile outputs must have shape [T, H, N]")
        return bounds[0] - rows.forecasts, bounds[1] - rows.forecasts

    def forward(self, rows: ResidualArchive) -> Tensor:
        low, high = self._band(rows)
        below, above = low - rows.residuals, rows.residuals - high
        if self.asymmetric:
            return torch.stack([below, above], dim=-1)
        return torch.maximum(below, above)

    def region(self, threshold: Tensor, rows: ResidualArchive) -> Interval:
        low, high = self._band(rows)
        if self.asymmetric:
            return Interval(low - threshold[..., 0], high + threshold[..., 1])
        return Interval(low - threshold, high + threshold)

    def levels(self, alpha: float, kept: tuple[int, ...]) -> Tensor:
        if self.asymmetric:
            return torch.full(kept, float(alpha) / 2, dtype=torch.float64)
        return super().levels(alpha, kept)


__all__ = ["CQRScore"]

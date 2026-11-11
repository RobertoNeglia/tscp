"""Signed residuals: one score per side of the interval."""

import torch
from torch import Tensor

from tscp.data import ResidualArchive
from tscp.regions import Interval
from tscp.scores.base import Score


class SignedScore(Score):
    """Two one-sided scores per element, for intervals that need not be symmetric.

    Channel ``lower`` scores ``-r`` and channel ``upper`` scores ``r``; thresholds
    ``q_lower`` and ``q_upper`` give the interval ``[-q_lower, q_upper]``. The
    miscoverage is shared between the sides, ``lower_share * alpha`` below and the rest
    above, so by a union bound the interval misses with probability at most ``alpha``.
    Skewed errors get a narrower side where they are rare.

    Args:
        lower_share: Fraction of ``alpha`` spent on the lower side, in (0, 1).
    """

    channels = ("lower", "upper")

    def __init__(self, lower_share: float = 0.5):
        super().__init__()
        if not 0 < lower_share < 1:
            raise ValueError("lower_share must lie in (0, 1)")
        self.lower_share = float(lower_share)

    def forward(self, rows: ResidualArchive) -> Tensor:
        residuals = rows.residuals
        return torch.stack([-residuals, residuals], dim=-1)

    def region(self, threshold: Tensor, rows: ResidualArchive) -> Interval:
        return Interval(-threshold[..., 0], threshold[..., 1])

    def levels(self, alpha: float, kept: tuple[int, ...]) -> Tensor:
        shares = torch.tensor([self.lower_share, 1 - self.lower_share], dtype=torch.float64)
        return (float(alpha) * shares).expand(kept).clone()


__all__ = ["SignedScore"]

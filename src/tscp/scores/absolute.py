"""The absolute residual."""

from torch import Tensor

from tscp.data import ResidualArchive
from tscp.regions import Interval
from tscp.registry import register
from tscp.scores.base import Score


@register("score", "absolute")
class AbsoluteScore(Score):
    """``|r|``, per element; a threshold ``q`` gives the symmetric interval ``[-q, q]``."""

    def forward(self, rows: ResidualArchive) -> Tensor:
        return rows.residuals.abs()

    def region(self, threshold: Tensor, rows: ResidualArchive) -> Interval:
        return Interval(-threshold, threshold)


__all__ = ["AbsoluteScore"]

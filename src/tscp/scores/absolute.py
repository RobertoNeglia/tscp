"""The absolute residual."""

from collections.abc import Mapping

from torch import Tensor

from tscp.regions import Interval
from tscp.scores.base import Score


class AbsoluteScore(Score):
    """``|r|``, per element; a threshold ``q`` gives the symmetric interval ``[-q, q]``."""

    def forward(self, residuals: Tensor, outputs: Mapping[str, Tensor] | None = None) -> Tensor:
        return residuals.abs()

    def region(self, threshold: Tensor, outputs: Mapping[str, Tensor] | None = None) -> Interval:
        return Interval(-threshold, threshold)


__all__ = ["AbsoluteScore"]

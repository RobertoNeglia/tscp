"""The quantile-model interface (axis 4)."""

from torch import Tensor

from tscp.typing import Alpha
from tscp.utils import Component


class QuantileModel(Component):
    """How weighted scores become a threshold.

    :meth:`forward` takes pool scores ``[P, *kept]``, weights ``[Q, P]`` (or broadcastable
    to ``[Q, P, *kept]``) and a miscoverage level, and returns thresholds ``[Q, *kept]``.
    ``weights=None`` means every pool residual counts the same for every query; the result
    then has no query dimension and broadcasts against ``alpha`` instead.
    """

    def forward(self, scores: Tensor, weights: Tensor | None, alpha: Alpha) -> Tensor:
        """Return thresholds for the given pool scores, weights and level."""
        raise NotImplementedError


__all__ = ["QuantileModel"]

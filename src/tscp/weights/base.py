"""The calibration-weights interface (axis 3)."""

from torch import Tensor

from tscp.data import ResidualArchive
from tscp.utils import Component


class Weights(Component):
    """Which past errors count, and how much.

    :meth:`forward` returns nonnegative, unnormalized weights of shape ``[Q, P]``, or
    ``[Q, P, *kept]`` when they differ by step or series. The composer multiplies them by
    the availability and observed masks, so a weight class never handles either.
    """

    def forward(self, pool: ResidualArchive, query: ResidualArchive) -> Tensor:
        """Return weights of the ``P`` pool rows for each of the ``Q`` query rows."""
        raise NotImplementedError


__all__ = ["Weights"]

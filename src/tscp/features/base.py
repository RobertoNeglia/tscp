"""The feature-map interface."""

from torch import Tensor

from tscp.data import ResidualArchive
from tscp.utils import Component


class FeatureMap(Component):
    """Turns what is known at each origin into a vector.

    Weights compare these vectors, quantile models regress on them, and Mondrian
    weights group by them. A feature map runs over the whole timeline causally: the
    features of origin ``t`` use only what is known at ``t`` (see
    :mod:`tscp.data.availability`), and are subset by split afterwards.
    """

    def forward(self, archive: ResidualArchive) -> Tensor:
        """Return ``[T, F]`` or ``[T, N, F]`` features, one row per origin."""
        raise NotImplementedError


__all__ = ["FeatureMap"]

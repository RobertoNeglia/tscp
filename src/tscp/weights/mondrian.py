"""Calibration within discrete groups known at the forecast origin."""

import torch
from torch import Tensor

from tscp.data import ResidualArchive
from tscp.features import FeatureMap
from tscp.registry import register
from tscp.weights.base import Weights


@register("weights", "mondrian")
class MondrianWeights(Weights):
    """Equal weight within the query's category, zero outside it.

    Categories are tuples of finite integer-valued origin features ``[T, F]``.
    There is no implicit binning or fallback to other categories. Under exchangeability
    within each fixed category this gives category-conditional split coverage (Vovk,
    2012, https://proceedings.mlr.press/v25/vovk12.html).

    Args:
        groups: A feature map, such as integer Calendar, Phase or ContextColumns.
    """

    def __init__(self, groups: FeatureMap):
        super().__init__()
        if not isinstance(groups, FeatureMap):
            raise TypeError("groups must be a FeatureMap")
        self.groups = groups

    @property
    def is_trainable(self) -> bool:
        return self.groups.is_trainable

    def fit(self, train, validation=None):
        if self.groups.is_trainable:
            self.groups.fit(train, validation)
        return self

    def _groups(self, rows: ResidualArchive) -> Tensor:
        values = self.groups(rows)
        if values.ndim != 2 or values.shape[0] != len(rows) or values.shape[1] == 0:
            raise ValueError("Mondrian groups must have shape [T, F] with F positive")
        if not torch.isfinite(values).all() or (
            values.is_floating_point() and not (values == values.round()).all()
        ):
            raise ValueError("Mondrian groups must be finite discrete integer values")
        valid = self.groups.valid(rows)
        if valid is not None and not bool(valid.all()):
            raise ValueError("Mondrian group features must be defined for every origin")
        return values

    def forward(self, pool: ResidualArchive, query: ResidualArchive) -> Tensor:
        groups, requested = self._groups(pool), self._groups(query)
        return (requested[:, None, :] == groups[None, :, :]).all(dim=-1).to(torch.float64)


__all__ = ["MondrianWeights"]

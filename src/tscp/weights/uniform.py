"""Equal weights."""

import torch
from torch import Tensor

from tscp.data import ResidualArchive
from tscp.registry import register
from tscp.weights.base import Weights


@register("weights", "uniform")
class UniformWeights(Weights):
    """Every pool residual counts the same.

    Passing ``weights=None`` to a predictor gives the same thresholds through a faster
    path; this class exists for ablations that swap weight classes.
    """

    def forward(self, pool: ResidualArchive, query: ResidualArchive) -> Tensor:
        return torch.ones(len(query), len(pool), dtype=torch.float64, device=pool.device)


__all__ = ["UniformWeights"]

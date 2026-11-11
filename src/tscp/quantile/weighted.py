"""The conformal quantile of (weighted) scores."""

import math

import torch
from torch import Tensor

from tscp.data import ResidualArchive
from tscp.quantile.base import Pool, QuantileModel
from tscp.quantile.distribution import EmpiricalDistribution
from tscp.typing import Alpha

_MAX_ELEMENTS = 2**23


def uniform_quantile(scores: Tensor, alpha: Alpha, test_weight: float = 1.0) -> Tensor:
    """Conformal quantile of equally weighted scores.

    The threshold is the ``ceil((1 - alpha) * (n + w))``-th smallest of the ``n`` pool
    scores, where ``w`` is the test point's weight (1 for split conformal); ``+inf`` when
    that rank exceeds ``n`` and ``-inf`` when ``1 - alpha <= 0``.

    Args:
        scores: ``[P, *kept]`` pool scores.
        alpha: Miscoverage level; a float, or a tensor that broadcasts against ``kept``
            from the right (for example ``[*kept]`` or ``[Q, *kept]``).
        test_weight: Weight of the test point, relative to 1 per pool score.

    Returns:
        Thresholds of shape ``broadcast(alpha.shape, kept)``, in the dtype of ``scores``.
    """
    ordered = torch.sort(scores, dim=0).values
    return EmpiricalDistribution(ordered, None, test_weight).threshold(alpha)


def weighted_quantile(
    scores: Tensor, weights: Tensor, alpha: Alpha, test_weight: float = 1.0
) -> Tensor:
    """Conformal quantile of weighted scores.

    For each query the threshold is the smallest pool score ``s`` with
    ``sum(w_i for s_i <= s) / (sum(w_i) + test_weight) >= 1 - alpha``: the
    ``(1 - alpha)``-quantile of the weighted pool with the test point's weight at
    ``+inf`` (Tibshirani et al., NeurIPS 2019; Barber et al., Annals of Statistics 2023).
    It is ``+inf`` when the pool never reaches ``1 - alpha`` and ``-inf`` when
    ``1 - alpha <= 0``. With all weights 1 it equals :func:`uniform_quantile`.

    Args:
        scores: ``[P, *kept]`` pool scores.
        weights: Nonnegative weights ``[Q, P]``, or broadcastable to ``[Q, P, *kept]``.
        alpha: Miscoverage level; a float, or a tensor broadcastable to ``[Q, *kept]``.
        test_weight: Weight of the test point.

    Returns:
        Thresholds ``[Q, *kept]``, in the dtype of ``scores``.
    """
    num_scores, kept = scores.shape[0], tuple(scores.shape[1:])
    if weights.ndim < 2 or weights.shape[1] != num_scores:
        raise ValueError("weights must have shape [Q, P] or [Q, P, *kept] with P pool scores")
    num_queries = weights.shape[0]
    alpha = torch.as_tensor(alpha, dtype=torch.float64, device=scores.device)
    alpha = alpha.broadcast_to((num_queries, *kept))
    ordered, order = torch.sort(scores, dim=0)
    out = torch.empty((num_queries, *kept), dtype=scores.dtype, device=scores.device)
    chunk = max(1, _MAX_ELEMENTS // max(1, num_scores * math.prod(kept)))
    for start in range(0, num_queries, chunk):
        stop = min(start + chunk, num_queries)
        part = EmpiricalDistribution.weighted(ordered, order, weights[start:stop], test_weight)
        out[start:stop] = part.threshold(alpha[start:stop])
    return out


class WeightedQuantile(QuantileModel):
    """Empirical conformal quantile of (weighted) scores, with the test point at ``+inf``.

    Returns an :class:`~tscp.quantile.EmpiricalDistribution`: one shared by every query
    when ``weights`` is None, one per query otherwise. Either way the pool is sorted once.

    Args:
        test_weight: Weight of the test point relative to the pool weights; 1 for split
            conformal and NexCP.
    """

    def __init__(self, test_weight: float = 1.0):
        super().__init__()
        if not test_weight >= 0:
            raise ValueError("test_weight must be nonnegative")
        self.test_weight = float(test_weight)

    def forward(
        self, pool: Pool, weights: Tensor | None, query: ResidualArchive
    ) -> EmpiricalDistribution:
        ordered, order = pool.sorted()
        if weights is None:
            return EmpiricalDistribution(ordered, None, self.test_weight)
        if (weights < 0).any() or weights.isnan().any():
            raise ValueError("weights must be nonnegative numbers")
        return EmpiricalDistribution.weighted(ordered, order, weights, self.test_weight)


__all__ = ["WeightedQuantile", "uniform_quantile", "weighted_quantile"]

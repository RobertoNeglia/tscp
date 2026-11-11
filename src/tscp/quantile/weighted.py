"""The conformal quantile of (weighted) scores."""

import math

import torch
from torch import Tensor

from tscp.quantile.base import QuantileModel
from tscp.typing import Alpha

# Levels within this tolerance of a cumulative weight count as reached, so float rounding
# in 1 - alpha or in the cumulative sums cannot shift the threshold by one score.
_TOLERANCE = 1e-9
_MAX_ELEMENTS = 2**23


def _levels(alpha: Alpha, device: torch.device) -> Tensor:
    return 1 - torch.as_tensor(alpha, dtype=torch.float64, device=device)


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
    num_scores = scores.shape[0]
    kept = tuple(scores.shape[1:])
    level = _levels(alpha, scores.device)
    shape = torch.broadcast_shapes(level.shape, kept)
    level = level.broadcast_to(shape)
    ordered = torch.sort(scores, dim=0).values
    rank = torch.ceil((level - _TOLERANCE) * (num_scores + test_weight))
    index = (rank - 1).clamp(0, num_scores - 1).to(torch.int64).reshape(-1, *kept)
    source = ordered.unsqueeze(0).expand(index.shape[0], num_scores, *kept)
    values = torch.gather(source, 1, index.unsqueeze(1)).squeeze(1).reshape(shape)
    values = torch.where(rank > num_scores, torch.inf, values.to(torch.float64))
    values = torch.where(level <= 0, -torch.inf, values)
    return values.to(scores.dtype)


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
    num_scores = scores.shape[0]
    kept = tuple(scores.shape[1:])
    if weights.ndim < 2 or weights.shape[1] != num_scores:
        raise ValueError("weights must have shape [Q, P] or [Q, P, *kept] with P pool scores")
    weights = weights.reshape(*weights.shape, *([1] * (2 + len(kept) - weights.ndim)))
    num_queries = weights.shape[0]
    level = _levels(alpha, scores.device).broadcast_to((num_queries, *kept))
    ordered, order = torch.sort(scores, dim=0)
    ordered = ordered.to(torch.float64)
    out = torch.empty((num_queries, *kept), dtype=torch.float64, device=scores.device)
    chunk = max(1, _MAX_ELEMENTS // max(1, num_scores * math.prod(kept)))
    for start in range(0, num_queries, chunk):
        stop = min(start + chunk, num_queries)
        full = (stop - start, num_scores, *kept)
        part = weights[start:stop].to(torch.float64).broadcast_to(full)
        part = torch.gather(part, 1, order.unsqueeze(0).expand(full))
        total = part.sum(dim=1, keepdim=True) + test_weight
        cumulative = torch.cumsum(part, dim=1) / total
        part_level = level[start:stop]
        reached = cumulative >= (part_level - _TOLERANCE).unsqueeze(1)
        # Cumulative weights never decrease, so the first reached index is the count of
        # unreached ones; it equals P when the level is never reached.
        first = (~reached).sum(dim=1)
        values = torch.gather(
            ordered.unsqueeze(0).expand(full), 1, first.clamp(max=num_scores - 1).unsqueeze(1)
        ).squeeze(1)
        values = torch.where(first >= num_scores, torch.inf, values)
        out[start:stop] = torch.where(part_level <= 0, -torch.inf, values)
    return out.to(scores.dtype)


class WeightedQuantile(QuantileModel):
    """Empirical conformal quantile of (weighted) scores, with the test point at ``+inf``.

    With ``weights=None`` it takes the uniform path (:func:`uniform_quantile`), which
    sorts the pool once and serves every query; otherwise :func:`weighted_quantile`.

    Args:
        test_weight: Weight of the test point relative to the pool weights; 1 for split
            conformal and NexCP.
    """

    def __init__(self, test_weight: float = 1.0):
        super().__init__()
        if not test_weight >= 0:
            raise ValueError("test_weight must be nonnegative")
        self.test_weight = float(test_weight)

    def forward(self, scores: Tensor, weights: Tensor | None, alpha: Alpha) -> Tensor:
        if weights is None:
            return uniform_quantile(scores, alpha, self.test_weight)
        if (weights < 0).any() or weights.isnan().any():
            raise ValueError("weights must be nonnegative numbers")
        return weighted_quantile(scores, weights, alpha, self.test_weight)


__all__ = ["WeightedQuantile", "uniform_quantile", "weighted_quantile"]

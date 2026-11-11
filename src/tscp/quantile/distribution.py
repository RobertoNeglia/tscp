"""Conformal distributions of scores: a threshold at any level, and the level of any score."""

import torch
from torch import Tensor

from tscp.typing import Alpha

TOLERANCE = 1e-9
"""Levels within this of a cumulative weight count as reached, so float rounding in
``1 - alpha`` or in cumulative sums cannot shift a threshold by one score."""


class ScoreDistribution:
    """The conformal distribution of a score, for a batch of queries and kept elements.

    Quantile models (axis 4) return one. :meth:`threshold` gives the threshold at any
    miscoverage level, several levels at once if asked; :meth:`level` gives the largest
    level at which a score is still covered, its conformal p-value. Online updates use
    both: the ACI family moves the level it asks a threshold for, and DtACI weighs its
    experts by the level of each realized score.

    The batch shape is ``[Q, *kept]`` for one distribution per query, or ``[*kept]`` when
    every query shares one.
    """

    @property
    def batch_shape(self) -> torch.Size:
        raise NotImplementedError

    def threshold(self, alpha: Alpha) -> Tensor:
        """Return the threshold at miscoverage ``alpha``.

        ``alpha`` broadcasts against :attr:`batch_shape` from the right and may add leading
        dimensions to ask for several levels at once: ``alpha`` of shape ``[L, *batch]``
        gives thresholds ``[L, *batch]``. A level at or below 0 gives ``+inf`` (an
        unbounded region), and one at or above 1 gives ``-inf`` (an empty region).
        """
        raise NotImplementedError

    def level(self, scores: Tensor) -> Tensor:
        """Return the largest miscoverage level whose threshold still covers ``scores``.

        ``scores`` broadcast against :attr:`batch_shape`. The result, in float64, is the
        conformal p-value of each score (up to the rounding tolerance of
        :meth:`threshold`): 1 below every pool score, and the test point's share of the
        weight above them all.
        """
        raise NotImplementedError

    def select(self, index: int) -> "ScoreDistribution":
        """Return the distribution of one query, of batch shape ``[*kept]``."""
        raise NotImplementedError


class EmpiricalDistribution(ScoreDistribution):
    """The weighted empirical distribution of pool scores, with the test point at ``+inf``.

    With weights ``w_i`` on pool scores ``s_i`` and the test point's weight ``w``, the
    threshold at ``alpha`` is the smallest pool score ``s`` with
    ``sum(w_i for s_i <= s) / (sum(w_i) + w) >= 1 - alpha`` (Tibshirani et al., NeurIPS
    2019; Barber et al., Annals of Statistics 2023), or ``+inf`` when the pool never
    reaches ``1 - alpha``. With equal weights it is the ``ceil((1 - alpha)(n + w))``-th
    smallest of the ``n`` pool scores.

    Args:
        ordered: ``[P, *kept]`` pool scores, sorted along the pool dimension.
        cumulative: None for equal weights. Otherwise the cumulative weights of the sorted
            scores divided by the total weight plus ``test_weight``: ``[Q, P, *kept]``, or
            ``[P, *kept]`` for one query.
        test_weight: Weight of the test point; with equal weights, relative to 1 per score.
    """

    def __init__(self, ordered: Tensor, cumulative: Tensor | None = None, test_weight=1.0):
        self.ordered = ordered
        self.cumulative = cumulative
        self.test_weight = float(test_weight)

    @classmethod
    def weighted(
        cls, ordered: Tensor, order: Tensor, weights: Tensor, test_weight: float = 1.0
    ) -> "EmpiricalDistribution":
        """Build a distribution per query from sorted scores and unsorted weights.

        Args:
            ordered: ``[P, *kept]`` pool scores, sorted along the pool dimension.
            order: ``[P, *kept]`` indices that sort the scores, as ``torch.sort`` returns.
            weights: Nonnegative weights ``[Q, P]``, or broadcastable to ``[Q, P, *kept]``.
            test_weight: Weight of the test point.
        """
        num_scores, kept = ordered.shape[0], tuple(ordered.shape[1:])
        if weights.ndim < 2 or weights.shape[1] != num_scores:
            raise ValueError("weights must have shape [Q, P] or [Q, P, *kept] with P pool scores")
        weights = weights.reshape(*weights.shape, *([1] * (2 + len(kept) - weights.ndim)))
        full = (weights.shape[0], num_scores, *kept)
        part = weights.to(torch.float64).broadcast_to(full)
        part = torch.gather(part, 1, order.unsqueeze(0).expand(full))
        total = part.sum(dim=1, keepdim=True) + test_weight
        return cls(ordered, torch.cumsum(part, dim=1) / total, test_weight)

    @property
    def _pool_dim(self) -> int:
        """Position of the pool dimension, counted from the right."""
        return -self.ordered.ndim

    @property
    def batch_shape(self) -> torch.Size:
        if self.cumulative is None:
            return self.ordered.shape[1:]
        shape = list(self.cumulative.shape)
        del shape[self._pool_dim]
        return torch.Size(shape)

    def threshold(self, alpha: Alpha) -> Tensor:
        level = 1 - torch.as_tensor(alpha, dtype=torch.float64, device=self.ordered.device)
        if self.cumulative is None:
            return self._uniform_threshold(level)
        shape = torch.broadcast_shapes(level.shape, self.batch_shape)
        level = level.broadcast_to(shape)
        pool_dim, num_scores = self._pool_dim, self.ordered.shape[0]
        # Cumulative weights never decrease, so the first index that reaches the level is
        # the number of indices that do not; it equals P when the level is never reached.
        unreached = self.cumulative < (level - TOLERANCE).unsqueeze(pool_dim)
        first = unreached.sum(dim=pool_dim)
        index = first.clamp(max=num_scores - 1).unsqueeze(pool_dim)
        values = torch.gather(self.ordered.expand(unreached.shape), pool_dim, index)
        values = torch.where(first >= num_scores, torch.inf, values.squeeze(pool_dim).double())
        return _extremes(values, level).to(self.ordered.dtype)

    def _uniform_threshold(self, level: Tensor) -> Tensor:
        num_scores, kept = self.ordered.shape[0], tuple(self.ordered.shape[1:])
        shape = torch.broadcast_shapes(level.shape, kept)
        level = level.broadcast_to(shape)
        rank = torch.ceil((level - TOLERANCE) * (num_scores + self.test_weight))
        index = (rank - 1).clamp(0, num_scores - 1).to(torch.int64).reshape(-1, *kept)
        source = self.ordered.unsqueeze(0).expand(index.shape[0], num_scores, *kept)
        values = torch.gather(source, 1, index.unsqueeze(1)).squeeze(1).reshape(shape)
        values = torch.where(rank > num_scores, torch.inf, values.to(torch.float64))
        return _extremes(values, level).to(self.ordered.dtype)

    def level(self, scores: Tensor) -> Tensor:
        scores = torch.as_tensor(scores, device=self.ordered.device)
        pool_dim, num_scores = self._pool_dim, self.ordered.shape[0]
        shape = torch.broadcast_shapes(scores.shape, self.batch_shape)
        scores = scores.broadcast_to(shape)
        below = (self.ordered < scores.unsqueeze(pool_dim)).sum(dim=pool_dim)
        if self.cumulative is None:
            return 1 - below.to(torch.float64) / (num_scores + self.test_weight)
        full = list(shape)
        full.insert(len(full) + pool_dim + 1, num_scores)
        index = (below - 1).clamp(min=0).unsqueeze(pool_dim)
        mass = torch.gather(self.cumulative.expand(full), pool_dim, index).squeeze(pool_dim)
        return 1 - torch.where(below > 0, mass, 0.0)

    def select(self, index: int) -> "EmpiricalDistribution":
        if self.cumulative is None or self.cumulative.ndim == self.ordered.ndim:
            return self
        return EmpiricalDistribution(self.ordered, self.cumulative[index], self.test_weight)


def _extremes(values: Tensor, level: Tensor) -> Tensor:
    """Apply the conventions at the ends: ``alpha <= 0`` covers all, ``alpha >= 1`` nothing.

    Without them, a test weight of 0 would give the largest pool score at ``alpha = 0``,
    and the long-run guarantee of the ACI family relies on an unbounded region there.
    """
    values = torch.where(level >= 1, torch.inf, values)
    return torch.where(level <= 0, -torch.inf, values)


__all__ = ["TOLERANCE", "EmpiricalDistribution", "ScoreDistribution"]

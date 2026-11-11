"""The quantile-model interface (axis 4)."""

import torch
from torch import Tensor

from tscp.data import ResidualArchive
from tscp.quantile.distribution import ScoreDistribution
from tscp.utils import Component


class Pool:
    """The scores a predictor calibrates with, with the rows they come from.

    The composer builds one per call to ``predict``. It sorts the scores once, however
    many queries read them.

    Args:
        scores: ``[P, *kept]`` pool scores.
        rows: The ``P`` pool rows, a row subset of the archive, for quantile models and
            weights that use features of the rows.
        valid: ``[P, *kept]`` bool, False where a score is undefined; None if all are.
    """

    def __init__(self, scores: Tensor, rows: ResidualArchive, valid: Tensor | None = None):
        if scores.shape[0] != len(rows):
            raise ValueError("the pool needs one row per score")
        if valid is not None and valid.shape != scores.shape:
            raise ValueError("valid must have the shape of the scores")
        self.scores = scores
        self.rows = rows
        self.valid = valid
        self._sorted: tuple[Tensor, Tensor] | None = None

    def __len__(self) -> int:
        return self.scores.shape[0]

    @property
    def kept(self) -> tuple[int, ...]:
        """Sizes of the kept dimensions."""
        return tuple(self.scores.shape[1:])

    def sorted(self) -> tuple[Tensor, Tensor]:
        """The scores sorted along the pool dimension, and the indices that sort them."""
        if self._sorted is None:
            values, indices = torch.sort(self.scores, dim=0)
            self._sorted = (values, indices)
        return self._sorted


class QuantileModel(Component):
    """How weighted scores become a threshold.

    :meth:`forward` takes the calibration pool, the weights of its scores for each query
    and the query rows, and returns a :class:`~tscp.quantile.ScoreDistribution` from which
    the composer and online updates read thresholds at any level.

    ``weights=None`` means every pool score counts the same for every query; the composer
    passes None only when every pool score is defined and available. Otherwise weights
    are ``[Q, P]`` or broadcastable to ``[Q, P, *kept]``, nonnegative, and already zero for
    scores that are undefined or not yet known at the query.

    A model that regresses scores on features of the rows, such as quantile regression on
    recent residuals, reads them from ``pool.rows`` and ``query`` and their timelines.
    """

    def forward(
        self, pool: Pool, weights: Tensor | None, query: ResidualArchive
    ) -> ScoreDistribution:
        """Return the distribution of the score at each of the ``Q`` query rows."""
        raise NotImplementedError


__all__ = ["Pool", "QuantileModel"]

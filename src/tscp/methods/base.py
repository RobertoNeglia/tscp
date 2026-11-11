"""The composer that runs one component per axis as a conformal method."""

import math
from collections.abc import Iterator
from typing import Self

import torch
from torch import Tensor

from tscp.data import ResidualArchive, availability_mask
from tscp.methods.engine import OnlineEngine, replay
from tscp.online import OnlineUpdate
from tscp.quantile import Pool, QuantileModel, ScoreDistribution, WeightedQuantile
from tscp.regions import Region
from tscp.scores import Score
from tscp.sources import HoldoutSource, Source
from tscp.utils import Component
from tscp.weights import Weights

AXES = ("score", "source", "weights", "quantile", "update")
"""Attribute names of the axis components."""

_AXIS_TYPES = {
    "score": Score,
    "source": Source,
    "weights": Weights,
    "quantile": QuantileModel,
    "update": OnlineUpdate,
}
_MAX_ELEMENTS = 2**23


class ConformalPredictor(Component):
    """A conformal method: one component per axis, run through a fixed lifecycle.

    1. :meth:`fit` fits every trainable component on ``archive.split("fit")``, with the
       ``validation`` split if there is one, and on nothing else.
    2. :meth:`calibrate` scores the source's rows, the calibration split.
    3. :meth:`predict` makes one region per test origin, in residual coordinates.

    Evaluation is a separate step (:func:`tscp.metrics.evaluate`).

    For each chunk of test origins, :meth:`predict` asks the quantile model for the
    conformal distribution of the score given the pool and its weights. Without an online
    update it reads every threshold from those distributions at the score's levels. With
    one, it answers the test origins in time order and hands the update each outcome once
    its target time has passed (:mod:`tscp.methods.engine`). An expanding source adds each
    test residual to the pool by the same rule. For a per-step score, step ``h`` of an
    origin becomes known ``h`` base steps after it; for a score over the whole horizon,
    after the last step.

    Args:
        score: Axis 2, how an error becomes a number.
        source: Axis 1, which past errors calibrate; the calibration split by default.
        weights: Axis 3; None weighs every available pool residual equally.
        quantile: Axis 4; :class:`~tscp.quantile.WeightedQuantile` by default.
        update: Axis 5; None for an offline method.
        alpha: Target miscoverage level, in (0, 1).
        chunk_size: Most query origins whose weights are held in memory at once.
    """

    def __init__(
        self,
        score: Score,
        source: Source | None = None,
        weights: Weights | None = None,
        quantile: QuantileModel | None = None,
        update: OnlineUpdate | None = None,
        alpha: float = 0.1,
        chunk_size: int = 256,
    ):
        super().__init__()
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie in (0, 1)")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self.score = score
        self.source = source if source is not None else HoldoutSource()
        self.weights = weights
        self.quantile = quantile if quantile is not None else WeightedQuantile()
        self.update = update
        for axis, expected in _AXIS_TYPES.items():
            component = getattr(self, axis)
            if component is not None and not isinstance(component, expected):
                raise TypeError(f"{axis} must be a {expected.__name__}, not {type(component)}")
        self.alpha = float(alpha)
        self.chunk_size = int(chunk_size)
        self.fit_fingerprint: str | None = None
        # Calibration state: moves with .to(device), stays out of state_dict().
        self.register_buffer("pool_scores", None, persistent=False)
        self.register_buffer("pool_origins", None, persistent=False)
        self.register_buffer("pool_valid", None, persistent=False)

    def components(self) -> dict[str, Component]:
        """Return the components that are set, keyed by axis."""
        return {axis: getattr(self, axis) for axis in AXES if getattr(self, axis) is not None}

    @property
    def is_trainable(self) -> bool:
        return any(component.is_trainable for component in self.components().values())

    # -- lifecycle -----------------------------------------------------------------------

    def fit(self, archive: ResidualArchive) -> Self:
        """Fit every trainable component on the ``fit`` split, and ``validation`` if any."""
        trainable = [axis for axis, c in self.components().items() if c.is_trainable]
        if not trainable:
            return self
        if archive.splits is None or "fit" not in archive.splits:
            names = ", ".join(trainable)
            raise ValueError(f"{names} must be fitted, and the archive has no 'fit' split")
        train = archive.split("fit")
        validation = archive.split("validation") if "validation" in archive.splits else None
        for axis in trainable:
            getattr(self, axis).fit(train, validation)
        # Which data the fitted state came from, for saved methods (tscp.methods.io).
        self.fit_fingerprint = archive.fingerprint()
        return self

    def calibrate(self, archive: ResidualArchive) -> Self:
        """Score the calibration pool of ``archive``."""
        pool = archive.select(self.source.rows(archive))
        self.pool_scores = self.score(pool)
        self.pool_origins = pool.origins
        self.pool_valid = self.score.valid(pool)
        return self

    def predict(self, archive: ResidualArchive) -> Region:
        """Return one region per origin of the ``test`` split, in residual coordinates."""
        if self.pool_scores is None:
            raise RuntimeError("calibrate() must run before predict()")
        rows = self.source.rows(archive)
        if not torch.equal(archive.origins[rows], self.pool_origins):
            raise ValueError("predict() must get the archive that calibrate() got")
        if archive.splits is None or "test" not in archive.splits:
            raise ValueError("predict() needs an archive with a 'test' split")
        test_rows = archive.splits["test"]
        query = archive.select(test_rows)
        query_scores = self.score(query)
        query_valid = self.score.valid(query)
        if self.source.expanding:
            # Test residuals join the pool; the availability mask releases each one only
            # after its target time.
            pool = Pool(
                torch.cat([self.pool_scores, query_scores]),
                archive.select(torch.cat([rows, test_rows])),
                _concat_valid(self.pool_valid, query_valid, self.pool_scores, query_scores),
            )
        else:
            pool = Pool(self.pool_scores, archive.select(rows), self.pool_valid)
        thresholds = self._thresholds(pool, query, query_scores, query_valid)
        return self.score.region(thresholds, query)

    # -- internals -----------------------------------------------------------------------

    def _thresholds(
        self,
        pool: Pool,
        query: ResidualArchive,
        query_scores: Tensor,
        query_valid: Tensor | None,
    ) -> Tensor:
        kept = pool.kept
        levels = self.score.levels(self.alpha, kept).to(pool.scores.device)
        step_kept = "step" not in self.score.reduce_dims
        # Steps after which each group of kept elements becomes known.
        offsets = query.steps if step_kept else query.steps[-1:]
        out = torch.empty((len(query), *kept), dtype=pool.scores.dtype, device=pool.scores.device)
        chunks = self._distributions(pool, query, offsets, step_kept)
        if self.update is None:
            for start, stop, distribution in chunks:
                out[start:stop] = distribution.threshold(levels)
            return out
        engine = OnlineEngine(self.update, levels, step_kept, len(offsets))
        return replay(engine, query.origins, offsets, chunks, query_scores, query_valid, out)

    def _distributions(
        self, pool: Pool, query: ResidualArchive, offsets: Tensor, step_kept: bool
    ) -> Iterator[tuple[int, int, ScoreDistribution]]:
        """Yield ``(start, stop, distribution)`` for consecutive chunks of the queries."""
        uniform = self.weights is None and not self.source.expanding and pool.valid is None
        chunk = self.chunk_size
        if not uniform:
            chunk = max(1, min(chunk, _MAX_ELEMENTS // (len(pool) * math.prod(pool.kept))))
        for start in range(0, len(query), chunk):
            stop = min(start + chunk, len(query))
            rows = query.select(torch.arange(start, stop, device=query.device))
            weights = None if uniform else self._weights(pool, rows, offsets, step_kept)
            yield start, stop, self.quantile(pool, weights, rows)

    def _weights(
        self, pool: Pool, rows: ResidualArchive, offsets: Tensor, step_kept: bool
    ) -> Tensor:
        """Weights ``[c, P, ...]`` for the ``c`` query ``rows``, with every mask applied."""
        kept_ndim = len(pool.kept)
        device = pool.scores.device
        weights = torch.ones(len(rows), len(pool), dtype=torch.float64, device=device)
        if self.weights is not None:
            weights = self.weights(pool.rows, rows).to(torch.float64)
        weights = weights.reshape(*weights.shape, *([1] * (2 + kept_ndim - weights.ndim)))
        if self.source.expanding:
            mask = availability_mask(pool.rows.origins, rows.origins, offsets)
            weights = weights * _group_mask(mask, step_kept, kept_ndim)
        if pool.valid is not None:
            weights = weights * pool.valid.unsqueeze(0)
        return weights


def _group_mask(mask: Tensor, step_kept: bool, kept_ndim: int) -> Tensor:
    """Reshape a ``[Q, P, G]`` availability mask to broadcast over ``[Q, P, *kept]``."""
    if step_kept:  # one group per step, and step is the first kept dimension
        return mask.reshape(*mask.shape, *([1] * (kept_ndim - 1)))
    return mask[..., 0].reshape(*mask.shape[:2], *([1] * kept_ndim))


def _concat_valid(
    pool_valid: Tensor | None, query_valid: Tensor | None, pool_scores: Tensor, query_scores: Tensor
) -> Tensor | None:
    if pool_valid is None and query_valid is None:
        return None
    if pool_valid is None:
        pool_valid = torch.ones_like(pool_scores, dtype=torch.bool)
    if query_valid is None:
        query_valid = torch.ones_like(query_scores, dtype=torch.bool)
    return torch.cat([pool_valid, query_valid])


__all__ = ["AXES", "ConformalPredictor"]

"""The composer that runs one component per axis as a conformal method."""

import math
from functools import partial
from typing import Self

import torch
from torch import Tensor

from tscp.data import ResidualArchive, availability_mask
from tscp.online import OnlineUpdate
from tscp.quantile import QuantileModel, WeightedQuantile
from tscp.regions import Region
from tscp.scores import Score
from tscp.sources import HoldoutSource, Source
from tscp.typing import Alpha
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

    1. :meth:`fit` fits every trainable component on ``archive.split("fit")``, and on
       nothing else.
    2. :meth:`calibrate` scores the source's rows, the calibration split.
    3. :meth:`predict` makes one region per test origin, in residual coordinates.

    Evaluation is a separate step (:func:`tscp.metrics.evaluate`).

    :meth:`predict` is vectorized unless an online update is set; then it walks the test
    origins in order and hands the update each outcome once its target time has passed.
    An expanding source adds each test residual to the pool by the same rule. For a
    per-step score, step ``h`` of an origin becomes known ``h`` base steps after it; for
    a score over the whole horizon, after the last step.

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
        """Fit every trainable component on the ``fit`` split only."""
        trainable = [axis for axis, c in self.components().items() if c.is_trainable]
        if not trainable:
            return self
        if archive.splits is None or "fit" not in archive.splits:
            names = ", ".join(trainable)
            raise ValueError(f"{names} must be fitted, and the archive has no 'fit' split")
        fit_rows = archive.split("fit")
        for axis in trainable:
            getattr(self, axis).fit(fit_rows)
        return self

    def calibrate(self, archive: ResidualArchive) -> Self:
        """Score the calibration pool of ``archive``."""
        pool = archive.select(self.source.rows(archive))
        self.pool_scores = self.score(pool.residuals, pool.outputs)
        self.pool_origins = pool.origins
        self.pool_valid = self._valid(pool)
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
        query_scores = self.score(query.residuals, query.outputs)
        query_valid = self._valid(query)
        if self.source.expanding:
            # Test residuals join the pool; the availability mask releases each one only
            # after its target time.
            pool = archive.select(torch.cat([rows, test_rows]))
            scores = torch.cat([self.pool_scores, query_scores])
            valid = _concat_valid(self.pool_valid, query_valid, self.pool_scores, query_scores)
        else:
            pool, scores, valid = archive.select(rows), self.pool_scores, self.pool_valid
        thresholds = self._thresholds(pool, scores, valid, query, query_scores, query_valid)
        return self.score.region(thresholds, query.outputs)

    # -- internals -----------------------------------------------------------------------

    def _valid(self, archive: ResidualArchive) -> Tensor | None:
        """Return which kept elements have an observed target, or None if all do."""
        if archive.observed is None or bool(archive.observed.all()):
            return None
        if self.score.reduce_dims:
            raise NotImplementedError(
                "scores that reduce steps or series do not handle missing targets yet"
            )
        return archive.observed

    def _thresholds(
        self,
        pool: ResidualArchive,
        scores: Tensor,
        valid: Tensor | None,
        query: ResidualArchive,
        query_scores: Tensor,
        query_valid: Tensor | None,
    ) -> Tensor:
        kept = tuple(scores.shape[1:])
        num_queries = len(query)
        step_kept = "step" not in self.score.reduce_dims
        # Steps after which each group of kept elements becomes known.
        offsets = query.steps if step_kept else query.steps[-1:]
        out = torch.empty((num_queries, *kept), dtype=scores.dtype, device=scores.device)
        uniform = self.weights is None and not self.source.expanding and valid is None
        chunk = max(1, min(self.chunk_size, _MAX_ELEMENTS // (len(pool) * math.prod(kept))))

        if self.update is None:
            if uniform:
                out[:] = self.quantile(scores, None, self.alpha)
                return out
            for start in range(0, num_queries, chunk):
                stop = min(start + chunk, num_queries)
                weights = self._weights(pool, query, start, stop, valid, offsets, step_kept)
                out[start:stop] = self.quantile(scores, weights, self.alpha)
            return out

        self.update.reset(self.alpha, kept, scores.device)
        selectors = _group_selectors(kept, step_kept, len(offsets), scores.device)
        origins = query.origins.tolist()
        offset_list = offsets.tolist()
        pending = [0] * len(offset_list)  # first origin whose outcome in group g is unreleased
        for start in range(0, num_queries, chunk):
            stop = min(start + chunk, num_queries)
            weights = None
            if not uniform:
                weights = self._weights(pool, query, start, stop, valid, offsets, step_kept)
            for u in range(start, stop):
                for group, offset in enumerate(offset_list):
                    while pending[group] < u and origins[pending[group]] + offset <= origins[u]:
                        t = pending[group]
                        released = selectors[group]
                        if query_valid is not None:
                            released = released & query_valid[t]
                        self.update.observe(query_scores[t], out[t], released)
                        pending[group] += 1
                row = None if weights is None else weights[u - start : u - start + 1]
                out[u] = self.update.threshold(partial(self._quantile_at, scores, row))
        return out

    def _quantile_at(self, scores: Tensor, weights: Tensor | None, alpha: Alpha) -> Tensor:
        """Quantile model output ``[*kept]`` for one query at level ``alpha``."""
        if weights is None:
            return self.quantile(scores, None, alpha)
        alpha = torch.as_tensor(alpha, dtype=torch.float64, device=scores.device)
        return self.quantile(scores, weights, alpha.unsqueeze(0))[0]

    def _weights(
        self,
        pool: ResidualArchive,
        query: ResidualArchive,
        start: int,
        stop: int,
        valid: Tensor | None,
        offsets: Tensor,
        step_kept: bool,
    ) -> Tensor:
        """Weights ``[c, P, ...]`` for queries ``start:stop``, with every mask applied."""
        kept_ndim = len(self.score.kept_shape(query.horizon, query.num_series))
        chunk = query.select(torch.arange(start, stop, device=query.device))
        weights = torch.ones(stop - start, len(pool), dtype=torch.float64, device=pool.device)
        if self.weights is not None:
            weights = self.weights(pool, chunk).to(torch.float64)
        weights = weights.reshape(*weights.shape, *([1] * (2 + kept_ndim - weights.ndim)))
        if self.source.expanding:
            mask = availability_mask(pool.origins, chunk.origins, offsets)
            weights = weights * _group_mask(mask, step_kept, kept_ndim)
        if valid is not None:
            weights = weights * valid.unsqueeze(0)
        return weights


def _group_mask(mask: Tensor, step_kept: bool, kept_ndim: int) -> Tensor:
    """Reshape a ``[Q, P, G]`` availability mask to broadcast over ``[Q, P, *kept]``."""
    if step_kept:  # one group per step, and step is the first kept dimension
        return mask.reshape(*mask.shape, *([1] * (kept_ndim - 1)))
    return mask[..., 0].reshape(*mask.shape[:2], *([1] * kept_ndim))


def _group_selectors(kept: tuple[int, ...], step_kept: bool, groups: int, device) -> Tensor:
    """``[G, *kept]`` bool: the kept elements whose outcome arrives with each group."""
    if not step_kept:
        return torch.ones((1, *kept), dtype=torch.bool, device=device)
    selectors = torch.zeros((groups, *kept), dtype=torch.bool, device=device)
    for group in range(groups):
        selectors[group, group] = True
    return selectors


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

"""Nonexchangeable conformal prediction."""

from tscp.methods.base import ConformalPredictor
from tscp.registry import register
from tscp.scores import AbsoluteScore, Score
from tscp.sources import ExpandingSource, Source
from tscp.weights import ExponentialWeights


@register("method", "nexcp")
class NexCP(ConformalPredictor):
    """Nonexchangeable conformal prediction (Barber, Candès, Ramdas & Tibshirani, 2023).

    Weighs each past residual by ``decay ** age``, so recent errors count more, and adds
    each test residual to the pool once it is known. The coverage loss under
    nonexchangeability is bounded by the weighted total-variation distance terms of the
    paper. With ``decay=1`` it is split conformal over the same pool.

    Args:
        decay: Weight ratio per base time step, in (0, 1].
        score: Defaults to :class:`~tscp.scores.AbsoluteScore`.
        source: Defaults to :class:`~tscp.sources.ExpandingSource`;
            :class:`~tscp.sources.HoldoutSource` keeps the calibration split fixed.
        alpha: Target miscoverage level.
    """

    def __init__(
        self,
        decay: float = 0.99,
        score: Score | None = None,
        source: Source | None = None,
        alpha: float = 0.1,
    ):
        super().__init__(
            score=score if score is not None else AbsoluteScore(),
            source=source if source is not None else ExpandingSource(),
            weights=ExponentialWeights(decay),
            alpha=alpha,
        )


__all__ = ["NexCP"]

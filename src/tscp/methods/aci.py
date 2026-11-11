"""Adaptive conformal inference on top of conformal thresholds."""

from tscp.methods.base import ConformalPredictor
from tscp.online import ACIUpdate
from tscp.scores import AbsoluteScore, Score
from tscp.sources import ExpandingSource, Source


class ACI(ConformalPredictor):
    """Adaptive conformal inference (Gibbs & Candès, NeurIPS 2021).

    The threshold at each test origin is the conformal quantile at the current level
    ``alpha_t``, which :class:`~tscp.online.ACIUpdate` moves after every released
    outcome. The long-run miss rate converges to ``alpha`` on any sequence.

    Deviation from the paper: the paper recalibrates on a recent window after refitting
    its model. tscp never refits the forecaster; the pool here is the calibration split
    plus every test residual already known (the expanding source).

    Args:
        gamma: Step size of the level update.
        score: Defaults to :class:`~tscp.scores.AbsoluteScore`.
        source: Defaults to :class:`~tscp.sources.ExpandingSource`.
        alpha: Target miscoverage level.
    """

    def __init__(
        self,
        gamma: float = 0.005,
        score: Score | None = None,
        source: Source | None = None,
        alpha: float = 0.1,
    ):
        super().__init__(
            score=score if score is not None else AbsoluteScore(),
            source=source if source is not None else ExpandingSource(),
            update=ACIUpdate(gamma),
            alpha=alpha,
        )


__all__ = ["ACI"]

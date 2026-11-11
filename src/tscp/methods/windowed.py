"""Conformal calibration using a rolling residual window."""

from tscp.methods.base import ConformalPredictor
from tscp.registry import register
from tscp.scores import AbsoluteScore, Score
from tscp.sources import ExpandingSource


@register("method", "windowed")
class WindowedCP(ConformalPredictor):
    """Empirical conformal quantiles over the latest available residual rows.

    Deviation: a rolling window alone does not guarantee coverage for arbitrary
    dependent sequences; the quantile retains the finite-sample test-point correction.

    Args:
        window_size: Number of available rows per release group, before missing masks.
        score: Defaults to AbsoluteScore.
        alpha: Target miscoverage.
    """

    def __init__(self, window_size: int, score: Score | None = None, alpha: float = 0.1):
        super().__init__(
            score=score if score is not None else AbsoluteScore(),
            source=ExpandingSource(window_size),
            alpha=alpha,
        )


__all__ = ["WindowedCP"]

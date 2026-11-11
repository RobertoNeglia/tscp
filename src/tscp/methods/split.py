"""Split conformal prediction."""

from tscp.methods.base import ConformalPredictor
from tscp.registry import register
from tscp.scores import AbsoluteScore, Score
from tscp.sources import HoldoutSource


@register("method", "split")
class SplitCP(ConformalPredictor):
    """Split conformal prediction (Vovk et al., 2005; Lei et al., JASA 2018).

    Scores the calibration split, weighs every score equally and takes the
    ``ceil((1 - alpha)(n + 1))``-th smallest as the threshold. Coverage is at least
    ``1 - alpha`` in finite samples when calibration and test residuals are exchangeable.

    Args:
        score: Defaults to :class:`~tscp.scores.AbsoluteScore`.
        alpha: Target miscoverage level.
    """

    def __init__(self, score: Score | None = None, alpha: float = 0.1):
        super().__init__(
            score=score if score is not None else AbsoluteScore(),
            source=HoldoutSource(),
            alpha=alpha,
        )


__all__ = ["SplitCP"]

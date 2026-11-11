"""Split conformal calibration within origin-level categories."""

from tscp.features import FeatureMap
from tscp.methods.base import ConformalPredictor
from tscp.registry import register
from tscp.scores import AbsoluteScore, Score
from tscp.sources import Source
from tscp.weights.mondrian import MondrianWeights


@register("method", "mondrian")
class MondrianCP(ConformalPredictor):
    """Category-conditional split conformal prediction (Vovk, ACML 2012).

    https://proceedings.mlr.press/v25/vovk12.html

    Deviation: categories are fixed discrete origin-level features; tscp does not
    infer a partition from calibration or test residuals. Coverage requires
    exchangeability within categories, which arbitrary time series need not satisfy.
    """

    def __init__(
        self,
        groups: FeatureMap,
        score: Score | None = None,
        source: Source | None = None,
        alpha: float = 0.1,
    ):
        super().__init__(
            score=score if score is not None else AbsoluteScore(),
            source=source,
            weights=MondrianWeights(groups),
            alpha=alpha,
        )


__all__ = ["MondrianCP"]

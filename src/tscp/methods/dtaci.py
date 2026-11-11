"""Dynamically tuned adaptive conformal inference."""

from tscp.methods.base import ConformalPredictor
from tscp.online.dtaci import DEFAULT_GAMMAS, DtACIUpdate
from tscp.registry import register
from tscp.scores import AbsoluteScore, Score
from tscp.sources import ExpandingSource, Source


@register("method", "dtaci")
class DtACI(ConformalPredictor):
    """Deterministic DtACI (Gibbs & Candès, JMLR 2024, Algorithm 2).

    Deviation from the paper: forecasts are supplied externally, with no forecaster
    refitting. The default calibration source expands as outcomes become available,
    and empirical quantiles use tscp's finite-sample correction. Each kept score element
    has independent expert state; delayed and missing outcomes use the release mask.
    """

    def __init__(
        self,
        gammas: list[float] | tuple[float, ...] = DEFAULT_GAMMAS,
        eta: float | None = None,
        sigma: float = 0.001,
        lookback: int = 500,
        score: Score | None = None,
        source: Source | None = None,
        alpha: float = 0.1,
    ):
        super().__init__(
            score=score if score is not None else AbsoluteScore(),
            source=source if source is not None else ExpandingSource(),
            update=DtACIUpdate(gammas=gammas, eta=eta, sigma=sigma, lookback=lookback),
            alpha=alpha,
        )


__all__ = ["DtACI"]

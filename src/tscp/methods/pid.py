"""Conformal PI and PID presets."""

from tscp.methods.base import ConformalPredictor
from tscp.online.pid import PIDUpdate
from tscp.online.scorecaster import Scorecaster, ThetaScorecaster
from tscp.registry import register
from tscp.scores import AbsoluteScore, Score


@register("method", "pi")
class ConformalPI(ConformalPredictor):
    """Conformal PI control, without scorecasting (Angelopoulos et al., NeurIPS 2023).

    Uses the authors' zero-initialized quantile tracker and lagged tangent integrator.
    Deviation: forecasts are supplied externally; delayed and missing outcomes use
    per-element released-score histories. The calibration scores are used only with
    ``warm_start``, which starts from their split-conformal quantile and scale
    (:class:`~tscp.online.PIDUpdate`); by default this direct threshold controller
    starts at zero, as in the reference implementation.
    """

    def __init__(
        self,
        lr: float = 0.1,
        csat: float = 5.0,
        ki: float = 10.0,
        proportional_lr: bool = True,
        lr_window: int = 100,
        score: Score | None = None,
        alpha: float = 0.1,
        warm_start: bool = False,
    ):
        super().__init__(
            score=score if score is not None else AbsoluteScore(),
            update=PIDUpdate(lr, csat, ki, proportional_lr, lr_window, warm_start=warm_start),
            alpha=alpha,
        )


@register("method", "pid")
class ConformalPID(ConformalPredictor):
    """Conformal PID with a causal scorecaster (Angelopoulos et al., NeurIPS 2023).

    Defaults to :class:`~tscp.online.ThetaScorecaster`, which requires ``tscp[pid]``.
    Deviation: supplied forecasts remain fixed, available calibration scores initialize
    the scorecaster, and its refits use released observations. Every kept score element
    runs its own controller and scorecaster. Initial tracker and integrator are zero,
    unless ``warm_start`` starts the tracker so that the first threshold is the
    calibration quantile (:class:`~tscp.online.PIDUpdate`).
    """

    def __init__(
        self,
        lr: float = 0.1,
        csat: float = 5.0,
        ki: float = 10.0,
        proportional_lr: bool = True,
        lr_window: int = 100,
        scorecaster: Scorecaster | None = None,
        score: Score | None = None,
        alpha: float = 0.1,
        warm_start: bool = False,
    ):
        super().__init__(
            score=score if score is not None else AbsoluteScore(),
            update=PIDUpdate(
                lr,
                csat,
                ki,
                proportional_lr,
                lr_window,
                scorecaster if scorecaster is not None else ThetaScorecaster(),
                warm_start=warm_start,
            ),
            alpha=alpha,
        )


__all__ = ["ConformalPI", "ConformalPID"]

"""Quantile tracking with error integration and optional scorecasting."""

import math

import torch
from torch import Tensor

from tscp.online.base import OnlineUpdate
from tscp.online.scorecaster import Scorecaster
from tscp.quantile import ScoreDistribution
from tscp.registry import register


@register("update", "pid")
class PIDUpdate(OnlineUpdate):
    """Conformal PI/PID control (Angelopoulos, Candès & Tibshirani, NeurIPS 2023).

    The threshold is a quantile tracker plus a tangent error integrator and, when
    supplied, a causal scorecast. Tracker and integral state start at zero. The
    integrator uses the cumulative errors before the current release, matching the
    lagged integrator in the authors' ``quantile_integrator_log_scorecaster`` routine.
    Infinite tangent values are preserved rather than clipped.

    Deviation from the reference implementation: arbitrary delayed and missing outcomes
    are processed per kept element, and rates depend only on previously released test
    scores. A nested scorecaster may initialize from available calibration history.

    Args:
        lr: Nonnegative tracker learning rate.
        csat: Positive tangent saturation scale.
        ki: Nonnegative integral gain; zero disables error integration.
        proportional_lr: Scale the rate by the range of previous observed test scores.
        lr_window: Most recent scores in the rate-scaling range.
        scorecaster: Optional nested online scorecaster; None gives PI control.
    """

    def __init__(
        self,
        lr: float = 0.1,
        csat: float = 5.0,
        ki: float = 10.0,
        proportional_lr: bool = True,
        lr_window: int = 100,
        scorecaster: Scorecaster | None = None,
    ):
        super().__init__()
        for name, value in (("lr", lr), ("ki", ki)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not math.isfinite(csat) or csat <= 0:
            raise ValueError("csat must be finite and positive")
        if isinstance(lr_window, bool) or not isinstance(lr_window, int) or lr_window < 1:
            raise ValueError("lr_window must be a positive integer")
        if scorecaster is not None and not isinstance(scorecaster, Scorecaster):
            raise TypeError("scorecaster must be a Scorecaster")
        self.lr = float(lr)
        self.csat = float(csat)
        self.ki = float(ki)
        self.proportional_lr = bool(proportional_lr)
        self.lr_window = lr_window
        self.scorecaster = scorecaster
        for name in ("target", "tracker", "error_sum", "integral", "count", "scorecast"):
            self.register_buffer(name, None, persistent=False)
        self._history: list[list[float]] = []

    @property
    def is_trainable(self) -> bool:
        return self.scorecaster is not None and self.scorecaster.is_trainable

    def fit(self, train, validation=None):
        if self.is_trainable:
            self.scorecaster.fit(train, validation)
        return self

    def reset(self, alpha: Tensor, device: torch.device) -> None:
        self.target = torch.as_tensor(alpha, dtype=torch.float64, device=device).clone()
        for name in ("tracker", "error_sum", "integral", "count", "scorecast"):
            setattr(self, name, torch.zeros_like(self.target))
        self._history = [[] for _ in range(self.target.numel())]
        if self.scorecaster is not None:
            self.scorecaster.reset(self.target, device)

    def initialize(
        self, scores: Tensor, origins: Tensor, target_times: Tensor, valid: Tensor | None
    ) -> None:
        if self.scorecaster is not None:
            self.scorecaster.initialize(scores, origins, target_times, valid)

    def prepare_query(self, t: int, *, origin: int, target_times: Tensor) -> None:
        if self.scorecaster is not None:
            self.scorecast = self.scorecaster.predict(t, origin=origin, target_times=target_times)
            if (
                self.scorecast.shape != self.target.shape
                or not torch.isfinite(self.scorecast).all()
            ):
                raise ValueError("scorecasts must be finite tensors with the kept shape")
            self.scorecast = self.scorecast.to(self.target)

    def threshold(self, t: int, distribution: ScoreDistribution) -> Tensor:
        return self.tracker + self.integral + self.scorecast

    def finish_query(self, t: int) -> None:
        if self.scorecaster is not None:
            self.scorecaster.finish_query(t)

    def observe(
        self,
        t: int,
        scores: Tensor,
        thresholds: Tensor,
        released: Tensor,
        levels: Tensor | None = None,
    ) -> None:
        if not released.any():
            return
        rate = torch.full_like(self.target, self.lr)
        for element in released.reshape(-1).nonzero().flatten().tolist():
            history = self._history[element]
            if self.proportional_lr and history:
                rate.reshape(-1)[element] *= max(history) - min(history)
            history.append(float(scores.reshape(-1)[element]))
            if len(history) > self.lr_window:
                del history[0]
        errors = (scores > thresholds).double() - self.target
        self.tracker = torch.where(released, self.tracker + rate * errors, self.tracker)
        if self.ki:
            argument = self.error_sum * torch.log1p(self.count) / (self.csat * (self.count + 1))
            integral = self.ki * torch.tan(argument)
            integral = torch.where(argument >= math.pi / 2, torch.inf, integral)
            integral = torch.where(argument <= -math.pi / 2, -torch.inf, integral)
            self.integral = torch.where(released, integral, self.integral)
        self.error_sum = torch.where(released, self.error_sum + errors, self.error_sum)
        self.count += released.double()
        if self.scorecaster is not None:
            self.scorecaster.observe(t, scores, released)


__all__ = ["PIDUpdate"]

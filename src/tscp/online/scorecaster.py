"""Causal forecasts of conformal scores, driven by released outcomes."""

import math

import torch
from torch import Tensor

from tscp.imports import require
from tscp.utils import Component


class Scorecaster(Component):
    """A nested online component that predicts the next conformal score.

    The controller resets it for each replay, supplies available calibration history,
    requests predictions with target times, and releases observed scores. It never
    receives unreleased outcomes. Predictions and release masks have shape ``[*kept]``.
    Initial fitting and parameter selection use the ordinary component fitting splits.
    """

    adapts_online = True

    def reset(self, alpha: Tensor, device: torch.device) -> None:
        raise NotImplementedError

    def initialize(
        self, scores: Tensor, origins: Tensor, target_times: Tensor, valid: Tensor | None
    ) -> None:
        """Read calibration scores whose target times are already available."""

    def predict(self, t: int, *, origin: int, target_times: Tensor) -> Tensor:
        raise NotImplementedError

    def observe(self, t: int, scores: Tensor, released: Tensor) -> None:
        raise NotImplementedError

    def finish_query(self, t: int) -> None:
        """Discard query-specific state after all its release groups have passed."""


class ThetaScorecaster(Scorecaster):
    """Refit a Theta model to each element's released score history.

    Uses ``statsmodels.tsa.forecasting.theta.ThetaModel`` from the optional ``pid``
    extra. Models fit on CPU; predictions return to the controller's device. Settings
    stay fixed during replay. Each fit uses one contiguous run of history on the observed
    time grid, so missing observations are never compressed into adjacent time steps:
    the latest run with enough observations. While a newer run after a gap, such as the
    gap between the calibration and test splits, is still too short, forecasts continue
    from the older run at a longer lead. Without any long enough run the scorecast is
    zero.

    Deviation from the reference PID implementation: available calibration scores seed
    the scorecaster, and every released score may enter the next scheduled refit. Target
    timestamps determine the forecast lead, including across gaps and delayed releases.

    Args:
        period: Seasonal period in forecast-origin increments; 1 disables seasonality.
        min_history: Minimum observations before fitting; at least twice ``period``
            are also required when seasonal adjustment is enabled.
        window: Most recent observations retained; None keeps all released history.
        refit_every: Number of newly observed scores between refits, per element.
    """

    def __init__(
        self,
        period: int = 1,
        min_history: int = 20,
        window: int | None = None,
        refit_every: int = 1,
    ):
        super().__init__()
        for name, value in (
            ("period", period),
            ("min_history", min_history),
            ("refit_every", refit_every),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if min_history < 2:
            raise ValueError("min_history must be at least 2")
        if window is not None and (
            isinstance(window, bool) or not isinstance(window, int) or window < min_history
        ):
            raise ValueError("window must be an integer at least min_history, or None")
        if window is not None and period > 1 and window < 2 * period:
            raise ValueError("a seasonal window must hold at least two periods")
        self.period = period
        self.min_history = min_history
        self.window = window
        self.refit_every = refit_every
        self.register_buffer("target", None, persistent=False)
        self._times: list[list[int]] = []
        self._values: list[list[float]] = []
        self._models: list = []
        self._counts: list[int] = []
        self._fit_counts: list[int] = []
        self._fit_times: list[int | None] = []
        self._fit_spans: list[tuple[int, int] | None] = []
        self._pending: dict[int, tuple[Tensor, Tensor]] = {}
        self._cadence: int | None = None
        self._last_origin: int | None = None

    def reset(self, alpha: Tensor, device: torch.device) -> None:
        self.target = torch.as_tensor(alpha, dtype=torch.float64, device=device).clone()
        size = self.target.numel()
        self._times = [[] for _ in range(size)]
        self._values = [[] for _ in range(size)]
        self._models = [None] * size
        self._counts = [0] * size
        self._fit_counts = [0] * size
        self._fit_times = [None] * size
        self._fit_spans = [None] * size
        self._pending.clear()
        self._cadence = None
        self._last_origin = None

    @staticmethod
    def _segment(times: list[int], cadence: int, required: int) -> tuple[int, int] | None:
        """Return ``(start, end)`` of the latest contiguous run of ``required`` or more."""
        end = len(times)
        while end >= required:
            start = end - 1
            while start > 0 and times[start] - times[start - 1] == cadence:
                start -= 1
            if end - start >= required:
                return start, end
            end = start
        return None

    def _add_cadence(self, gap: int) -> None:
        if gap > 0:
            cadence = gap if self._cadence is None else math.gcd(self._cadence, gap)
            if cadence != self._cadence:
                self._models = [None] * len(self._models)
            self._cadence = cadence

    def _append(self, element: int, time: int, value: float) -> None:
        times, values = self._times[element], self._values[element]
        if not math.isfinite(value):
            raise ValueError("observed scores must be finite")
        if times and time <= times[-1]:
            raise ValueError("score observations must have strictly increasing target times")
        times.append(time)
        values.append(value)
        self._counts[element] += 1
        if self.window is not None and len(times) > self.window:
            del times[0]
            del values[0]

    def initialize(
        self, scores: Tensor, origins: Tensor, target_times: Tensor, valid: Tensor | None
    ) -> None:
        shape = scores.shape
        times = torch.as_tensor(target_times).broadcast_to(shape).reshape(len(scores), -1).cpu()
        values = scores.reshape(len(scores), -1).double().cpu()
        mask = torch.ones_like(scores, dtype=torch.bool) if valid is None else valid
        mask = mask.broadcast_to(shape).reshape(len(scores), -1).cpu()
        for gap in torch.diff(origins).tolist():
            self._add_cadence(int(gap))
        if len(origins):
            self._last_origin = int(origins[-1])
        for row in range(len(scores)):
            for element in range(self.target.numel()):
                if mask[row, element]:
                    self._append(element, int(times[row, element]), float(values[row, element]))

    def predict(self, t: int, *, origin: int, target_times: Tensor) -> Tensor:
        origin = int(origin)
        if self._last_origin is not None:
            self._add_cadence(origin - self._last_origin)
        self._last_origin = origin
        target_times = torch.as_tensor(target_times).broadcast_to(self.target.shape).clone()
        self._pending[t] = (target_times, torch.zeros_like(self.target, dtype=torch.bool))
        predictions = torch.zeros_like(self.target).reshape(-1)
        cadence = self._cadence
        required = max(self.min_history, 2 * self.period if self.period > 1 else 2)
        for element, target_time in enumerate(target_times.reshape(-1).tolist()):
            times, values = self._times[element], self._values[element]
            if cadence is None or len(times) < required:
                continue
            segment = self._segment(times, cadence, required)
            if segment is None:
                continue
            start, end = segment
            span = (times[start], times[end - 1])
            if self._models[element] is None or (
                self._counts[element] - self._fit_counts[element] >= self.refit_every
                and span != self._fit_spans[element]
            ):
                theta = require("statsmodels.tsa.forecasting.theta", "pid")
                self._models[element] = theta.ThetaModel(
                    torch.tensor(values[start:end], dtype=torch.float64).numpy(),
                    period=self.period,
                    deseasonalize=self.period > 1,
                ).fit()
                self._fit_counts[element] = self._counts[element]
                self._fit_times[element] = span[1]
                self._fit_spans[element] = span
            lead = int(target_time) - self._fit_times[element]
            if lead <= 0 or lead % cadence:
                raise ValueError("scorecast target times must be future points on the history grid")
            forecast = self._models[element].forecast(steps=lead // cadence)
            value = float(forecast.iloc[-1])
            if not math.isfinite(value):
                raise ValueError("Theta scorecasts must be finite")
            predictions[element] = value
        return predictions.reshape(self.target.shape)

    def observe(self, t: int, scores: Tensor, released: Tensor) -> None:
        if not released.any():
            return
        target_times, seen = self._pending[t]
        if (seen & released).any():
            raise ValueError("an outcome may only be released once")
        for element in released.reshape(-1).nonzero().flatten().tolist():
            self._append(
                element,
                int(target_times.reshape(-1)[element]),
                float(scores.reshape(-1)[element]),
            )
        seen |= released
        if seen.all():
            del self._pending[t]

    def finish_query(self, t: int) -> None:
        self._pending.pop(t, None)


__all__ = ["Scorecaster", "ThetaScorecaster"]

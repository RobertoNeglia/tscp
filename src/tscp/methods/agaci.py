"""Adaptive conformal intervals aggregated independently at their two endpoints."""

import math
from collections.abc import Sequence
from typing import Self

import torch
from torch import Tensor

from tscp.data import ResidualArchive
from tscp.methods.base import ConformalPredictor, _concat_valid
from tscp.methods.engine import release_schedule
from tscp.quantile import Pool
from tscp.regions import Interval
from tscp.registry import register
from tscp.scores import AbsoluteScore
from tscp.sources import ExpandingSource, Source

DEFAULT_GAMMAS = (
    0.0,
    0.000005,
    0.00005,
    *(i * 0.0001 for i in range(1, 10)),
    *(i * 0.001 for i in range(1, 10)),
    *(i * 0.01 for i in range(1, 10)),
)


class _BernsteinAggregation:
    """Prediction-scoped adaptive BOA with the pinball gradient trick.

    The multiple-rate construction follows Wintenberger (2017), Figure 2,
    https://arxiv.org/abs/1404.1356. The range rounding, initial rate and use of the
    newly estimated rate in the regularizer match the OPERA 1.2.0 convention used
    by the AgACI reference implementation. All accumulation is in float64.
    """

    def __init__(self, num_experts: int, tau: float, device: torch.device):
        self.tau = tau
        self.log_experts = math.log(num_experts)
        self.regret = torch.zeros(num_experts, dtype=torch.float64, device=device)
        self.squares = torch.zeros_like(self.regret)
        self.range = torch.full_like(self.regret, 2.0**-20)
        self.rate = torch.ones_like(self.regret)

    def weights(self) -> Tensor:
        """Normalized expert weights for the next prediction."""
        return torch.softmax(self.rate.log() + self.rate * self.regret, dim=0)

    def predict(self, experts: Tensor) -> Tensor:
        return torch.dot(self.weights(), experts.to(torch.float64))

    def observe(self, experts: Tensor, prediction: Tensor, target: Tensor) -> None:
        """Update from the endpoints and aggregate actually issued at this query."""
        if self.regret.numel() == 1:
            return
        gradient = (target < prediction).to(torch.float64) - self.tau
        advantage = gradient * (prediction - experts.to(torch.float64))
        self.range = torch.maximum(self.range, advantage.abs())
        rounded = torch.exp2(torch.ceil(torch.log2(self.range)))
        self.squares = self.squares + advantage.square()
        candidate = torch.where(
            self.squares > 0,
            torch.sqrt(self.log_experts / self.squares),
            torch.inf,
        )
        self.rate = torch.minimum(rounded.reciprocal(), candidate)
        correction = rounded * (self.rate * advantage > 0.5)
        self.regret = self.regret + 0.5 * (advantage - self.rate * advantage.square() + correction)


@register("method", "agaci")
class AgACI(ConformalPredictor):
    """Aggregated ACI for one-step, univariate forecasts (Zaffran et al., ICML 2022).

    Each expert adapts its entire interval's miscoverage level using a different
    ``gamma``. Two independent Bernstein online aggregators combine its lower and
    upper endpoints, with pinball levels ``alpha / 2`` and ``1 - alpha / 2``.
    Their loss gradients are evaluated at the aggregate endpoints, as in the
    authors' BOA implementation. See Algorithm 1 in
    https://proceedings.mlr.press/v162/zaffran22a.html.

    ``endpoint_bounds`` are finite replacements, in target coordinates, for
    nonfinite expert endpoints. Finite endpoints are never clipped to these values.
    Returned intervals use residual coordinates, like every tscp region.

    Deviation:
        The point forecasts are supplied and never refitted. Experts use tscp's
        conformal order statistic, with an expanding calibration pool unless
        another source is supplied. As in the paper, an expert level at or above
        one gives a zero-radius interval. Endpoint replacement values are required
        explicitly instead of using the reference experiments' data-specific constants.
        Missing outcomes do not update experts or aggregators. Only one-step, univariate archives
        and the absolute-residual expert score are supported.

    Args:
        endpoint_bounds: Finite ``(lower, upper)`` replacements in target units.
        gammas: Nonnegative ACI rates; defaults to the paper's 30-expert grid,
            including a fixed-level expert with rate zero.
        source: Calibration source; an expanding source by default.
        alpha: Target miscoverage of every ACI expert's whole interval.
        chunk_size: Maximum number of query distributions constructed together.
    """

    def __init__(
        self,
        endpoint_bounds: Sequence[float],
        gammas: Sequence[float] | None = None,
        source: Source | None = None,
        alpha: float = 0.1,
        chunk_size: int = 256,
    ):
        bounds = tuple(float(value) for value in endpoint_bounds)
        if len(bounds) != 2 or not all(math.isfinite(value) for value in bounds):
            raise ValueError("endpoint_bounds must contain two finite target-coordinate values")
        if not bounds[0] < bounds[1]:
            raise ValueError("endpoint_bounds must be strictly increasing")
        rates = tuple(float(value) for value in (DEFAULT_GAMMAS if gammas is None else gammas))
        if not rates or not all(math.isfinite(value) and value >= 0 for value in rates):
            raise ValueError("gammas must be a nonempty sequence of finite nonnegative rates")
        super().__init__(
            AbsoluteScore(),
            source=source if source is not None else ExpandingSource(),
            alpha=alpha,
            chunk_size=chunk_size,
        )
        self.endpoint_bounds = bounds
        self.gammas = rates
        if self.source.adapts_online:
            raise ValueError("AgACI does not support online-refitted source components")

    @staticmethod
    def _validate_archive(archive: ResidualArchive) -> None:
        if archive.horizon != 1 or archive.num_series != 1 or int(archive.steps[0]) != 1:
            raise ValueError("AgACI supports one-step, univariate archives only")
        if archive.forecasts is None:
            raise ValueError("AgACI needs forecasts for target-coordinate endpoint aggregation")

    def fit(self, archive: ResidualArchive) -> Self:
        self._validate_archive(archive)
        return super().fit(archive)

    def calibrate(self, archive: ResidualArchive) -> Self:
        self._validate_archive(archive)
        return super().calibrate(archive)

    def predict(self, archive: ResidualArchive) -> Interval:
        self._validate_archive(archive)
        if self.pool_scores is None:
            raise RuntimeError("calibrate() must run before predict()")
        rows = self.source.rows(archive)
        if not torch.equal(archive.origins[rows], self.pool_origins):
            raise ValueError("predict() must get the archive that calibrate() got")
        if archive.splits is None or "test" not in archive.splits:
            raise ValueError("predict() needs an archive with a 'test' split")
        test_rows = archive.splits["test"]
        query = archive.select(test_rows)
        if not bool(torch.isfinite(query.forecasts).all()):
            raise ValueError("AgACI needs finite forecasts at every test origin")
        scores, valid = self.score(query), self.score.valid(query)
        if self.source.expanding:
            pool = Pool(
                torch.cat((self.pool_scores, scores)),
                archive.select(torch.cat((rows, test_rows))),
                _concat_valid(self.pool_valid, valid, self.pool_scores, scores),
            )
        else:
            pool = Pool(self.pool_scores, archive.select(rows), self.pool_valid)

        rates = torch.tensor(self.gammas, device=archive.device, dtype=torch.float64)
        levels = torch.full_like(rates, self.alpha)
        lower_learner = _BernsteinAggregation(len(rates), self.alpha / 2, archive.device)
        upper_learner = _BernsteinAggregation(len(rates), 1 - self.alpha / 2, archive.device)
        # All mutable online state belongs to this prediction, never to fitted state.
        pending: dict[int, tuple[Tensor, Tensor, Tensor, Tensor, Tensor]] = {}
        lower = torch.empty_like(query.residuals)
        upper = torch.empty_like(lower)
        events = iter(release_schedule(query.origins, query.steps))
        for start, stop, distribution in self._distributions(pool, query, query.steps, True):
            for u in range(start, stop):
                _, releases = next(events)
                for t, _group in releases:
                    issued = pending.pop(t)
                    threshold, expert_low, expert_high, aggregate_low, aggregate_high = issued
                    if valid is not None and not bool(valid[t].all()):
                        continue
                    target = (query.forecasts[t] + query.residuals[t]).reshape(()).double()
                    lower_learner.observe(expert_low, aggregate_low, target)
                    upper_learner.observe(expert_high, aggregate_high, target)
                    miss = (scores[t].reshape(()) > threshold).double()
                    levels = levels + rates * (self.alpha - miss)

                current = distribution.select(u - start)
                threshold = current.threshold(levels[:, None, None]).reshape(-1)
                threshold = torch.where(levels >= 1, 0.0, threshold)
                forecast = query.forecasts[u].reshape(()).double()
                expert_low = forecast - threshold.double()
                expert_high = forecast + threshold.double()
                expert_low = torch.where(
                    torch.isfinite(expert_low), expert_low, self.endpoint_bounds[0]
                )
                expert_high = torch.where(
                    torch.isfinite(expert_high), expert_high, self.endpoint_bounds[1]
                )
                aggregate_low = lower_learner.predict(expert_low)
                aggregate_high = upper_learner.predict(expert_high)
                lower[u] = aggregate_low - forecast
                upper[u] = aggregate_high - forecast
                pending[u] = (threshold, expert_low, expert_high, aggregate_low, aggregate_high)
        return Interval(lower, upper)


__all__ = ["AgACI"]

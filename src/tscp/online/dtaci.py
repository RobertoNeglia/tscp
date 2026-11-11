"""Deterministic dynamically tuned adaptive conformal inference."""

import math

import torch
from torch import Tensor

from tscp.online.base import OnlineUpdate
from tscp.quantile import ScoreDistribution
from tscp.registry import register

DEFAULT_GAMMAS = (0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128)


@register("update", "dtaci")
class DtACIUpdate(OnlineUpdate):
    """Deterministic DtACI (Gibbs & Candès, JMLR 2024, Algorithm 2).

    Each expert adapts its own miscoverage level. Exponential pinball-loss weights,
    mixed with a uniform share, combine those levels before taking a quantile. The
    realized level used for each loss belongs to the distribution issued at that query.
    Expert misses use their issued thresholds, including at empirical quantile ties.

    Fixed learning and mixing rates do not guarantee exact long-run target coverage.
    Delayed outcomes update only their released elements.

    Args:
        gammas: Positive expert step sizes; the paper's eight-point grid by default.
        eta: Weight-learning rate. None uses the paper's fixed heuristic, separately
            for each target level, with a local interval of length ``lookback``.
        sigma: Uniform mixing share in (0, 1); 1/1000 in the paper's experiments.
        lookback: Local interval length for the heuristic learning rate.
    """

    needs_levels = True

    def __init__(
        self,
        gammas: list[float] | tuple[float, ...] = DEFAULT_GAMMAS,
        eta: float | None = None,
        sigma: float = 0.001,
        lookback: int = 500,
    ):
        super().__init__()
        if not gammas or any(not math.isfinite(g) or g <= 0 for g in gammas):
            raise ValueError("gammas must be a nonempty sequence of finite positive values")
        if eta is not None and (not math.isfinite(eta) or eta <= 0):
            raise ValueError("eta must be finite and positive")
        if not 0 < sigma < 1:
            raise ValueError("sigma must lie in (0, 1)")
        if isinstance(lookback, bool) or not isinstance(lookback, int) or lookback < 1:
            raise ValueError("lookback must be a positive integer")
        self.gammas = tuple(float(g) for g in gammas)
        self.eta = eta
        self.sigma = float(sigma)
        self.lookback = lookback
        for name in ("target", "expert_levels", "probabilities", "learning_rate", "step_sizes"):
            self.register_buffer(name, None, persistent=False)
        self._issued: dict[int, tuple[Tensor, Tensor, Tensor]] = {}

    def reset(self, alpha: Tensor, device: torch.device) -> None:
        self.target = torch.as_tensor(alpha, dtype=torch.float64, device=device).clone()
        if not ((self.target > 0) & (self.target < 1)).all():
            raise ValueError("target levels must lie in (0, 1)")
        count = len(self.gammas)
        self.step_sizes = torch.tensor(self.gammas, dtype=torch.float64, device=device).reshape(
            count, *([1] * self.target.ndim)
        )
        self.expert_levels = self.target.expand(count, *self.target.shape).clone()
        self.probabilities = torch.full_like(self.expert_levels, 1 / count)
        if self.eta is None:
            numerator = 3 * (math.log(2 * count * self.lookback) + 1)
            self.learning_rate = (
                numerator / (self.lookback * self.target.square() * (1 - self.target).square())
            ).sqrt()
        else:
            self.learning_rate = torch.full_like(self.target, self.eta)
        self._issued.clear()

    def threshold(self, t: int, distribution: ScoreDistribution) -> Tensor:
        expert_thresholds = distribution.threshold(self.expert_levels)
        self._issued[t] = (
            self.expert_levels.clone(),
            expert_thresholds.clone(),
            torch.zeros_like(self.target, dtype=torch.bool),
        )
        combined = (self.probabilities * self.expert_levels).sum(dim=0)
        return distribution.threshold(combined)

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
        if levels is None:
            raise ValueError("DtACI needs the realized levels of the issued distribution")
        expert_levels, expert_thresholds, seen = self._issued[t]
        if (seen & released).any():
            raise ValueError("an outcome may only be released once")
        delta = levels.double().unsqueeze(0) - expert_levels
        losses = self.target * delta - delta.clamp(max=0)
        log_weights = self.probabilities.log() - self.learning_rate * losses
        next_probabilities = (1 - self.sigma) * log_weights.softmax(dim=0)
        next_probabilities = next_probabilities + self.sigma / len(self.gammas)
        misses = (scores.unsqueeze(0) > expert_thresholds).double()
        next_levels = self.expert_levels + self.step_sizes * (self.target - misses)
        mask = released.unsqueeze(0)
        self.probabilities = torch.where(mask, next_probabilities, self.probabilities)
        self.expert_levels = torch.where(mask, next_levels, self.expert_levels)
        seen |= released
        if seen.all():
            del self._issued[t]

    def finish_query(self, t: int) -> None:
        """Discard issued expert state once every release group has passed."""
        self._issued.pop(t, None)


__all__ = ["DEFAULT_GAMMAS", "DtACIUpdate"]

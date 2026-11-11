"""Adaptive conformal inference."""

import torch
from torch import Tensor

from tscp.online.base import OnlineUpdate
from tscp.quantile import ScoreDistribution
from tscp.registry import register


@register("update", "aci")
class ACIUpdate(OnlineUpdate):
    """Adaptive conformal inference (Gibbs & Candès, NeurIPS 2021).

    Keeps a level ``alpha_t`` per kept element and moves it after each released outcome:
    ``alpha_{t+1} = alpha_t + gamma * (alpha - err_t)``, where ``err_t`` is 1 when the
    target fell outside its region. A level at or below 0 gives an unbounded region and
    one at or above 1 an empty region, so over any sequence of ``T`` outcomes the miss
    rate stays within ``(max(alpha_1, 1 - alpha_1) + gamma) / (gamma * T)`` of ``alpha``.

    Args:
        gamma: Step size, positive.
    """

    def __init__(self, gamma: float = 0.005):
        super().__init__()
        if not gamma > 0:
            raise ValueError("gamma must be positive")
        self.gamma = float(gamma)
        self.register_buffer("target", None, persistent=False)
        self.register_buffer("level", None, persistent=False)

    def reset(self, alpha: Tensor, device: torch.device) -> None:
        self.target = torch.as_tensor(alpha, dtype=torch.float64, device=device).clone()
        self.level = self.target.clone()

    def threshold(self, t: int, distribution: ScoreDistribution) -> Tensor:
        return distribution.threshold(self.level)

    def observe(
        self,
        t: int,
        scores: Tensor,
        thresholds: Tensor,
        released: Tensor,
        levels: Tensor | None = None,
    ) -> None:
        miss = (scores > thresholds).to(torch.float64)
        stepped = self.level + self.gamma * (self.target - miss)
        self.level = torch.where(released, stepped, self.level)


__all__ = ["ACIUpdate"]

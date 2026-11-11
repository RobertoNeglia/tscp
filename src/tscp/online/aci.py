"""Adaptive conformal inference."""

from collections.abc import Callable

import torch
from torch import Tensor

from tscp.online.base import OnlineUpdate


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
        self.target: float | None = None
        self.register_buffer("level", None, persistent=False)

    def reset(self, alpha: float, shape: tuple[int, ...], device: torch.device) -> None:
        self.target = float(alpha)
        self.level = torch.full(tuple(shape), self.target, dtype=torch.float64, device=device)

    def threshold(self, quantile: Callable[[Tensor], Tensor]) -> Tensor:
        return quantile(self.level)

    def observe(self, scores: Tensor, thresholds: Tensor, released: Tensor) -> None:
        miss = (scores > thresholds).to(torch.float64)
        stepped = self.level + self.gamma * (self.target - miss)
        self.level = torch.where(released, stepped, self.level)


__all__ = ["ACIUpdate"]

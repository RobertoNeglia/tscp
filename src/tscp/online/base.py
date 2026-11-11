"""The online-update interface (axis 5)."""

from collections.abc import Callable

import torch
from torch import Tensor

from tscp.utils import Component


class OnlineUpdate(Component):
    """How the threshold reacts to its own misses.

    The composer walks the test origins in order. At each origin it first passes every
    outcome that has become known (:meth:`observe`), then asks for the threshold
    (:meth:`threshold`), handing over a function that evaluates the quantile model at any
    miscoverage level. Updates that move a level (the ACI family) call that function;
    updates that move the threshold itself (conformal PID) may adjust or ignore it.
    """

    def reset(self, alpha: float, shape: tuple[int, ...], device: torch.device) -> None:
        """Start a new pass with target level ``alpha`` and state of shape ``kept``."""
        raise NotImplementedError

    def threshold(self, quantile: Callable[[Tensor], Tensor]) -> Tensor:
        """Return the threshold ``[*kept]`` for the current origin.

        Args:
            quantile: Maps a miscoverage level, broadcastable to ``[*kept]``, to the
                quantile model's thresholds ``[*kept]`` at this origin.
        """
        raise NotImplementedError

    def observe(self, scores: Tensor, thresholds: Tensor, released: Tensor) -> None:
        """Take one past origin's outcome.

        Args:
            scores: ``[*kept]`` scores of that origin's residuals.
            thresholds: ``[*kept]`` thresholds that were used there.
            released: ``[*kept]`` bool, True for the elements whose outcome is known now.
                Per-step outcomes arrive one step at a time; missing targets never arrive.
        """
        raise NotImplementedError


__all__ = ["OnlineUpdate"]

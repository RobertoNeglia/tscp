"""Intervals, one per element."""

import torch
from torch import Tensor

from tscp.regions.base import Region


class Interval(Region):
    """Closed intervals ``[lower, upper]``, one per element of the batch.

    Bounds may be infinite: ``(-inf, inf)`` is the whole line, and ``lower > upper``
    (for instance ``(inf, -inf)``) is empty.

    Args:
        lower: Lower bounds, in residual coordinates.
        upper: Upper bounds, the same shape as ``lower``.
    """

    event_dims = ()

    def __init__(self, lower: Tensor, upper: Tensor):
        lower = torch.as_tensor(lower)
        upper = torch.as_tensor(upper, device=lower.device)
        if lower.shape != upper.shape:
            raise ValueError("lower and upper must have the same shape")
        if not (lower.is_floating_point() and upper.is_floating_point()):
            raise ValueError("interval bounds must be floating tensors")
        self.lower = lower
        self.upper = upper

    @property
    def batch_shape(self) -> torch.Size:
        return self.lower.shape

    @property
    def dimension(self) -> int:
        return 1

    def is_empty(self) -> Tensor:
        """True where ``lower > upper``."""
        return self.upper < self.lower

    def width(self) -> Tensor:
        """``upper - lower``, with empty intervals at 0."""
        return torch.where(self.is_empty(), 0.0, self.upper - self.lower)

    def contains(self, residuals: Tensor) -> Tensor:
        return (self.lower <= residuals) & (residuals <= self.upper)

    def log_volume(self) -> Tensor:
        width = self.upper - self.lower
        return torch.where(self.is_empty(), -torch.inf, torch.log(width.clamp(min=0)))

    def to_target(self, forecasts: Tensor) -> "Interval":
        return Interval(self.lower + forecasts, self.upper + forecasts)

    def __getitem__(self, index) -> "Interval":
        return Interval(self.lower[index], self.upper[index])

    def to(self, device: torch.device | str) -> "Interval":
        return Interval(self.lower.to(device), self.upper.to(device))

    def __repr__(self) -> str:
        return f"Interval(batch_shape={tuple(self.batch_shape)})"


__all__ = ["Interval"]

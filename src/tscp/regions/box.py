"""Boxes: an interval per coordinate, covered jointly."""

import torch
from torch import Tensor

from tscp.regions.base import Region, check_event_dims, event_size, move_event_dims


class Box(Region):
    """Axis-aligned boxes over the dimensions in ``event_dims``.

    A box covers a point when every coordinate lies within its bounds. Bounds have shape
    ``[*batch, *event]``: the batch dimensions (the query origin, then whichever of step
    and series is not covered jointly), then the dimensions in ``event_dims``, both in
    ``[H, N]`` order. For example, a box per series over the whole horizon has bounds
    ``[Q, N, H]`` and ``event_dims=("step",)``.

    The log-volume is the sum of the coordinates' log-widths. A box with an empty or
    zero-width coordinate has log-volume ``-inf``; otherwise any infinite coordinate makes
    it ``+inf``.

    Args:
        lower: Lower bounds ``[*batch, *event]``, in residual coordinates.
        upper: Upper bounds, the same shape as ``lower``.
        event_dims: The dimensions of ``[H, N]`` each box covers jointly.
    """

    def __init__(self, lower: Tensor, upper: Tensor, event_dims=("step",)):
        event_dims = check_event_dims(event_dims)
        if not event_dims:
            raise ValueError("a box covers at least one dimension jointly; use an Interval")
        lower = torch.as_tensor(lower)
        upper = torch.as_tensor(upper, device=lower.device)
        if lower.shape != upper.shape:
            raise ValueError("lower and upper must have the same shape")
        if not (lower.is_floating_point() and upper.is_floating_point()):
            raise ValueError("box bounds must be floating tensors")
        if lower.ndim < len(event_dims):
            raise ValueError("bounds must end with one dimension per entry of event_dims")
        self.lower = lower
        self.upper = upper
        self.event_dims = event_dims

    @property
    def batch_shape(self) -> torch.Size:
        return self.lower.shape[: self.lower.ndim - len(self.event_dims)]

    @property
    def dimension(self) -> int:
        return event_size(self.lower.shape, len(self.event_dims))

    def tensors(self) -> dict[str, Tensor]:
        return {"lower": self.lower, "upper": self.upper}

    def _rebuild(self, tensors: dict[str, Tensor]) -> "Box":
        return Box(tensors["lower"], tensors["upper"], self.event_dims)

    def _flat(self, values: Tensor) -> Tensor:
        """``[*batch, *event]`` values with the event dimensions flattened into one."""
        return values.reshape(*self.batch_shape, self.dimension)

    def is_empty(self) -> Tensor:
        """True where some coordinate has ``lower > upper``."""
        return self._flat(self.upper < self.lower).any(dim=-1)

    def contains(self, residuals: Tensor) -> Tensor:
        points = move_event_dims(residuals, self.event_dims)
        inside = (self.lower <= points) & (points <= self.upper)
        return self._flat(inside).all(dim=-1)

    def log_volume(self) -> Tensor:
        width = self._flat(self.upper - self.lower)
        log_width = torch.log(width.clamp(min=0))  # -inf for empty or zero-width coordinates
        degenerate = torch.isneginf(log_width).any(dim=-1)
        return torch.where(degenerate, -torch.inf, log_width.sum(dim=-1))

    def to_target(self, forecasts: Tensor) -> "Box":
        shift = move_event_dims(forecasts, self.event_dims)
        return Box(self.lower + shift, self.upper + shift, self.event_dims)


__all__ = ["Box"]

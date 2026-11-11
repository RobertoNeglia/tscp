"""The region interface (axis 6, output)."""

import torch
from torch import Tensor


class Region:
    """A batch of prediction regions, one per query origin and kept element.

    Regions live in residual coordinates: a target ``y`` with point forecast ``f`` is
    covered when ``y - f`` lies in the region. Coverage and volume do not change under
    this shift, so evaluation never needs the forecasts; :meth:`to_target` maps a region
    back to target units.
    """

    #: Dimensions of ``[H, N]`` that each region covers jointly: the ones its score reduced.
    event_dims: tuple[str, ...] = ()

    @property
    def batch_shape(self) -> torch.Size:
        """``[Q, *kept]``: one region per query origin and kept element."""
        raise NotImplementedError

    @property
    def dimension(self) -> int:
        """Number of coordinates each region covers jointly (1 for an interval)."""
        raise NotImplementedError

    def contains(self, residuals: Tensor) -> Tensor:
        """Return a bool tensor of shape :attr:`batch_shape`, True where covered."""
        raise NotImplementedError

    def log_volume(self) -> Tensor:
        """Return the log-volume of each region: ``+inf`` if unbounded, ``-inf`` if empty."""
        raise NotImplementedError

    def to_target(self, forecasts: Tensor) -> "Region":
        """Return the region shifted by the point forecasts, in target units."""
        raise NotImplementedError

    def __getitem__(self, index) -> "Region":
        raise NotImplementedError

    def to(self, device: torch.device | str) -> "Region":
        raise NotImplementedError

    def __len__(self) -> int:
        return self.batch_shape[0]


__all__ = ["Region"]

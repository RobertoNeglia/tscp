"""The region interface (axis 6, output)."""

import math

import torch
from torch import Tensor

RESIDUAL_DIMS = ("step", "series")
"""Names of the dimensions of ``[H, N]``, in the order residuals store them."""


def check_event_dims(event_dims) -> tuple[str, ...]:
    """Return ``event_dims`` as a tuple in ``[H, N]`` order, checking the names."""
    event_dims = tuple(event_dims)
    unknown = sorted(set(event_dims) - set(RESIDUAL_DIMS))
    if unknown or len(set(event_dims)) != len(event_dims):
        raise ValueError(f"event_dims must name distinct dimensions of {RESIDUAL_DIMS}")
    return tuple(name for name in RESIDUAL_DIMS if name in event_dims)


def move_event_dims(values: Tensor, event_dims: tuple[str, ...]) -> Tensor:
    """Reorder ``[..., H, N]`` values to ``[..., *kept, *event]``.

    The dimensions named in ``event_dims`` move to the end and the others stay in front,
    each group in ``[H, N]`` order. This is how a region that covers some dimensions
    jointly lines residuals or forecasts up with its bounds.
    """
    if values.ndim < 2:
        raise ValueError("expected values of shape [..., H, N]")
    lead = values.ndim - 2
    kept = [lead + i for i, name in enumerate(RESIDUAL_DIMS) if name not in event_dims]
    event = [lead + i for i, name in enumerate(RESIDUAL_DIMS) if name in event_dims]
    return values.permute(*range(lead), *kept, *event)


class Region:
    """A batch of prediction regions, one per query origin and kept element.

    Regions live in residual coordinates: a target ``y`` with point forecast ``f`` is
    covered when ``y - f`` lies in the region. Coverage and volume do not change under
    this shift, so evaluation never needs the forecasts; :meth:`to_target` maps a region
    back to target units.

    A region covers the dimensions of ``[H, N]`` in :attr:`event_dims` jointly and makes
    one region per element of the others. :meth:`contains` always takes the residuals of
    the query rows, ``[Q, H, N]``, and returns one answer per region, of shape
    :attr:`batch_shape`.

    A subclass stores its regions in tensors whose leading dimensions are the batch
    dimensions, returns them from :meth:`tensors`, and can be rebuilt from them by its
    constructor; indexing and device moves then work without more code. A subclass with
    settings that are not tensors overrides :meth:`_rebuild`.
    """

    #: Dimensions of ``[H, N]`` that each region covers jointly.
    event_dims: tuple[str, ...] = ()

    @property
    def batch_shape(self) -> torch.Size:
        """``[Q, *kept]``: one region per query origin and element not covered jointly."""
        raise NotImplementedError

    @property
    def dimension(self) -> int:
        """Number of coordinates each region covers jointly (1 for an interval)."""
        raise NotImplementedError

    def tensors(self) -> dict[str, Tensor]:
        """The tensors that define the regions, by constructor argument name."""
        raise NotImplementedError

    def contains(self, residuals: Tensor) -> Tensor:
        """Return a bool tensor of shape :attr:`batch_shape`, True where covered.

        Args:
            residuals: ``[Q, H, N]`` residuals (or any points in the region's coordinates).
        """
        raise NotImplementedError

    def log_volume(self) -> Tensor:
        """Return the log-volume of each region.

        ``+inf`` means unbounded; ``-inf`` means empty or of zero volume.
        """
        raise NotImplementedError

    def to_target(self, forecasts: Tensor) -> "Region":
        """Return the regions shifted by the ``[Q, H, N]`` point forecasts, in target units."""
        raise NotImplementedError

    def _rebuild(self, tensors: dict[str, Tensor]) -> "Region":
        """Return a region of the same type and settings, made of other tensors."""
        return type(self)(**tensors)

    def __getitem__(self, index) -> "Region":
        """Select regions along the batch dimensions, as ``region[3]`` or ``region[mask]``."""
        items = index if isinstance(index, tuple) else (index,)
        if any(item is Ellipsis for item in items):
            raise IndexError("regions do not support Ellipsis in indices")
        if len(items) > len(self.batch_shape):
            raise IndexError("too many indices: regions are indexed along batch dimensions")
        return self._rebuild({name: value[index] for name, value in self.tensors().items()})

    def to(self, device: torch.device | str) -> "Region":
        """Return the regions on ``device``."""
        return self._rebuild({name: value.to(device) for name, value in self.tensors().items()})

    def __len__(self) -> int:
        return self.batch_shape[0]

    def __repr__(self) -> str:
        parts = [f"batch_shape={tuple(self.batch_shape)}"]
        if self.event_dims:
            parts.append(f"event_dims={self.event_dims}")
        return f"{type(self).__name__}({', '.join(parts)})"


def event_size(shape: torch.Size, event_ndim: int) -> int:
    """Number of coordinates in the last ``event_ndim`` dimensions of ``shape``."""
    return math.prod(shape[len(shape) - event_ndim :]) if event_ndim else 1


__all__ = ["RESIDUAL_DIMS", "Region", "check_event_dims", "move_event_dims"]

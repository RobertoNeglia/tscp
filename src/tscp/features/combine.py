"""Several feature maps side by side."""

import torch
from torch import Tensor

from tscp.data import ResidualArchive
from tscp.features.base import FeatureMap
from tscp.registry import register


def _element(values: Tensor) -> int:
    """Number of element dimensions of ``[T, *element, ...]`` features or masks."""
    return values.ndim - 2


def _spread(values: Tensor, ndim: int, target: tuple[int, ...]) -> Tensor:
    """Repeat ``[T, *element, ...]`` across missing leading element dimensions.

    An element of ``()`` or ``(N,)`` lacks the leading dimensions of ``(H, N)``, so new
    dimensions go right after the row dimension, not at the end.
    """
    missing = len(target) - ndim
    for _ in range(missing):
        values = values.unsqueeze(1)
    return values.expand(values.shape[0], *target, *values.shape[1 + len(target) :])


@register("features", "concat")
class Concat(FeatureMap):
    """Several feature maps side by side, broadcast to a common element shape.

    A map per row (element ``()``) or per series (``(N,)``) is repeated over the steps
    and series of a map per step and series (``(H, N)``), and the features are
    concatenated. An element is valid where every map's features are.

    Args:
        maps: The feature maps, at least one.
    """

    def __init__(self, maps: list[FeatureMap]):
        super().__init__()
        if not maps:
            raise ValueError("Concat needs at least one feature map")
        self.maps = torch.nn.ModuleList(maps)

    def _target(self, parts: list[Tensor]) -> tuple[int, ...]:
        widest = max(parts, key=_element)
        return tuple(widest.shape[1:-1])

    def forward(self, rows: ResidualArchive) -> Tensor:
        parts = [feature_map(rows) for feature_map in self.maps]
        target = self._target(parts)
        spread = [_spread(part, _element(part), target) for part in parts]
        return torch.cat(spread, dim=-1)

    def valid(self, rows: ResidualArchive) -> Tensor | None:
        parts = [feature_map(rows) for feature_map in self.maps]
        target = self._target(parts)
        masks = [feature_map.valid(rows) for feature_map in self.maps]
        valid = None
        for part, mask in zip(parts, masks, strict=True):
            if mask is None:
                continue
            mask = _spread(mask.unsqueeze(-1), _element(part), target)[..., 0]
            valid = mask if valid is None else valid & mask
        return valid


__all__ = ["Concat"]

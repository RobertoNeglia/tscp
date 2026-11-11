"""Residuals divided by their expected size."""

from typing import Self

import torch
from torch import Tensor

from tscp.data import ResidualArchive
from tscp.regions import Interval
from tscp.registry import register
from tscp.scores.base import Score


@register("score", "normalized")
class NormalizedScore(Score):
    """``|r| / scale``: each error in units of its expected size.

    The scale comes either from the forecaster, as an output with a predicted spread per
    target (``scale="name"``), or from the fit split (``scale=None``): the root mean
    square of the observed residuals of each step and series. A threshold ``q`` gives the
    interval ``[-q * scale, q * scale]``, so regions widen where errors are expected to be
    large.

    Args:
        scale: Name of a forecaster output of shape ``[T, H, N]`` with positive scales,
            or None to fit one scale per step and series on the fit split.
    """

    def __init__(self, scale: str | None = None):
        super().__init__()
        self.scale = scale
        self.register_buffer("fitted_scale", None)

    @property
    def is_trainable(self) -> bool:
        return self.scale is None

    def fit(self, train: ResidualArchive, validation: ResidualArchive | None = None) -> Self:
        residuals = train.residuals.to(torch.float64)
        observed = train.observed
        if observed is None:
            observed = torch.ones_like(train.residuals, dtype=torch.bool)
        count = observed.sum(dim=0)
        if (count == 0).any():
            raise ValueError("every step and series needs an observed residual in the fit split")
        scale = ((residuals.square() * observed).sum(dim=0) / count).sqrt()
        if (scale == 0).any():
            raise ValueError("a fitted scale is zero: the fit split has only zero residuals")
        self.fitted_scale = scale.to(train.residuals.dtype)
        return self

    def _scale(self, rows: ResidualArchive) -> Tensor:
        if self.scale is None:
            if self.fitted_scale is None:
                raise RuntimeError("fit() must run before a fitted scale is used")
            return self.fitted_scale  # [H, N], broadcast over rows
        if self.scale not in rows.outputs:
            raise KeyError(f"the archive has no output {self.scale!r}")
        scale = rows.outputs[self.scale]
        if scale.shape != rows.residuals.shape:
            raise ValueError(f"output {self.scale!r} must have shape [T, H, N]")
        if not (scale > 0).all():
            raise ValueError(f"output {self.scale!r} must be positive")
        return scale

    def forward(self, rows: ResidualArchive) -> Tensor:
        return rows.residuals.abs() / self._scale(rows)

    def region(self, threshold: Tensor, rows: ResidualArchive) -> Interval:
        width = threshold * self._scale(rows)
        return Interval(-width, width)


__all__ = ["NormalizedScore"]

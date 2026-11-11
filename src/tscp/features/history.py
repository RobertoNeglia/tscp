"""The residuals known before each origin."""

import torch
from torch import Tensor

from tscp.data import ResidualArchive
from tscp.features.base import FeatureMap, timeline_of
from tscp.registry import register


@register("features", "history")
class ResidualHistory(FeatureMap):
    """The last ``lags`` known residuals of each step and series: ``[T, H, N, lags]``.

    For a row with origin ``t``, lag 1 of step ``h`` is the most recent residual of step
    ``h`` whose target time has passed by ``t``, lag 2 the one before, and so on. The
    timeline's rows are in time order, so these are the last rows among the first
    ``known_counts(t)[h]``. A lag that is not known yet, or whose target was not observed,
    is 0 and makes the element invalid.

    Args:
        lags: Number of past residuals per step, positive.
        absolute: Give absolute residuals.
    """

    def __init__(self, lags: int = 1, absolute: bool = False):
        super().__init__()
        if lags < 1:
            raise ValueError("lags must be positive")
        self.lags = int(lags)
        self.absolute = bool(absolute)

    def _history(self, rows: ResidualArchive) -> tuple[Tensor, Tensor]:
        """Values and known mask, both ``[T, H, N, lags]``."""
        timeline = timeline_of(rows)
        device = timeline.device
        counts = timeline.known_counts(rows.origins)  # [T, H]
        index = counts[..., None] - torch.arange(1, self.lags + 1, device=device)  # [T, H, L]
        safe = index.clamp(min=0)
        steps = torch.arange(timeline.horizon, device=device)[None, :, None]
        values = timeline.residuals.transpose(0, 1)[steps, safe]  # [T, H, L, N]
        known = (index >= 0)[..., None].expand_as(values)
        if timeline.observed is not None:
            known = known & timeline.observed.transpose(0, 1)[steps, safe]
        values = torch.where(known, values, torch.zeros((), dtype=values.dtype, device=device))
        if self.absolute:
            values = values.abs()
        return values.permute(0, 1, 3, 2), known.permute(0, 1, 3, 2)

    def forward(self, rows: ResidualArchive) -> Tensor:
        return self._history(rows)[0]

    def valid(self, rows: ResidualArchive) -> Tensor | None:
        known = self._history(rows)[1].all(dim=-1)
        return None if bool(known.all()) else known


__all__ = ["ResidualHistory"]

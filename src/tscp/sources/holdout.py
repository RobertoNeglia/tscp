"""The calibration split as a fixed pool."""

from torch import Tensor

from tscp.data import ResidualArchive
from tscp.registry import register
from tscp.sources.base import Source, calibration_rows


@register("source", "holdout")
class HoldoutSource(Source):
    """The calibration split, fixed for the whole test period (split conformal)."""

    def rows(self, archive: ResidualArchive) -> Tensor:
        return calibration_rows(archive)


__all__ = ["HoldoutSource"]

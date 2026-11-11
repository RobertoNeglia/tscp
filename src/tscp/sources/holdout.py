"""The calibration split as a fixed pool."""

from torch import Tensor

from tscp.data import ResidualArchive
from tscp.sources.base import Source, calibration_rows


class HoldoutSource(Source):
    """The calibration split, fixed for the whole test period (split conformal)."""

    def rows(self, archive: ResidualArchive) -> Tensor:
        return calibration_rows(archive)


__all__ = ["HoldoutSource"]

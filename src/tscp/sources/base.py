"""The residual-source interface (axis 1)."""

from torch import Tensor

from tscp.data import ResidualArchive
from tscp.utils import Component


class Source(Component):
    """Which past residuals calibrate.

    :meth:`rows` names the archive rows scored at calibration time. An expanding source
    also adds each test residual to the pool once its target time has passed (see
    :mod:`tscp.data.availability`); a static source keeps the pool fixed.
    """

    #: Whether test residuals join the pool once they are known.
    expanding: bool = False

    def rows(self, archive: ResidualArchive) -> Tensor:
        """Return the archive rows scored at calibration time."""
        raise NotImplementedError


def calibration_rows(archive: ResidualArchive) -> Tensor:
    """Return the rows of the calibration split, with a clear error if there is none."""
    if archive.splits is None or "calibration" not in archive.splits:
        raise ValueError("this source needs an archive with a 'calibration' split")
    return archive.splits["calibration"]


__all__ = ["Source", "calibration_rows"]

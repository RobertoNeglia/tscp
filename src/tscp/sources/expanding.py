"""The calibration split plus test residuals as they become known."""

from torch import Tensor

from tscp.data import ResidualArchive
from tscp.sources.base import Source, calibration_rows


class ExpandingSource(Source):
    """The calibration split, plus each test residual once its target time has passed.

    The pool at test origin ``t`` holds every calibration residual and every test
    residual of (origin ``s``, step ``h``) with ``s + h <= t``. The ``fit`` split never
    joins the pool: components were fitted on it.
    """

    expanding = True

    def rows(self, archive: ResidualArchive) -> Tensor:
        return calibration_rows(archive)


__all__ = ["ExpandingSource"]

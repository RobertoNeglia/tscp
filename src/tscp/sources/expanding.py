"""The calibration split plus test residuals as they become known."""

from torch import Tensor

from tscp.data import ResidualArchive
from tscp.registry import register
from tscp.sources.base import Source, calibration_rows


@register("source", "expanding")
class ExpandingSource(Source):
    """The calibration split, plus each test residual once its target time has passed.

    The pool at test origin ``t`` holds every calibration residual and every test
    residual of (origin ``s``, step ``h``) with ``s + h <= t``. The ``fit`` split never
    joins the pool: components were fitted on it.

    ``window_size`` limits support to the latest available rows per release group,
    before excluding missing scores. Missing rows consume slots. It limits calibration
    support, rather than the size of the archive held in memory.
    """

    expanding = True

    def __init__(self, window_size: int | None = None):
        super().__init__()
        if window_size is not None and (
            isinstance(window_size, bool) or not isinstance(window_size, int) or window_size < 1
        ):
            raise ValueError("window_size must be a positive integer or None")
        self.window_size = window_size

    def rows(self, archive: ResidualArchive) -> Tensor:
        return calibration_rows(archive)


__all__ = ["ExpandingSource"]

"""Minimal components for exercising code paths in checks and tests."""

from torch import Tensor

from tscp.data import ResidualArchive
from tscp.regions import Box
from tscp.scores import Score


class MaxAbsScore(Score):
    """The largest absolute residual over the horizon: one joint box per series.

    A minimal score that reduces the step dimension. Checking a method with it shows that
    the method handles scores over the whole horizon, including their release after the
    last step and their validity under missing targets.
    """

    reduce_dims = ("step",)

    def forward(self, rows: ResidualArchive) -> Tensor:
        return rows.residuals.abs().amax(dim=1)

    def region(self, threshold: Tensor, rows: ResidualArchive) -> Box:
        bound = threshold.unsqueeze(-1).expand(*threshold.shape, rows.horizon)
        return Box(-bound, bound, event_dims=("step",))


__all__ = ["MaxAbsScore"]

"""The score interface (axis 2)."""

import torch
from torch import Tensor

from tscp.data import ResidualArchive
from tscp.regions import RESIDUAL_DIMS, Region
from tscp.utils import Component


class Score(Component):
    """How an error becomes a number, and the region a threshold on it defines.

    A score maps the rows of an archive to ``[T, *kept]``. The kept dimensions are the
    dimensions of ``[H, N]`` it does not reduce, followed by one dimension for its
    :attr:`channels` if it has any. A per-step score keeps step and series and yields an
    interval per element. A score over the whole horizon reduces ``"step"`` and yields one
    joint region per series; a score across series reduces ``"series"``. Weights, quantile
    models and online updates treat each kept element as its own problem.

    A score sees the whole rows: residuals, forecasts, forecaster outputs and context,
    and, through the rows' timeline, the history before them. The region of a threshold
    ``q`` is ``{r : score(r) <= q}``, so a target is missed exactly when its score exceeds
    the threshold.

    Channels split one element into several scores that are thresholded separately, such
    as the lower and upper side of an interval. :meth:`levels` then shares the miscoverage
    ``alpha`` among them so that the combined region still misses with probability at
    most ``alpha``.
    """

    #: Dimensions of ``[H, N]`` this score reduces into one number.
    reduce_dims: tuple[str, ...] = ()
    #: Names of the channels, each thresholded separately; empty for one score per element.
    channels: tuple[str, ...] = ()

    def forward(self, rows: ResidualArchive) -> Tensor:
        """Return scores of shape ``[T, *kept]`` for the ``T`` rows of ``rows``."""
        raise NotImplementedError

    def region(self, threshold: Tensor, rows: ResidualArchive) -> Region:
        """Return the region ``{r : score(r) <= threshold}`` for each query row.

        Args:
            threshold: ``[Q, *kept]`` thresholds, one per query row and kept element.
            rows: The ``Q`` query rows. Their forecasts, outputs and context are known at
                their origins, so a region may use them; their residuals are not, and a
                region must never read them.
        """
        raise NotImplementedError

    def kept_shape(self, horizon: int, num_series: int) -> tuple[int, ...]:
        """Return the sizes of the kept dimensions for an archive with ``H`` and ``N``."""
        unknown = set(self.reduce_dims) - set(RESIDUAL_DIMS)
        if unknown:
            raise ValueError(f"reduce_dims may only name {RESIDUAL_DIMS}, not {sorted(unknown)}")
        sizes = {"step": horizon, "series": num_series}
        kept = tuple(size for name, size in sizes.items() if name not in self.reduce_dims)
        return (*kept, len(self.channels)) if self.channels else kept

    def levels(self, alpha: float, kept: tuple[int, ...]) -> Tensor:
        """Return the miscoverage level of each kept element, ``[*kept]`` in float64.

        By default every element gets ``alpha``. A score with channels overrides this to
        share ``alpha`` among them.
        """
        return torch.full(kept, float(alpha), dtype=torch.float64)

    def valid(self, rows: ResidualArchive) -> Tensor | None:
        """Return which scores are defined, ``[T, *kept]`` bool, or None if all are.

        A score is defined where every residual it combines is observed: its own element
        for a per-step score, and every step or series it reduces otherwise. Channels
        share the mask of their element.
        """
        observed = rows.observed
        if observed is None or bool(observed.all()):
            return None
        reduced = tuple(1 + RESIDUAL_DIMS.index(name) for name in self.reduce_dims)
        if reduced:
            observed = observed.all(dim=reduced)
        if self.channels:
            observed = observed.unsqueeze(-1).expand(*observed.shape, len(self.channels))
        return observed


__all__ = ["Score"]

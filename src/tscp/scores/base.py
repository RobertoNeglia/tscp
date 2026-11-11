"""The score interface (axis 2)."""

from collections.abc import Mapping

from torch import Tensor

from tscp.regions import Region
from tscp.utils import Component

_DIMS = ("step", "series")


class Score(Component):
    """How an error becomes a number, and the region a threshold on it defines.

    A score maps residuals ``[T, H, N]`` (and forecaster outputs) to ``[T, *kept]``,
    where ``kept`` are the dimensions of ``[H, N]`` it does not reduce. A per-step score
    keeps both and yields an interval per element. A score over the whole horizon
    reduces ``"step"`` and yields one joint region per series; a score across series
    reduces ``"series"``. Weights, quantile models and online updates then treat each
    kept element as its own problem.

    The region of a threshold ``q`` is ``{r : score(r) <= q}``, so a target is missed
    exactly when its score exceeds the threshold.
    """

    #: Dimensions of ``[H, N]`` this score reduces into one number.
    reduce_dims: tuple[str, ...] = ()

    def forward(self, residuals: Tensor, outputs: Mapping[str, Tensor] | None = None) -> Tensor:
        """Return scores of shape ``[T, *kept]`` for residuals of shape ``[T, H, N]``."""
        raise NotImplementedError

    def region(self, threshold: Tensor, outputs: Mapping[str, Tensor] | None = None) -> Region:
        """Return the region ``{r : score(r) <= threshold}`` for thresholds ``[Q, *kept]``."""
        raise NotImplementedError

    def kept_shape(self, horizon: int, num_series: int) -> tuple[int, ...]:
        """Return the sizes of the kept dimensions for an archive with ``H`` and ``N``."""
        unknown = set(self.reduce_dims) - set(_DIMS)
        if unknown:
            raise ValueError(f"reduce_dims may only name {_DIMS}, not {sorted(unknown)}")
        sizes = {"step": horizon, "series": num_series}
        return tuple(size for name, size in sizes.items() if name not in self.reduce_dims)


__all__ = ["Score"]

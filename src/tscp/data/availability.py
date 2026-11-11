"""When a residual becomes known.

The residual of (origin ``s``, step ``h``) is known once its target time ``s + h`` has
passed, so at forecast origin ``t`` it is usable only if ``s + h <= t``. Every pool that
grows during the test period, every feature built from past residuals and every online
update goes through this rule, so nothing sees a residual before its target time.
"""

from torch import Tensor


def target_times(origins: Tensor, steps: Tensor) -> Tensor:
    """Return the ``[T, H]`` target time of each (origin, step): ``origin + step``."""
    return origins[:, None] + steps[None, :]


def availability_mask(pool_origins: Tensor, query_origins: Tensor, offsets: Tensor) -> Tensor:
    """Return which pool residuals are known at each query origin.

    Args:
        pool_origins: ``[P]`` origins of the pool rows.
        query_origins: ``[Q]`` origins at which the pool is used.
        offsets: ``[G]`` steps after which each group of residuals is known: the horizon
            steps for per-step scores, or the last step for a score over the whole horizon.

    Returns:
        A ``[Q, P, G]`` bool mask, True where ``pool_origin + offset <= query_origin``.
    """
    return (pool_origins[None, :, None] + offsets[None, None, :]) <= query_origins[:, None, None]


__all__ = ["availability_mask", "target_times"]

"""When a residual becomes known.

The residual of (origin ``s``, step ``h``) is known once its target time ``s + h`` has
passed, so at forecast origin ``t`` it is usable only if ``s + h <= t``. Every pool that
grows during the test period, every feature built from past residuals and every online
update goes through this rule, so nothing sees a residual before its target time.
"""

import torch
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


def known_counts(origins: Tensor, steps: Tensor, query_origins: Tensor) -> Tensor:
    """Return how many leading rows have a known residual at each query origin, per step.

    Rows are in time order, so the residuals of step ``h`` known at origin ``t`` are always
    a prefix of the rows: those with ``origin + h <= t``. This is the rule of
    :func:`availability_mask` in a form whose size does not grow with the pool, for
    components that read the history of an archive.

    Args:
        origins: ``[T]`` origins of the rows, strictly increasing.
        steps: ``[H]`` horizon steps.
        query_origins: ``[Q]`` origins at which the history is read.

    Returns:
        ``[Q, H]`` counts: at query ``q``, the residuals of step ``h`` in rows
        ``0 .. counts[q, h] - 1`` are known, and no others.
    """
    times = (origins[None, :] + steps[:, None]).contiguous()
    queries = query_origins[None, :].expand(steps.numel(), -1).contiguous()
    return torch.searchsorted(times, queries, right=True).T


__all__ = ["availability_mask", "known_counts", "target_times"]

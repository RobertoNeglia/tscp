"""The online event loop.

An online method answers queries in time order and learns from outcomes as they become
known. :class:`OnlineEngine` turns one outcome into the release mask an update observes;
:func:`replay` runs it over the test queries of an archive, releasing each outcome once
its target time has passed, by the rule of :mod:`tscp.data.availability`. The engine
itself does not depend on archives: code that answers live queries can drive it with
outcomes as they arrive.
"""

from collections.abc import Iterable

import torch
from torch import Tensor

from tscp.online import OnlineUpdate
from tscp.quantile import ScoreDistribution


def group_selectors(
    kept: tuple[int, ...], step_kept: bool, groups: int, device: torch.device
) -> Tensor:
    """``[G, *kept]`` bool: the kept elements whose outcome arrives with each group.

    With steps kept, group ``g`` is horizon step ``g`` (the first kept dimension); with
    a score over the horizon there is one group, released after the last step.
    """
    if not step_kept:
        return torch.ones((1, *kept), dtype=torch.bool, device=device)
    selectors = torch.zeros((groups, *kept), dtype=torch.bool, device=device)
    for group in range(groups):
        selectors[group, group] = True
    return selectors


class OnlineEngine:
    """Feeds an online update its queries and outcomes.

    Args:
        update: The update to drive; :meth:`OnlineUpdate.reset` is called here.
        levels: ``[*kept]`` target miscoverage of each kept element.
        step_kept: Whether the score keeps the step dimension, so that each step's
            outcome arrives separately.
        num_groups: Number of release groups: the number of steps if they are kept, else 1.
    """

    def __init__(self, update: OnlineUpdate, levels: Tensor, step_kept: bool, num_groups: int):
        self.update = update
        self.selectors = group_selectors(tuple(levels.shape), step_kept, num_groups, levels.device)
        update.reset(levels, levels.device)

    def threshold(self, t: int, distribution: ScoreDistribution) -> Tensor:
        """The threshold of query ``t``."""
        return self.update.threshold(t, distribution)

    def release(
        self,
        t: int,
        group: int,
        scores: Tensor,
        thresholds: Tensor,
        valid: Tensor | None,
        levels: Tensor | None,
    ) -> None:
        """Pass the outcome of release group ``group`` of query ``t`` to the update."""
        released = self.selectors[group]
        if valid is not None:
            released = released & valid
        self.update.observe(t, scores, thresholds, released, levels)


def replay(
    engine: OnlineEngine,
    origins: Tensor,
    offsets: Tensor,
    chunks: Iterable[tuple[int, int, ScoreDistribution]],
    scores: Tensor,
    valid: Tensor | None,
    out: Tensor,
) -> Tensor:
    """Answer every query in time order, releasing each outcome once its target time passes.

    At each query origin, every outcome whose target time has come is released first;
    then the query is answered. Outcomes that would arrive at the same origin as a query
    are known before it is answered.

    Args:
        engine: The engine to drive.
        origins: ``[Q]`` query origins.
        offsets: ``[G]`` steps after which each release group becomes known.
        chunks: ``(start, stop, distribution)`` for consecutive chunks of queries, in order;
            ``distribution`` covers queries ``start .. stop - 1``.
        scores: ``[Q, *kept]`` scores of the queries' residuals, released once known.
        valid: ``[Q, *kept]`` bool, or None if every score is defined.
        out: ``[Q, *kept]`` tensor that receives the thresholds.

    Returns:
        ``out``.
    """
    origin_list, offset_list = origins.tolist(), offsets.tolist()
    levels = None
    if engine.update.needs_levels:
        levels = torch.empty(out.shape, dtype=torch.float64, device=out.device)
    pending = [0] * len(offset_list)  # first query whose outcome in group g is unreleased
    for start, stop, distribution in chunks:
        for u in range(start, stop):
            for group, offset in enumerate(offset_list):
                while pending[group] < u and origin_list[pending[group]] + offset <= origin_list[u]:
                    t = pending[group]
                    engine.release(
                        t,
                        group,
                        scores[t],
                        out[t],
                        None if valid is None else valid[t],
                        None if levels is None else levels[t],
                    )
                    pending[group] += 1
            current = distribution.select(u - start)
            out[u] = engine.threshold(u, current)
            if levels is not None:
                # Stored now, while the distribution is at hand; the update sees it only
                # when the outcome is released.
                levels[u] = current.level(scores[u])
    return out


__all__ = ["OnlineEngine", "group_selectors", "replay"]

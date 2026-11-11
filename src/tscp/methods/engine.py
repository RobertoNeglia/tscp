"""The online event loop.

An online method answers queries in time order and learns from outcomes as they become
known. :class:`OnlineEngine` turns one outcome into the release mask an update observes;
:func:`replay` runs it over the test queries of an archive, releasing each outcome once
its target time has passed, by the rule of :mod:`tscp.data.availability`. The engine
itself does not depend on archives: code that answers live queries can drive it with
outcomes as they arrive.
"""

from collections.abc import Callable, Iterable, Iterator

import torch
from torch import Tensor

from tscp.online import OnlineUpdate
from tscp.quantile import ScoreDistribution


def release_schedule(
    origins: Tensor, offsets: Tensor
) -> Iterator[tuple[int, list[tuple[int, int]]]]:
    """Yield each query and the earlier outcomes available before it is answered.

    Each released pair is ``(query_position, release_group)``. Outcomes at the same
    time as a query are released first; every outcome is released at most once.
    """
    times, delays = origins.tolist(), offsets.tolist()
    pending = [0] * len(delays)
    for u, origin in enumerate(times):
        released = []
        for group, delay in enumerate(delays):
            while pending[group] < u and times[pending[group]] + delay <= origin:
                released.append((pending[group], group))
                pending[group] += 1
        yield u, released


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
        self._remaining: dict[int, set[int]] = {}
        update.reset(levels, levels.device)

    def threshold(self, t: int, distribution: ScoreDistribution) -> Tensor:
        """The threshold of query ``t``."""
        threshold = self.update.threshold(t, distribution)
        self._remaining[t] = set(range(len(self.selectors)))
        return threshold

    def release(
        self,
        t: int,
        group: int,
        scores: Tensor,
        thresholds: Tensor,
        valid: Tensor | None,
        levels: Tensor | None,
    ) -> None:
        """Pass only this group's observed outcomes, then finish completed queries."""
        remaining = self._remaining.get(t)
        if remaining is None or group not in remaining:
            raise ValueError("the release must belong to an issued, unreleased query group")
        released = self.selectors[group]
        if valid is not None:
            released = released & valid
        known_scores = torch.where(released, scores, 0)
        known_levels = None if levels is None else torch.where(released, levels, 0)
        self.update.observe(t, known_scores, thresholds, released, known_levels)
        remaining.remove(group)
        if not remaining:
            del self._remaining[t]
            self.update.finish_query(t)


def replay(
    engine: OnlineEngine,
    origins: Tensor,
    offsets: Tensor,
    chunks: Iterable[tuple[int, int, ScoreDistribution]] | Callable[[int], ScoreDistribution],
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
        chunks: Consecutive ``(start, stop, distribution)`` chunks, or a callable that
            constructs query ``t``'s distribution after its outcome releases.
        scores: ``[Q, *kept]`` scores of the queries' residuals, released once known.
        valid: ``[Q, *kept]`` bool, or None if every score is defined.
        out: ``[Q, *kept]`` tensor that receives the thresholds.

    Returns:
        ``out``.
    """
    levels = None
    if engine.update.needs_levels:
        levels = torch.empty(out.shape, dtype=torch.float64, device=out.device)
    iterator = None if callable(chunks) else iter(chunks)
    start, stop, distribution = 0, 0, None
    for u, released in release_schedule(origins, offsets):
        for t, group in released:
            engine.release(
                t,
                group,
                scores[t],
                out[t],
                None if valid is None else valid[t],
                None if levels is None else levels[t],
            )
        if callable(chunks):
            current = chunks(u)
        else:
            if u >= stop:
                start, stop, distribution = next(iterator)
                if start != u or stop <= start:
                    raise ValueError("distribution chunks must cover consecutive queries")
            current = distribution.select(u - start)
        target_times = (
            (origins[u] + offsets).reshape(-1, *([1] * (out.ndim - 1))) * engine.selectors
        ).sum(dim=0)
        engine.update.prepare_query(u, origin=origins[u], target_times=target_times)
        out[u] = engine.threshold(u, current)
        if levels is not None:
            levels[u] = current.level(scores[u])
    return out


__all__ = ["OnlineEngine", "group_selectors", "release_schedule", "replay"]

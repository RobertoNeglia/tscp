"""The online-update interface (axis 5)."""

import torch
from torch import Tensor

from tscp.quantile import ScoreDistribution
from tscp.utils import Component


class OnlineUpdate(Component):
    """How the threshold reacts to its own misses.

    The composer answers the test queries in time order, numbered ``t = 0, 1, ...``.
    Before answering a query it passes every outcome that has become known
    (:meth:`observe`); then it asks for the query's threshold (:meth:`threshold`),
    handing over the conformal distribution of its score. Updates that move a level (the
    ACI family) read the distribution's threshold at their level, several levels at once
    if they run experts; updates that move the threshold itself (conformal PID, SAOCP)
    may adjust or ignore it.

    Outcomes arrive late: the outcome of step ``h`` of a query is known ``h`` base steps
    after its origin, so later queries are often answered before earlier outcomes
    arrive. An update that keeps state per query, such as expert thresholds, keys it by
    ``t``.
    """

    #: Whether :meth:`observe` needs the realized level of each outcome (DtACI does).
    needs_levels: bool = False

    def reset(self, alpha: Tensor, device: torch.device) -> None:
        """Start a new pass with target miscoverage ``alpha``, ``[*kept]`` in float64."""
        raise NotImplementedError

    def threshold(self, t: int, distribution: ScoreDistribution) -> Tensor:
        """Return the threshold ``[*kept]`` for query ``t``.

        Args:
            t: Position of the query, counting from 0 in time order.
            distribution: The conformal distribution of query ``t``'s score, of batch
                shape ``[*kept]``.
        """
        raise NotImplementedError

    def observe(
        self,
        t: int,
        scores: Tensor,
        thresholds: Tensor,
        released: Tensor,
        levels: Tensor | None = None,
    ) -> None:
        """Take an outcome of query ``t``.

        Args:
            t: Position of the query the outcome belongs to.
            scores: ``[*kept]`` scores of its residuals.
            thresholds: ``[*kept]`` thresholds it was answered with.
            released: ``[*kept]`` bool, True for the elements whose outcome is known now.
                Per-step outcomes arrive one step at a time; missing targets never arrive.
            levels: ``[*kept]`` realized levels, ``distribution.level(scores)`` under the
                distribution query ``t`` was answered with, if :attr:`needs_levels`;
                otherwise None.
        """
        raise NotImplementedError


__all__ = ["OnlineUpdate"]

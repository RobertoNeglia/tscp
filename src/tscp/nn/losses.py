"""Losses for training learned components."""

import torch
from torch import Tensor

from tscp.metrics import winkler_score
from tscp.regions import Interval


def _mean(values: Tensor, weights: Tensor | None) -> Tensor:
    if weights is None:
        return values.mean()
    weights = weights.to(values.dtype).expand_as(values)
    return (values * weights).sum() / weights.sum().clamp(min=torch.finfo(values.dtype).tiny)


def pinball_loss(
    quantiles: Tensor,
    targets: Tensor,
    level: float | Tensor,
    weights: Tensor | None = None,
) -> Tensor:
    """Mean pinball loss of predicted quantiles, minimized by the true ``level``-quantile.

    ``level`` is the probability of the quantile, such as 0.9 for the 90% quantile, not a
    miscoverage level. With ``e = target - quantile`` the loss is ``level * e`` when
    ``e >= 0`` and ``(level - 1) * e`` otherwise.

    Args:
        quantiles: Predicted quantiles.
        targets: Observed values, broadcastable against ``quantiles``.
        level: Probability in (0, 1), a float or a broadcastable tensor.
        weights: Optional nonnegative weights, such as an observed mask; the loss is then
            their weighted mean, and 0 if every weight is 0.
    """
    error = targets - quantiles
    return _mean(torch.maximum(level * error, (level - 1) * error), weights)


def interval_score(
    lower: Tensor,
    upper: Tensor,
    targets: Tensor,
    alpha: float,
    weights: Tensor | None = None,
) -> Tensor:
    """Mean interval (Winkler) score: the width plus ``2 / alpha`` times the distance missed.

    It is minimized by the central ``1 - alpha`` interval of the targets' distribution,
    and it is :func:`tscp.metrics.winkler_score` averaged, so it trains for what evaluation
    reports.

    Args:
        lower: Lower bounds.
        upper: Upper bounds, broadcastable against ``lower``.
        targets: Observed values, broadcastable against the bounds.
        alpha: Target miscoverage level.
        weights: Optional nonnegative weights; see :func:`pinball_loss`.
    """
    return _mean(winkler_score(Interval(lower, upper), targets, alpha), weights)


__all__ = ["interval_score", "pinball_loss"]

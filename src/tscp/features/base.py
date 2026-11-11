"""The feature-map interface."""

import math

import torch
from torch import Tensor

from tscp.data import ResidualArchive
from tscp.utils import Component

ENCODINGS = ("integer", "cyclic")
"""How cyclic quantities such as the hour of day become features."""


def timeline_of(rows: ResidualArchive) -> ResidualArchive:
    """The archive ``rows`` were taken from; a whole archive is its own timeline."""
    return rows.timeline if rows.timeline is not None else rows


def encode_cycles(values: Tensor, periods: Tensor, encoding: str) -> Tensor:
    """Encode ``[T, K]`` integer positions in cycles of the given ``[K]`` periods.

    ``"integer"`` keeps the positions (``K`` features). ``"cyclic"`` gives the sine and
    cosine of each position's angle in its cycle (``2 K`` features: all sines, then all
    cosines), so that the end of a cycle lies next to its start.
    """
    if encoding not in ENCODINGS:
        raise ValueError(f"encoding must be one of {ENCODINGS}, not {encoding!r}")
    dtype = torch.get_default_dtype()
    if encoding == "integer":
        return values.to(dtype)
    angles = 2 * math.pi * values.to(torch.float64) / periods.to(values.device, torch.float64)
    return torch.cat([angles.sin(), angles.cos()], dim=-1).to(dtype)


class FeatureMap(Component):
    """Turns what is known at each origin into features.

    :meth:`forward` returns ``[T, *element, F]`` for ``T`` rows: one vector of ``F``
    features per row (element ``()``), per series (``(N,)``) or per step and series
    (``(H, N)``). Weights compare these vectors, quantile models regress on them, and
    Mondrian weights group by them.

    The features of a row may use only what is known at its origin: its own forecasts,
    outputs, context and timestamp, and the residuals of its timeline whose target time
    has passed, which :meth:`~tscp.data.ResidualArchive.known_counts` finds. A row
    subset's ``timeline`` holds that history; a whole archive is its own timeline
    (:func:`timeline_of`).
    """

    def forward(self, rows: ResidualArchive) -> Tensor:
        """Return ``[T, *element, F]`` features of the ``T`` rows."""
        raise NotImplementedError

    def valid(self, rows: ResidualArchive) -> Tensor | None:
        """Return ``[T, *element]`` bool, False where a row's features are incomplete.

        Incomplete features, such as a residual history before enough of it is known, are
        0. None means every row's features are complete.
        """
        return None


__all__ = ["ENCODINGS", "FeatureMap", "encode_cycles", "timeline_of"]

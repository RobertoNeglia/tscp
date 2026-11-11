"""Where each origin falls in a cycle of base steps."""

import torch
from torch import Tensor

from tscp.data import ResidualArchive
from tscp.features.base import ENCODINGS, FeatureMap, encode_cycles
from tscp.registry import register


@register("features", "phase")
class Phase(FeatureMap):
    """The position of each origin in a cycle of ``period`` base steps: ``[T, F]``.

    With hourly origins that start at midnight, ``Phase(24)`` is the hour of day; with
    daily origins that start on a Monday, ``Phase(7)`` is the weekday. It needs no
    timestamps. The ``"integer"`` encoding gives ``(origin + offset) mod period``, one
    feature; ``"cyclic"`` gives its sine and cosine, two features.

    Args:
        period: Length of the cycle in base time steps, at least 2.
        offset: Added to the origins before taking the remainder.
        encoding: ``"integer"`` or ``"cyclic"``.
    """

    def __init__(self, period: int, offset: int = 0, encoding: str = "integer"):
        super().__init__()
        if period < 2:
            raise ValueError("period must be at least 2")
        if encoding not in ENCODINGS:
            raise ValueError(f"encoding must be one of {ENCODINGS}, not {encoding!r}")
        self.period = int(period)
        self.offset = int(offset)
        self.encoding = encoding

    def forward(self, rows: ResidualArchive) -> Tensor:
        phase = torch.remainder(rows.origins + self.offset, self.period)
        return encode_cycles(phase[:, None], torch.tensor([self.period]), self.encoding)


__all__ = ["Phase"]

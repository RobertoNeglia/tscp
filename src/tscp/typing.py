"""Type aliases shared across tscp.

Shape conventions used in docstrings:

* ``T``: forecast origins (rows of a residual archive);
* ``H``: horizon steps;
* ``N``: series;
* ``P``: residuals in a calibration pool;
* ``Q``: query origins, one region each;
* ``*kept``: the dimensions of ``[H, N]`` that a score keeps (see
  :attr:`tscp.scores.Score.reduce_dims`).
"""

from typing import TypeAlias

import torch
from torch import Tensor

Alpha: TypeAlias = float | Tensor
"""Miscoverage level: a float, or a tensor that broadcasts against the thresholds."""

Device: TypeAlias = str | torch.device

__all__ = ["Alpha", "Device", "Tensor"]

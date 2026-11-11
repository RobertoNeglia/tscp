"""Axis 6, output: what the user gets, with a membership test and a size."""

from .base import RESIDUAL_DIMS, Region, check_event_dims, move_event_dims, restore_event_dims
from .box import Box
from .interval import Interval

__all__ = [
    "RESIDUAL_DIMS",
    "Box",
    "Interval",
    "Region",
    "check_event_dims",
    "move_event_dims",
    "restore_event_dims",
]

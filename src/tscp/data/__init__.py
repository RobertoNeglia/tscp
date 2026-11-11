"""The data contract: residual archives, split plans and the availability rule."""

from .archive import Graph, ResidualArchive
from .availability import availability_mask, target_times
from .io import load_archive, save_archive
from .splits import SPLIT_NAMES, SplitPlan, chronological_split

__all__ = [
    "SPLIT_NAMES",
    "Graph",
    "ResidualArchive",
    "SplitPlan",
    "availability_mask",
    "chronological_split",
    "load_archive",
    "save_archive",
    "target_times",
]

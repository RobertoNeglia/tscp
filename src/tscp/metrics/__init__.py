"""Metrics by output type, plus the default report.

Region size is always per-context log-volume, summarized per dimension; the mean of raw
volumes is never reported, because the widest regions dominate it.
"""

from .conditional import grouped_coverage, rolling_coverage, worst_window_coverage
from .interval import coverage, coverage_gap, log_width, mean_width, winkler_score
from .report import evaluate

__all__ = [
    "coverage",
    "coverage_gap",
    "evaluate",
    "grouped_coverage",
    "log_width",
    "mean_width",
    "rolling_coverage",
    "winkler_score",
    "worst_window_coverage",
]

"""Checks and fixtures for anyone who builds on tscp's interfaces.

The checks verify that a method or component keeps tscp's contracts: causality, split
discipline, config round trips, devices, and each axis's own rules. tscp checks its own
presets and components with them; a method or component written elsewhere can be checked
the same way, in any test framework::

    from tscp.testing import check_score

    check_score(lambda: MyScore(width=2.0))
"""

from .archives import HOUR, MONDAY, synthetic_archive, with_missing_targets
from .checks import (
    assert_regions_equal,
    check_feature_map,
    check_method,
    check_online_update,
    check_quantile_model,
    check_region,
    check_score,
    check_weights,
)
from .components import MaxAbsScore

__all__ = [
    "HOUR",
    "MONDAY",
    "MaxAbsScore",
    "assert_regions_equal",
    "check_feature_map",
    "check_method",
    "check_online_update",
    "check_quantile_model",
    "check_region",
    "check_score",
    "check_weights",
    "synthetic_archive",
    "with_missing_targets",
]

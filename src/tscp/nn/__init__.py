"""Helpers for learned components: losses, row batches and a reproducible training loop.

A learned component is a :class:`~tscp.utils.Component` with torch parameters, fitted in
its ``fit(train, validation)``: it batches the fit split with :class:`RowBatches` and
fits itself with :func:`train`, which stops early on the validation split. Its fitted
parameters live in ``state_dict()``, so a saved method keeps them.
"""

from .data import RowBatches
from .losses import interval_score, pinball_loss
from .training import TrainingResult, seeded, train

__all__ = ["RowBatches", "TrainingResult", "interval_score", "pinball_loss", "seeded", "train"]

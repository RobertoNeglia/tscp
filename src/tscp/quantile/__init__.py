"""Axis 4, quantile model: how weighted scores become a threshold."""

from .base import QuantileModel
from .weighted import WeightedQuantile, uniform_quantile, weighted_quantile

__all__ = ["QuantileModel", "WeightedQuantile", "uniform_quantile", "weighted_quantile"]

"""Axis 3, calibration weights: which past errors count, and how much."""

from .base import Weights
from .exponential import ExponentialWeights
from .uniform import UniformWeights

__all__ = ["ExponentialWeights", "UniformWeights", "Weights"]

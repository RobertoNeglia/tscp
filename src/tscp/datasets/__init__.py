"""Datasets as residual archives."""

from . import synthetic
from .synthetic import gaussian_control, variance_shift

__all__ = ["gaussian_control", "synthetic", "variance_shift"]

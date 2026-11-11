"""Datasets as residual archives."""

from . import synthetic
from .synthetic import context_scale, gaussian_control, variance_shift

__all__ = ["context_scale", "gaussian_control", "synthetic", "variance_shift"]

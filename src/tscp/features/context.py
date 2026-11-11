"""Columns of the archive's context."""

import torch
from torch import Tensor

from tscp.data import ResidualArchive
from tscp.features.base import FeatureMap
from tscp.registry import register


@register("features", "context")
class ContextColumns(FeatureMap):
    """Columns of the archive's context: ``[T, F]``, or ``[T, N, F]`` for context per series.

    Context holds covariates known at each origin, so a row may always use its own.

    Args:
        names: Names of the columns, from the archive's ``context_names``; None for all.
    """

    def __init__(self, names: list[str] | None = None):
        super().__init__()
        if names is not None and not names:
            raise ValueError("names must name at least one column, or be None for all")
        self.names = None if names is None else list(names)

    def forward(self, rows: ResidualArchive) -> Tensor:
        if rows.context is None:
            raise ValueError("this archive has no context")
        context = rows.context
        if not context.is_floating_point():
            context = context.to(torch.get_default_dtype())
        if self.names is None:
            return context
        unknown = sorted(set(self.names) - set(rows.context_names))
        if unknown:
            raise ValueError(f"the context has no columns {unknown}; it has {rows.context_names}")
        index = torch.tensor([rows.context_names.index(name) for name in self.names])
        return context[..., index.to(context.device)]


__all__ = ["ContextColumns"]

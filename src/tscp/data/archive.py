"""The residual archive: the input of every method."""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import torch
from torch import Tensor

from tscp.data.availability import target_times
from tscp.data.splits import SplitPlan
from tscp.utils.hashing import array_fingerprint


def _int_vector(values, name: str, size: int, device: torch.device) -> Tensor:
    vector = torch.as_tensor(values, device=device)
    if vector.ndim != 1 or vector.numel() != size:
        raise ValueError(f"{name} must be a vector of length {size}")
    if vector.dtype.is_floating_point or vector.dtype == torch.bool:
        raise ValueError(f"{name} must hold integers")
    return vector.to(torch.int64)


@dataclass(frozen=True, eq=False)
class Graph:
    """A graph over the series of an archive, in PyG's edge-list convention.

    Args:
        edge_index: ``[2, E]`` source and target series of each edge.
        num_nodes: Number of series; must equal the archive's ``N``.
        edge_weight: Optional ``[E]`` edge weights.
    """

    edge_index: Tensor
    num_nodes: int
    edge_weight: Tensor | None = None

    def __post_init__(self):
        edge_index = torch.as_tensor(self.edge_index)
        if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.dtype.is_floating_point:
            raise ValueError("edge_index must be an integer tensor of shape [2, E]")
        edge_index = edge_index.to(torch.int64)
        if edge_index.numel() and (edge_index.min() < 0 or edge_index.max() >= self.num_nodes):
            raise ValueError("edge_index refers to a node outside [0, num_nodes)")
        object.__setattr__(self, "edge_index", edge_index)
        if self.edge_weight is not None:
            weight = torch.as_tensor(self.edge_weight, device=edge_index.device)
            if weight.shape != (edge_index.shape[1],):
                raise ValueError("edge_weight must have one entry per edge")
            object.__setattr__(self, "edge_weight", weight)

    def to(self, device: torch.device | str) -> "Graph":
        weight = None if self.edge_weight is None else self.edge_weight.to(device)
        return Graph(self.edge_index.to(device), self.num_nodes, weight)


@dataclass(frozen=True, eq=False, repr=False)
class ResidualArchive:
    """Residuals of one forecaster on one dataset, with what methods need beside them.

    Residuals are truth minus forecast, in the target's original units. tscp never
    standardizes them; a component that rescales fits its scaling on the ``fit`` split.
    Rows are forecast origins. The archive holds the whole timeline and a
    :class:`~tscp.data.SplitPlan`, so the same archive can be split again.

    Args:
        residuals: ``[T, H, N]`` truth minus forecast.
        origins: ``[T]`` forecast origins, in base time steps, strictly increasing.
        steps: ``[H]`` horizon steps, in base time steps, positive and strictly
            increasing. The target of (origin ``s``, step ``h``) is at time ``s + h``.
        splits: Split plan over the rows, or None for a row subset.
        forecasts: ``[T, H, N]`` point forecasts, to map regions back to target units.
        outputs: Other forecaster outputs, each ``[T, H, N, ...]``, such as ``"scale"``.
        context: ``[T, C]`` or ``[T, N, C]`` covariates known at each origin.
        context_names: Names of the ``C`` context columns.
        observed: ``[T, H, N]`` bool; False marks a missing target, whose residual must be
            stored as exactly 0.
        graph: Graph over the ``N`` series.
        series: Names of the ``N`` series.
        provenance: JSON-serializable metadata: dataset, forecaster, horizon, protocol.
    """

    residuals: Tensor
    origins: Tensor
    steps: Tensor
    splits: SplitPlan | None = None
    forecasts: Tensor | None = None
    outputs: Mapping[str, Tensor] = field(default_factory=dict)
    context: Tensor | None = None
    context_names: tuple[str, ...] = ()
    observed: Tensor | None = None
    graph: Graph | None = None
    series: tuple[str, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        assign = object.__setattr__
        residuals = torch.as_tensor(self.residuals)
        if residuals.ndim != 3 or not residuals.is_floating_point():
            raise ValueError("residuals must be a floating tensor of shape [T, H, N]")
        num_origins, horizon, num_series = residuals.shape
        if 0 in residuals.shape:
            raise ValueError("residuals must have at least one origin, step and series")
        device = residuals.device
        assign(self, "residuals", residuals)

        origins = _int_vector(self.origins, "origins", num_origins, device)
        if (origins.diff() <= 0).any():
            raise ValueError("origins must strictly increase")
        assign(self, "origins", origins)
        steps = _int_vector(self.steps, "steps", horizon, device)
        if steps[0] < 1 or (steps.diff() <= 0).any():
            raise ValueError("steps must be positive and strictly increase")
        assign(self, "steps", steps)

        if self.forecasts is not None:
            forecasts = torch.as_tensor(self.forecasts, device=device)
            if forecasts.shape != residuals.shape or not forecasts.is_floating_point():
                raise ValueError("forecasts must be a floating tensor shaped like residuals")
            assign(self, "forecasts", forecasts)

        outputs = {}
        for name, value in self.outputs.items():
            if not name.isidentifier():
                raise ValueError(f"output name {name!r} must be a Python identifier")
            value = torch.as_tensor(value, device=device)
            if tuple(value.shape[:3]) != tuple(residuals.shape):
                raise ValueError(f"output {name!r} must have shape [T, H, N, ...]")
            outputs[name] = value
        assign(self, "outputs", outputs)

        context_names = tuple(self.context_names)
        if self.context is not None:
            context = torch.as_tensor(self.context, device=device)
            leading = {2: (num_origins,), 3: (num_origins, num_series)}.get(context.ndim)
            if leading is None or tuple(context.shape[:-1]) != leading:
                raise ValueError("context must have shape [T, C] or [T, N, C]")
            if context_names and len(context_names) != context.shape[-1]:
                raise ValueError("context_names must name every context column")
            assign(self, "context", context)
        elif context_names:
            raise ValueError("context_names given without context")
        assign(self, "context_names", context_names)

        valid = None
        if self.observed is not None:
            observed = torch.as_tensor(self.observed, device=device)
            if observed.dtype != torch.bool or observed.shape != residuals.shape:
                raise ValueError("observed must be a bool tensor shaped like residuals")
            if (residuals[~observed] != 0).any():
                raise ValueError("unobserved residuals must be stored as exactly 0")
            assign(self, "observed", observed)
            valid = observed
        finite = torch.isfinite(residuals) if valid is None else torch.isfinite(residuals) | ~valid
        if not finite.all():
            raise ValueError("residuals must be finite where observed")

        series = tuple(self.series)
        if series and len(series) != num_series:
            raise ValueError("series must name every series")
        assign(self, "series", series)

        if self.graph is not None:
            if self.graph.num_nodes != num_series:
                raise ValueError("graph.num_nodes must equal the number of series")
            assign(self, "graph", self.graph.to(device))

        if self.splits is not None:
            splits = self.splits if isinstance(self.splits, SplitPlan) else SplitPlan(self.splits)
            splits = splits.to(device)
            splits.validate(origins, steps)
            assign(self, "splits", splits)
        assign(self, "provenance", dict(self.provenance))

    # -- shape and time ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.residuals.shape[0]

    @property
    def num_origins(self) -> int:
        """``T``, the number of forecast origins (rows)."""
        return self.residuals.shape[0]

    @property
    def horizon(self) -> int:
        """``H``, the number of horizon steps."""
        return self.residuals.shape[1]

    @property
    def num_series(self) -> int:
        """``N``, the number of series."""
        return self.residuals.shape[2]

    @property
    def device(self) -> torch.device:
        return self.residuals.device

    @property
    def target_times(self) -> Tensor:
        """``[T, H]`` target time of each (origin, step)."""
        return target_times(self.origins, self.steps)

    # -- rows ----------------------------------------------------------------------------

    def select(self, rows: Tensor) -> "ResidualArchive":
        """Return the archive restricted to ``rows`` (strictly increasing), without splits."""
        rows = torch.as_tensor(rows, device=self.device)
        if rows.ndim != 1 or rows.dtype.is_floating_point or rows.dtype == torch.bool:
            raise ValueError("rows must be a vector of integer indices")
        rows = rows.to(torch.int64)
        if rows.numel() == 0:
            raise ValueError("cannot select zero rows")
        if rows[0] < 0 or rows[-1] >= len(self) or (rows.diff() <= 0).any():
            raise ValueError("rows must be strictly increasing indices into the archive")

        def take(value):
            return None if value is None else value[rows]

        return replace(
            self,
            residuals=self.residuals[rows],
            origins=self.origins[rows],
            splits=None,
            forecasts=take(self.forecasts),
            outputs={name: value[rows] for name, value in self.outputs.items()},
            context=take(self.context),
            observed=take(self.observed),
        )

    def split(self, name: str) -> "ResidualArchive":
        """Return the rows of one split, such as ``archive.split("calibration")``."""
        if self.splits is None:
            raise KeyError("this archive has no split plan; it is already a row subset")
        if name not in self.splits:
            raise KeyError(f"no {name!r} split; this archive has {self.splits.names}")
        return self.select(self.splits[name])

    def to(self, device: torch.device | str) -> "ResidualArchive":
        def move(value):
            return None if value is None else value.to(device)

        return replace(
            self,
            residuals=self.residuals.to(device),
            origins=self.origins.to(device),
            steps=self.steps.to(device),
            splits=None if self.splits is None else self.splits.to(device),
            forecasts=move(self.forecasts),
            outputs={name: value.to(device) for name, value in self.outputs.items()},
            context=move(self.context),
            observed=move(self.observed),
            graph=None if self.graph is None else self.graph.to(device),
        )

    # -- identity ------------------------------------------------------------------------

    def arrays(self) -> dict[str, Tensor]:
        """Every tensor of the archive under a flat name, as stored in ``.npz`` files."""
        arrays = {"residuals": self.residuals, "origins": self.origins, "steps": self.steps}
        if self.splits is not None:
            arrays.update({f"split.{name}": rows for name, rows in self.splits.indices.items()})
        if self.forecasts is not None:
            arrays["forecasts"] = self.forecasts
        arrays.update({f"output.{name}": value for name, value in self.outputs.items()})
        if self.context is not None:
            arrays["context"] = self.context
        if self.observed is not None:
            arrays["observed"] = self.observed
        if self.graph is not None:
            arrays["graph.edge_index"] = self.graph.edge_index
            if self.graph.edge_weight is not None:
                arrays["graph.edge_weight"] = self.graph.edge_weight
        return arrays

    def fingerprint(self) -> str:
        """SHA-256 of every array, names and splits included; metadata is not hashed."""
        return array_fingerprint(self.arrays())

    def __repr__(self) -> str:
        parts = [f"T={self.num_origins}", f"H={self.horizon}", f"N={self.num_series}"]
        if self.splits is not None:
            sizes = ", ".join(f"{name}={size}" for name, size in self.splits.sizes().items())
            parts.append(f"splits=({sizes})")
        extras = [
            name
            for name, value in (
                ("forecasts", self.forecasts),
                ("context", self.context),
                ("observed", self.observed),
                ("graph", self.graph),
            )
            if value is not None
        ]
        extras += [f"output.{name}" for name in self.outputs]
        if extras:
            parts.append(f"with=({', '.join(extras)})")
        return f"ResidualArchive({', '.join(parts)})"


__all__ = ["Graph", "ResidualArchive"]

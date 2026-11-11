"""The residual archive: the input of every method."""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import torch
from torch import Tensor

from tscp.data.availability import known_counts, target_times
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

    A row subset, made by :meth:`select` or :meth:`split`, remembers the archive its rows
    came from as its ``timeline``, with their positions in it as ``rows``. Components read
    the history before a row from there, through :meth:`known_counts`, so nothing they
    compute depends on a residual before its target time.

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
        timeline: For a row subset, the whole archive its rows were taken from, or the part
            of it up to where the subset's holder may look. Set by :meth:`select` and
            :meth:`split`; never part of the fingerprint.
        rows: For a row subset, the positions of its rows in ``timeline``.
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
    timeline: "ResidualArchive | None" = None
    rows: Tensor | None = None

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

        if (self.timeline is None) != (self.rows is None):
            raise ValueError("timeline and rows must be given together")
        if self.timeline is not None:
            timeline = self.timeline
            if timeline.timeline is not None:
                raise ValueError("a timeline must be a whole archive, not a row subset")
            if timeline.device != device:
                raise ValueError("a row subset must be on the device of its timeline")
            rows = _int_vector(self.rows, "rows", num_origins, device)
            if rows[0] < 0 or rows[-1] >= len(timeline) or (rows.diff() <= 0).any():
                raise ValueError("rows must be strictly increasing positions in the timeline")
            same = (
                torch.equal(timeline.origins[rows], origins)
                and torch.equal(timeline.steps, steps)
                and timeline.num_series == num_series
            )
            if not same:
                raise ValueError("rows must match the origins, steps and series of the timeline")
            assign(self, "rows", rows)

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

    def known_counts(self, origins: Tensor) -> Tensor:
        """``[Q, H]``: how many leading rows have a known residual at each origin, per step.

        At ``origins[q]`` the residuals of step ``h`` in rows ``0 .. counts[q, h] - 1`` are
        known, and no others (see :func:`~tscp.data.known_counts`). Unobserved targets
        among them are marked by :attr:`observed`.
        """
        origins = torch.as_tensor(origins, device=self.device)
        return known_counts(self.origins, self.steps, origins)

    # -- rows ----------------------------------------------------------------------------

    def select(self, rows: Tensor) -> "ResidualArchive":
        """Return the archive restricted to ``rows`` (strictly increasing), without splits.

        The result remembers this archive's timeline (this archive itself, unless it is a
        row subset already), with the positions of the selected rows in it.
        """
        rows = torch.as_tensor(rows, device=self.device)
        if rows.ndim != 1 or rows.dtype.is_floating_point or rows.dtype == torch.bool:
            raise ValueError("rows must be a vector of integer indices")
        rows = rows.to(torch.int64)
        if rows.numel() == 0:
            raise ValueError("cannot select zero rows")
        if rows[0] < 0 or rows[-1] >= len(self) or (rows.diff() <= 0).any():
            raise ValueError("rows must be strictly increasing indices into the archive")

        if self.timeline is None:
            timeline, positions = self, rows
        else:
            timeline, positions = self.timeline, self.rows[rows]
        return self._take(rows, timeline=timeline, rows=positions)

    def head(self, num_rows: int) -> "ResidualArchive":
        """Return the first ``num_rows`` rows as a whole archive, without splits."""
        if not 1 <= num_rows <= len(self):
            raise ValueError(f"num_rows must lie between 1 and {len(self)}")
        return self._take(slice(0, num_rows), timeline=None, rows=None)

    def _take(self, index, **fields) -> "ResidualArchive":
        def take(value):
            return None if value is None else value[index]

        return replace(
            self,
            residuals=self.residuals[index],
            origins=self.origins[index],
            splits=None,
            forecasts=take(self.forecasts),
            outputs={name: value[index] for name, value in self.outputs.items()},
            context=take(self.context),
            observed=take(self.observed),
            **fields,
        )

    def split(self, name: str) -> "ResidualArchive":
        """Return the rows of one split, such as ``archive.split("calibration")``.

        The timeline of the result ends with the split's last row: it holds the history
        before and within the split, and nothing after it.
        """
        if self.splits is None:
            raise KeyError("this archive has no split plan; it is already a row subset")
        if name not in self.splits:
            raise KeyError(f"no {name!r} split; this archive has {self.splits.names}")
        rows = self.splits[name]
        return self.head(int(rows[-1]) + 1).select(rows)

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
            timeline=None if self.timeline is None else self.timeline.to(device),
            rows=move(self.rows),
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
        if self.timeline is not None:
            parts.append(f"rows of a timeline with T={self.timeline.num_origins}")
        return f"ResidualArchive({', '.join(parts)})"


__all__ = ["Graph", "ResidualArchive"]

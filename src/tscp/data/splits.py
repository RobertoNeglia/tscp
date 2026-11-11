"""Chronological split plans with no shared target times."""

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

SPLIT_NAMES = ("fit", "validation", "calibration", "test")
"""Split names in chronological order.

``fit`` (and the optional ``validation``) are for trainable components; ``calibration``
supplies the conformal scores; ``test`` is where regions are made and evaluated.
"""


def _rows(values, name: str) -> Tensor:
    rows = torch.as_tensor(values)
    if rows.ndim != 1 or rows.dtype.is_floating_point or rows.dtype == torch.bool:
        raise ValueError(f"split {name!r} must be a vector of integer row indices")
    rows = rows.to(torch.int64)
    if rows.numel() == 0:
        raise ValueError(f"split {name!r} is empty")
    if rows[0] < 0 or (rows.diff() <= 0).any():
        raise ValueError(f"split {name!r} rows must be nonnegative and strictly increasing")
    return rows


@dataclass(frozen=True, eq=False)
class SplitPlan:
    """Named sets of archive rows, in time order, that share no target time.

    Rows are positions in the archive, not origins. :meth:`validate` checks the plan against
    an archive's timeline; :class:`~tscp.data.ResidualArchive` does so on construction.

    Args:
        indices: Row indices per split name; names come from :data:`SPLIT_NAMES`.
    """

    indices: Mapping[str, Tensor]

    def __post_init__(self):
        unknown = sorted(set(self.indices) - set(SPLIT_NAMES))
        if unknown:
            raise ValueError(f"unknown split names {unknown}; use names from {SPLIT_NAMES}")
        ordered = {
            name: _rows(self.indices[name], name) for name in SPLIT_NAMES if name in self.indices
        }
        object.__setattr__(self, "indices", ordered)

    @property
    def names(self) -> tuple[str, ...]:
        """Split names present, in time order."""
        return tuple(self.indices)

    def sizes(self) -> dict[str, int]:
        """Number of rows per split."""
        return {name: rows.numel() for name, rows in self.indices.items()}

    def __getitem__(self, name: str) -> Tensor:
        return self.indices[name]

    def __contains__(self, name: object) -> bool:
        return name in self.indices

    def __iter__(self) -> Iterator[str]:
        return iter(self.indices)

    def to(self, device: torch.device | str) -> "SplitPlan":
        return SplitPlan({name: rows.to(device) for name, rows in self.indices.items()})

    def validate(self, origins: Tensor, steps: Tensor) -> None:
        """Check that the splits fit the timeline, keep time order and share no target time.

        The target of (origin ``s``, step ``h``) is at time ``s + h``. Consecutive splits
        must not share a target time, so the first target of a split must come after the
        last target of the split before it. For steps ``1..H`` this means the first origin
        of a split is at least ``H`` base steps after the last origin of the previous one.

        Raises:
            ValueError: If a split runs past the archive, overlaps or precedes the split
                before it, or shares a target time with it.
        """
        num_rows = origins.numel()
        previous: tuple[str, Tensor] | None = None
        for name, rows in self.indices.items():
            if rows[-1] >= num_rows:
                raise ValueError(f"split {name!r} refers to row {int(rows[-1])}, past the archive")
            if previous is not None:
                previous_name, previous_rows = previous
                last_origin = int(origins[previous_rows[-1]])
                first_origin = int(origins[rows[0]])
                if first_origin <= last_origin:
                    message = f"split {name!r} must start after split {previous_name!r} ends"
                    raise ValueError(message)
                last_target = last_origin + int(steps[-1])
                first_target = first_origin + int(steps[0])
                if first_target <= last_target:
                    needed = last_target - int(steps[0]) + 1
                    raise ValueError(
                        f"splits {previous_name!r} and {name!r} share target times: "
                        f"{previous_name!r} has targets up to time {last_target}, so "
                        f"{name!r} must start at origin {needed} or later, not {first_origin}"
                    )
            previous = (name, rows)


def chronological_split(
    origins: Tensor, steps: Tensor, sizes: Mapping[str, int | float | None]
) -> SplitPlan:
    """Cut consecutive splits in time order, skipping the origins needed between them.

    Args:
        origins: ``[T]`` forecast origins, strictly increasing.
        steps: ``[H]`` horizon steps, strictly increasing.
        sizes: Rows per split, keyed by names from :data:`SPLIT_NAMES` (any order; splits
            are laid out in time order). An ``int`` is a row count, a ``float`` in (0, 1) a
            fraction of all ``T`` rows (rounded down), and ``None``, allowed only for the
            last split, takes every remaining row.

    Returns:
        A validated plan. Between two splits it skips the fewest rows that keep their
        target times apart.

    Raises:
        ValueError: If a size is invalid or the archive has too few origins.
    """
    origins = torch.as_tensor(origins, dtype=torch.int64)
    steps = torch.as_tensor(steps, dtype=torch.int64)
    unknown = sorted(set(sizes) - set(SPLIT_NAMES))
    if unknown:
        raise ValueError(f"unknown split names {unknown}; use names from {SPLIT_NAMES}")
    names = [name for name in SPLIT_NAMES if name in sizes]
    if not names:
        raise ValueError("sizes names no split")
    num_rows = origins.numel()
    indices: dict[str, Tensor] = {}
    start = 0
    for position, name in enumerate(names):
        if indices:
            last_target = origins[next(reversed(indices.values()))[-1]] + steps[-1]
            # First row whose earliest target comes after the previous split's last target.
            start = int(torch.searchsorted(origins, last_target - steps[0], right=True))
        size = sizes[name]
        if size is None:
            if position != len(names) - 1:
                raise ValueError(f"only the last split may take the remaining rows, not {name!r}")
            count = num_rows - start
        elif isinstance(size, bool) or not isinstance(size, int | float):
            raise ValueError(f"size of split {name!r} must be an int, a float or None")
        elif isinstance(size, float):
            if not 0 < size < 1:
                raise ValueError(f"fraction for split {name!r} must lie in (0, 1)")
            count = math.floor(size * num_rows)
        else:
            count = size
        if count < 1 or start + count > num_rows:
            raise ValueError(
                f"not enough origins for split {name!r}: it needs {count} rows from row {start}, "
                f"and the archive has {num_rows} rows"
            )
        indices[name] = torch.arange(start, start + count)
    plan = SplitPlan(indices)
    plan.validate(origins, steps)
    return plan


__all__ = ["SPLIT_NAMES", "SplitPlan", "chronological_split"]

"""Batches of archive rows for training learned components."""

import math
from collections.abc import Iterator

import torch
from torch import Tensor

from tscp.data import ResidualArchive
from tscp.features import FeatureMap


class RowBatches:
    """Batches of the rows of an archive, with their features, for :func:`tscp.nn.train`.

    Each pass yields dicts with:

    - ``"index"``: ``[B]`` positions of the batch's rows within ``rows``;
    - ``"residuals"``: ``[B, H, N]``;
    - ``"observed"``: ``[B, H, N]`` bool, all True when the archive has no mask;
    - ``"features"`` and ``"features_valid"``: ``[B, *element, F]`` and ``[B, *element]``,
      when a feature map is given.

    Features are computed once, when the batches are made, causally from the rows'
    timeline. With ``shuffle`` the order changes on every pass, drawn from a generator
    seeded with ``seed``, so fresh batches with the same seed give the same passes.

    Args:
        rows: The rows, such as ``train`` in a component's ``fit``.
        features: Optional feature map.
        batch_size: Rows per batch.
        shuffle: Shuffle the rows on every pass.
        seed: Seed of the shuffles.
    """

    def __init__(
        self,
        rows: ResidualArchive,
        features: FeatureMap | None = None,
        *,
        batch_size: int = 256,
        shuffle: bool = False,
        seed: int = 0,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.rows = rows
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self._generator = torch.Generator().manual_seed(seed)
        self._observed = rows.observed
        if self._observed is None:
            self._observed = torch.ones_like(rows.residuals, dtype=torch.bool)
        self.features: Tensor | None = None
        self.features_valid: Tensor | None = None
        if features is not None:
            self.features = features(rows)
            valid = features.valid(rows)
            if valid is None:
                valid = torch.ones(self.features.shape[:-1], dtype=torch.bool, device=rows.device)
            self.features_valid = valid

    def __len__(self) -> int:
        return math.ceil(len(self.rows) / self.batch_size)

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        count = len(self.rows)
        if self.shuffle:
            order = torch.randperm(count, generator=self._generator).to(self.rows.device)
        else:
            order = torch.arange(count, device=self.rows.device)
        for start in range(0, count, self.batch_size):
            index = order[start : start + self.batch_size]
            batch = {
                "index": index,
                "residuals": self.rows.residuals[index],
                "observed": self._observed[index],
            }
            if self.features is not None:
                batch["features"] = self.features[index]
                batch["features_valid"] = self.features_valid[index]
            yield batch


__all__ = ["RowBatches"]

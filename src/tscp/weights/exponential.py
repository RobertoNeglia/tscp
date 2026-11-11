"""Weights that decay with age (NexCP)."""

import torch
from torch import Tensor

from tscp.data import ResidualArchive
from tscp.weights.base import Weights


class ExponentialWeights(Weights):
    """``decay ** (query_origin - pool_origin)``: recent errors count more.

    With contiguous origins and an expanding pool, the most recent residual gets
    ``decay``, the one before ``decay**2``, and so on: the fixed weights of NexCP
    (Barber, Candès, Ramdas & Tibshirani, Annals of Statistics 2023). Age counts base time
    steps, so an archive whose origins are ``k`` steps apart decays by ``decay**k`` per row.

    Args:
        decay: Weight ratio per base time step, in (0, 1]; 1 gives equal weights.
    """

    def __init__(self, decay: float = 0.99):
        super().__init__()
        if not 0 < decay <= 1:
            raise ValueError("decay must lie in (0, 1]")
        self.decay = float(decay)

    def forward(self, pool: ResidualArchive, query: ResidualArchive) -> Tensor:
        # Rows after the query get age 0 here; the availability mask removes them anyway.
        age = (query.origins[:, None] - pool.origins[None, :]).clamp(min=0)
        decay = torch.tensor(self.decay, dtype=torch.float64, device=pool.device)
        return decay.pow(age.to(torch.float64))


__all__ = ["ExponentialWeights"]

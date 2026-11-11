import pytest
import torch

from tscp.data import ResidualArchive, chronological_split


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.fixture
def make_archive():
    """Build an archive from residuals ``[T]`` or ``[T, H, N]`` and split sizes."""

    def build(residuals, sizes, **fields) -> ResidualArchive:
        residuals = torch.as_tensor(residuals, dtype=torch.float32)
        if residuals.ndim == 1:
            residuals = residuals[:, None, None]
        num_origins, horizon, _ = residuals.shape
        origins = fields.pop("origins", torch.arange(num_origins))
        steps = fields.pop("steps", torch.arange(1, horizon + 1))
        return ResidualArchive(
            residuals=residuals,
            origins=origins,
            steps=steps,
            splits=chronological_split(origins, steps, sizes),
            **fields,
        )

    return build

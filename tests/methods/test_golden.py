"""Regions of fixed method and archive pairs, compared with stored arrays.

The fixture holds the archives' residuals as well as the regions, because random draws
differ across platforms and torch versions. The regions were made with the first slice of
tscp; any change to them must be deliberate. To store new arrays after such a change, run
``python tests/methods/test_golden.py NEW_PATH.npz`` and replace the fixture.
"""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from tscp.data import ResidualArchive, chronological_split
from tscp.datasets import gaussian_control, variance_shift
from tscp.datasets.synthetic import DEFAULT_SIZES
from tscp.methods import ConformalPredictor, get_method
from tscp.online import ACIUpdate
from tscp.scores import AbsoluteScore
from tscp.sources import ExpandingSource, HoldoutSource
from tscp.weights import ExponentialWeights

FIXTURE = Path(__file__).parents[1] / "fixtures" / "golden_regions.npz"


def with_missing_targets(archive, seed: int):
    """Mark about 5% of the targets as unobserved."""
    generator = torch.Generator().manual_seed(seed)
    observed = torch.rand(archive.residuals.shape, generator=generator) > 0.05
    return replace(archive, residuals=archive.residuals * observed, observed=observed)


GENERATORS = {  # how the stored archives were drawn
    "shift_h1": lambda: variance_shift(horizon=1, num_origins=400, seed=11),
    "shift_h3_n2": lambda: variance_shift(horizon=3, num_origins=300, num_series=2, seed=12),
    "control_h2_n2_missing": lambda: with_missing_targets(
        gaussian_control(horizon=2, num_origins=300, num_series=2, seed=13), seed=14
    ),
}

METHODS = {
    "split": lambda: get_method("split"),
    "nexcp": lambda: get_method("nexcp", decay=0.98),
    "aci": lambda: get_method("aci", gamma=0.01),
    "holdout_exponential": lambda: ConformalPredictor(
        AbsoluteScore(), source=HoldoutSource(), weights=ExponentialWeights(0.99)
    ),
    "expanding_uniform": lambda: ConformalPredictor(AbsoluteScore(), source=ExpandingSource()),
    "holdout_aci": lambda: ConformalPredictor(
        AbsoluteScore(), source=HoldoutSource(), update=ACIUpdate(0.02)
    ),
}

CASES = [(archive, method) for archive in GENERATORS for method in METHODS]


def stored_archive(stored, name: str) -> ResidualArchive:
    """Rebuild an archive from its stored residuals, with the default splits."""
    residuals = torch.from_numpy(stored[f"input/{name}/residuals"])
    key = f"input/{name}/observed"
    observed = torch.from_numpy(stored[key]) if key in stored.files else None
    num_origins, horizon, _ = residuals.shape
    origins, steps = torch.arange(num_origins), torch.arange(1, horizon + 1)
    return ResidualArchive(
        residuals=residuals,
        origins=origins,
        steps=steps,
        splits=chronological_split(origins, steps, DEFAULT_SIZES),
        observed=observed,
    )


def upper_bounds(archive: ResidualArchive, method_name: str) -> np.ndarray:
    region = METHODS[method_name]().fit(archive).calibrate(archive).predict(archive)
    assert torch.equal(region.lower, -region.upper)  # absolute scores give symmetric intervals
    return region.upper.numpy()


@pytest.mark.parametrize(("archive_name", "method_name"), CASES)
def test_regions_match_the_stored_arrays(archive_name, method_name):
    with np.load(FIXTURE) as stored:
        archive = stored_archive(stored, archive_name)
        expected = stored[f"{archive_name}/{method_name}"]
    np.testing.assert_array_equal(upper_bounds(archive, method_name), expected)


if __name__ == "__main__":
    path = Path(sys.argv[1])
    arrays = {}
    for name, generate in GENERATORS.items():
        archive = generate()
        arrays[f"input/{name}/residuals"] = archive.residuals.numpy()
        if archive.observed is not None:
            arrays[f"input/{name}/observed"] = archive.observed.numpy()
        for method in METHODS:
            arrays[f"{name}/{method}"] = upper_bounds(archive, method)
    with path.open("xb") as handle:  # never overwrite
        np.savez_compressed(handle, **arrays)
    print(f"wrote {len(arrays)} arrays to {path}")

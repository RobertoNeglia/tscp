"""Multi-step outcomes are released one step at a time, as their targets arrive."""

from dataclasses import replace

import torch

from tscp.datasets import gaussian_control
from tscp.methods import ConformalPredictor
from tscp.online import ACIUpdate
from tscp.scores import AbsoluteScore
from tscp.sources import ExpandingSource, HoldoutSource

SPIKE = 30  # test position whose residuals differ between the two runs


def two_runs(method_factory):
    # A large pool keeps ACI's level above 1 / (n + 1), so the spike's region is bounded.
    archive = gaussian_control(
        horizon=2, num_origins=800, sizes={"calibration": 500, "test": None}, seed=3
    )
    row = archive.splits["test"][SPIKE]
    quiet, spiked = archive.residuals.clone(), archive.residuals.clone()
    quiet[row] = 0.0  # always covered
    spiked[row] = 1e6  # missed whenever the region is bounded
    return [
        method_factory().calibrate(run).predict(run)
        for run in (replace(archive, residuals=quiet), replace(archive, residuals=spiked))
    ]


def first_difference(regions, step: int) -> int:
    differs = (regions[0].upper[:, step] != regions[1].upper[:, step]).any(dim=-1)
    return int(differs.nonzero()[0])


def test_online_updates_hear_step_h_after_h_steps():
    regions = two_runs(
        lambda: ConformalPredictor(AbsoluteScore(), source=HoldoutSource(), update=ACIUpdate(0.01))
    )
    assert torch.isfinite(regions[1].upper[SPIKE]).all()  # the spike was a miss
    assert first_difference(regions, step=0) == SPIKE + 1
    assert first_difference(regions, step=1) == SPIKE + 2


def test_expanding_pools_add_step_h_after_h_steps():
    regions = two_runs(lambda: ConformalPredictor(AbsoluteScore(), source=ExpandingSource()))
    assert first_difference(regions, step=0) == SPIKE + 1
    assert first_difference(regions, step=1) == SPIKE + 2

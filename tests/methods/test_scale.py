"""The composer's costs: one sort per prediction, bounded weight chunks, long archives."""

import pytest
import torch

import tscp.methods.base as composer
from tscp.datasets import variance_shift
from tscp.methods import ACI, ConformalPredictor, get_method
from tscp.scores import AbsoluteScore
from tscp.sources import ExpandingSource
from tscp.weights import ExponentialWeights

DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    ),
]


def test_the_pool_is_sorted_once_per_prediction(monkeypatch):
    archive = variance_shift(horizon=2, num_origins=400)
    method = ACI(gamma=0.01, source=ExpandingSource()).calibrate(archive)
    calls = []
    sort = torch.sort

    def counting_sort(*args, **kwargs):
        calls.append(1)
        return sort(*args, **kwargs)

    monkeypatch.setattr(torch, "sort", counting_sort)
    method.predict(archive)
    assert len(calls) == 1  # not once per test origin


def test_weight_chunks_stay_under_the_element_cap(monkeypatch):
    monkeypatch.setattr(composer, "_MAX_ELEMENTS", 5000)
    archive = variance_shift(horizon=2, num_origins=400, num_series=2)
    method = ConformalPredictor(
        AbsoluteScore(), source=ExpandingSource(), weights=ExponentialWeights(0.99)
    )
    sizes = []
    weights = composer.ConformalPredictor._weights

    def recording_weights(self, *args, **kwargs):
        values = weights(self, *args, **kwargs)
        sizes.append(values.expand(*values.shape[:2], *args[0].kept).numel())
        return values

    monkeypatch.setattr(composer.ConformalPredictor, "_weights", recording_weights)
    method.calibrate(archive).predict(archive)
    assert len(sizes) > 1 and max(sizes) <= 5000


@pytest.mark.slow
@pytest.mark.parametrize("device", DEVICES)
def test_a_long_archive_runs_online(device):
    archive = variance_shift(horizon=1, num_origins=20_000).to(device)
    method = get_method("aci", gamma=0.005).to(device)
    region = method.calibrate(archive).predict(archive)
    assert len(region) == len(archive.split("test")) and torch.isfinite(region.upper).any()

"""Checks every registered preset must pass."""

import json
from dataclasses import replace

import pytest
import torch

from tscp.datasets import gaussian_control, variance_shift
from tscp.methods import ConformalPredictor, SplitCP, get_method, method_map
from tscp.scores import AbsoluteScore
from tscp.utils import from_config

PRESETS = sorted(method_map)
DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    ),
]


def run(method: ConformalPredictor, archive):
    return method.fit(archive).calibrate(archive).predict(archive)


@pytest.mark.parametrize("name", PRESETS)
@pytest.mark.parametrize("horizon", [1, 3])
def test_regions_ignore_residuals_not_yet_known(name, horizon):
    """Changing residuals whose target time is after origin t leaves regions up to t alone."""
    archive = variance_shift(horizon=horizon, num_origins=400, seed=1)
    position = 60  # a test origin
    origin = archive.origins[archive.splits["test"][position]]
    future = archive.target_times > origin
    changed = archive.residuals.clone()
    changed[future] = 1e3 * torch.randn(int(future.sum()), archive.num_series)
    before = run(get_method(name), archive)
    after = run(get_method(name), replace(archive, residuals=changed))
    assert torch.equal(before.lower[: position + 1], after.lower[: position + 1])
    assert torch.equal(before.upper[: position + 1], after.upper[: position + 1])


class SpyScore(AbsoluteScore):
    """An absolute score that records the rows its fit step sees."""

    def __init__(self):
        super().__init__()
        self.seen = None

    def fit(self, archive):
        self.seen = archive.origins.clone()
        return self


@pytest.mark.parametrize("name", PRESETS)
def test_trainable_components_see_only_the_fit_split(name):
    archive = gaussian_control(horizon=2, num_origins=300)
    spy = SpyScore()
    method = get_method(name, score=spy)
    assert method.is_trainable
    method.fit(archive)
    assert torch.equal(spy.seen, archive.split("fit").origins)


def test_fitting_without_a_fit_split_raises():
    archive = gaussian_control(horizon=1, sizes={"calibration": 100, "test": None})
    with pytest.raises(ValueError, match="no 'fit' split"):
        SplitCP(score=SpyScore()).fit(archive)
    SplitCP().fit(archive)  # nothing to fit, nothing needed


@pytest.mark.parametrize("name", PRESETS)
def test_config_round_trip_rebuilds_the_preset(name):
    archive = variance_shift(horizon=2, num_origins=300)
    method = get_method(name, alpha=0.2)
    config = json.loads(json.dumps(method.to_config()))
    rebuilt = from_config(config)
    assert type(rebuilt) is type(method) and rebuilt.to_config() == method.to_config()
    by_name = get_method(name, **method.get_config())
    for other in (rebuilt, by_name):
        assert torch.equal(run(other, archive).upper, run(method, archive).upper)


@pytest.mark.parametrize("name", PRESETS)
@pytest.mark.parametrize("device", DEVICES)
def test_presets_run_on_the_device_of_their_data(name, device):
    archive = variance_shift(horizon=2, num_origins=300)
    region = run(get_method(name).to(device), archive.to(device))
    assert region.upper.device.type == device
    reference = run(get_method(name), archive)
    assert torch.equal(region.upper.cpu(), reference.upper)


def test_get_method_checks_its_arguments():
    assert isinstance(get_method("NexCP"), method_map["nexcp"])
    with pytest.raises(ValueError, match="unknown method"):
        get_method("nope")
    with pytest.raises(TypeError, match="does not take"):
        get_method("split", decay=0.9)
    assert isinstance(get_method("split", decay=0.9, ignore_unknown=True), SplitCP)


def test_lifecycle_errors():
    archive = gaussian_control(horizon=1)
    with pytest.raises(RuntimeError, match="calibrate"):
        SplitCP().predict(archive)
    other = gaussian_control(horizon=1, sizes={"fit": 0.1, "calibration": 0.4, "test": None})
    with pytest.raises(ValueError, match="archive that calibrate"):
        SplitCP().calibrate(archive).predict(other)
    with pytest.raises(TypeError, match="score must be a Score"):
        ConformalPredictor(score="absolute")
    with pytest.raises(ValueError, match="alpha"):
        SplitCP(alpha=1.0)

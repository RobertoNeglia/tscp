"""Checks every registered preset must pass."""

import json
from dataclasses import replace

import pytest
import torch

from tscp.datasets import gaussian_control, variance_shift
from tscp.methods import ConformalPredictor, SplitCP, get_method, method_map
from tscp.regions import Box
from tscp.scores import AbsoluteScore, CQRScore, NormalizedScore, Score, SignedScore
from tscp.utils import from_config

PRESETS = sorted(method_map)
DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    ),
]


class MaxAbsScore(Score):
    """The largest absolute residual over the horizon: a joint score, one box per series."""

    reduce_dims = ("step",)

    def forward(self, rows):
        return rows.residuals.abs().amax(dim=1)

    def region(self, threshold, rows):
        bound = threshold.unsqueeze(-1).expand(*threshold.shape, rows.horizon)
        return Box(-bound, bound, event_dims=("step",))


SCORES = {
    "absolute": AbsoluteScore,
    "signed": lambda: SignedScore(lower_share=0.3),
    "normalized": lambda: NormalizedScore("scale"),
    "normalized_fitted": NormalizedScore,
    "cqr": CQRScore,
    "cqr_asymmetric": lambda: CQRScore(asymmetric=True),
    "max_abs": MaxAbsScore,
}


def scored_archive(**kwargs):
    """A synthetic archive with every output the scores in SCORES read."""
    archive = variance_shift(**kwargs)
    outputs = {
        "scale": 0.5 + torch.rand_like(archive.residuals),
        "lower": archive.forecasts - 1.0,
        "upper": archive.forecasts + 1.0,
    }
    return replace(archive, outputs=outputs)


def run(method: ConformalPredictor, archive):
    return method.fit(archive).calibrate(archive).predict(archive)


def assert_regions_equal(first, second):
    assert type(first) is type(second) and first.event_dims == second.event_dims
    tensors, others = first.tensors(), second.tensors()
    assert tensors.keys() == others.keys()
    for name, value in tensors.items():
        assert torch.equal(value, others[name]), name


@pytest.mark.parametrize("name", PRESETS)
@pytest.mark.parametrize("score", sorted(SCORES))
@pytest.mark.parametrize("horizon", [1, 3])
def test_regions_ignore_residuals_not_yet_known(name, score, horizon):
    """Changing residuals whose target time is after origin t leaves regions up to t alone."""
    archive = scored_archive(horizon=horizon, num_origins=400, seed=1)
    position = 60  # a test origin
    origin = archive.origins[archive.splits["test"][position]]
    future = archive.target_times > origin
    changed = archive.residuals.clone()
    changed[future] = 1e3 * torch.randn(int(future.sum()), archive.num_series)
    before = run(get_method(name, score=SCORES[score]()), archive)
    after = run(get_method(name, score=SCORES[score]()), replace(archive, residuals=changed))
    assert_regions_equal(before[: position + 1], after[: position + 1])


class SpyScore(AbsoluteScore):
    """An absolute score that records what its fit step sees."""

    def __init__(self):
        super().__init__()
        self.seen = None

    def fit(self, train, validation=None):
        self.seen = (train, validation)
        return self


@pytest.mark.parametrize("name", PRESETS)
def test_trainable_components_see_only_the_fit_split(name):
    sizes = {"fit": 60, "validation": 30, "calibration": 90, "test": None}
    archive = gaussian_control(horizon=2, num_origins=300, sizes=sizes)
    spy = SpyScore()
    method = get_method(name, score=spy)
    assert method.is_trainable
    method.fit(archive)
    train, validation = spy.seen
    for split, seen in (("fit", train), ("validation", validation)):
        rows = archive.splits[split]
        assert torch.equal(seen.origins, archive.origins[rows])
        assert len(seen.timeline) == int(rows[-1]) + 1  # nothing after the split is in reach

    spy = SpyScore()
    get_method(name, score=spy).fit(gaussian_control(horizon=2, num_origins=300))
    assert spy.seen[1] is None  # no validation split, no validation rows


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
        assert_regions_equal(run(other, archive), run(method, archive))


@pytest.mark.parametrize("name", PRESETS)
@pytest.mark.parametrize("score", sorted(SCORES))
def test_config_round_trip_keeps_the_score(name, score):
    archive = scored_archive(horizon=2, num_origins=300)
    method = get_method(name, score=SCORES[score]())
    rebuilt = from_config(json.loads(json.dumps(method.to_config())))
    assert type(rebuilt.score) is type(method.score)
    assert_regions_equal(run(rebuilt, archive), run(method, archive))


@pytest.mark.parametrize("name", PRESETS)
@pytest.mark.parametrize("score", sorted(SCORES))
@pytest.mark.parametrize("device", DEVICES)
def test_presets_run_on_the_device_of_their_data(name, score, device):
    archive = scored_archive(horizon=2, num_origins=300)
    region = run(get_method(name, score=SCORES[score]()).to(device), archive.to(device))
    assert all(value.device.type == device for value in region.tensors().values())
    assert_regions_equal(region.to("cpu"), run(get_method(name, score=SCORES[score]()), archive))


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

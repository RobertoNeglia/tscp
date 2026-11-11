import json

import pytest

from tscp.methods import ConformalPredictor, NexCP, compose
from tscp.online import ACIUpdate
from tscp.scores import CQRScore, SignedScore
from tscp.sources import ExpandingSource
from tscp.testing import assert_regions_equal, synthetic_archive
from tscp.utils import from_config
from tscp.weights import ExponentialWeights


def run(method, archive):
    return method.fit(archive).calibrate(archive).predict(archive)


def test_a_preset_spec_builds_the_preset():
    method = compose({"method": "nexcp", "decay": 0.9, "alpha": 0.2})
    assert isinstance(method, NexCP) and method.weights.decay == 0.9 and method.alpha == 0.2
    method = compose({"method": "aci", "score": {"name": "signed", "lower_share": 0.3}})
    assert isinstance(method.score, SignedScore) and method.score.lower_share == 0.3


def test_an_axis_spec_builds_what_the_constructor_builds():
    spec = {
        "score": "signed",
        "source": "expanding",
        "weights": {"name": "exponential", "decay": 0.98},
        "update": {"name": "aci", "gamma": 0.01},
        "alpha": 0.15,
    }
    built = ConformalPredictor(
        SignedScore(),
        source=ExpandingSource(),
        weights=ExponentialWeights(0.98),
        update=ACIUpdate(0.01),
        alpha=0.15,
    )
    composed = compose(spec)
    assert composed.to_config() == built.to_config()
    archive = synthetic_archive(horizon=2, num_origins=200)
    assert_regions_equal(run(composed, archive), run(built, archive))


def test_components_can_be_configs_instances_or_name_their_kind():
    weights = ExponentialWeights(0.9)
    method = compose({"score": SignedScore().to_config(), "weights": weights})
    assert isinstance(method.score, SignedScore) and method.weights is weights
    method = compose(
        {"method": "split", "score": {"kind": "score", "name": "cqr", "asymmetric": True}}
    )
    assert isinstance(method.score, CQRScore) and method.score.asymmetric


def test_a_composed_method_round_trips_through_its_config():
    method = compose({"score": "absolute", "update": "aci"})
    rebuilt = from_config(json.loads(json.dumps(method.to_config())))
    archive = synthetic_archive(horizon=1, num_origins=200)
    assert_regions_equal(run(rebuilt, archive), run(method, archive))


@pytest.mark.parametrize(
    ("spec", "error", "message"),
    [
        ({"weights": "uniform"}, ValueError, "needs a 'score'"),
        ({"score": "absolute", "pool": "holdout"}, ValueError, "unknown keys"),
        ({"score": "nope"}, ValueError, "unknown score"),
        ({"score": 3}, ValueError, "cannot build"),
        ({"method": "split", "decay": 0.9}, TypeError, "does not take"),
    ],
)
def test_bad_specs_raise(spec, error, message):
    with pytest.raises(error, match=message):
        compose(spec)

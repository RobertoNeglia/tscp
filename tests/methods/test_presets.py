"""Common method contracts on each preset's supported configurations."""

import inspect
import json

import pytest
import torch

from tscp.datasets import gaussian_control, variance_shift
from tscp.features import Phase
from tscp.imports import has_module
from tscp.methods import ConformalPredictor, SplitCP, get_method, method_map
from tscp.online import ThetaScorecaster
from tscp.scores import AbsoluteScore, CQRScore, NormalizedScore, SignedScore
from tscp.testing import MaxAbsScore, assert_regions_equal, check_method
from tscp.utils import from_config

PRESETS = sorted(
    name for name, cls in method_map.items() if "score" in inspect.signature(cls).parameters
)
COUPLED = sorted(set(method_map) - set(PRESETS))


def preset(name, **kwargs):
    defaults = {
        "windowed": {"window_size": 40},
        "mondrian": {"groups": Phase(3)},
        "pid": {"scorecaster": ThetaScorecaster(min_history=4, window=12, refit_every=20)},
    }
    if name == "pid" and not has_module("statsmodels"):
        pytest.skip("the pid extra is not installed")
    return get_method(name, **(defaults.get(name, {}) | kwargs))


SCORES = {
    "absolute": AbsoluteScore,
    "signed": lambda: SignedScore(lower_share=0.3),
    "normalized": lambda: NormalizedScore("scale"),
    "normalized_fitted": NormalizedScore,
    "cqr": CQRScore,
    "cqr_asymmetric": lambda: CQRScore(asymmetric=True),
    "max_abs": MaxAbsScore,
}


def run(method: ConformalPredictor, archive):
    return method.fit(archive).calibrate(archive).predict(archive)


@pytest.mark.parametrize("name", PRESETS)
@pytest.mark.parametrize("score", sorted(SCORES))
def test_presets_keep_the_method_contracts(name, score):
    from tscp.testing import synthetic_archive

    archives = None
    if name == "pid":
        archives = [
            synthetic_archive(horizon=1, num_origins=60),
            synthetic_archive(horizon=3, num_series=2, num_origins=60, missing=0.05),
        ]
    check_method(lambda: preset(name, score=SCORES[score]()), archives=archives)


class SpyScore(AbsoluteScore):
    """An absolute score that records what its fit step sees."""

    def __init__(self):
        super().__init__()
        self.seen = None

    def fit(self, train, validation=None):
        self.seen = (train, validation)
        return self


@pytest.mark.parametrize("name", PRESETS)
def test_trainable_components_see_only_the_fitting_splits(name):
    sizes = {"fit": 60, "validation": 30, "calibration": 90, "test": None}
    archive = gaussian_control(horizon=2, num_origins=300, sizes=sizes)
    spy = SpyScore()
    method = preset(name, score=spy)
    assert method.is_trainable
    method.fit(archive)
    train, validation = spy.seen
    for split, seen in (("fit", train), ("validation", validation)):
        rows = archive.splits[split]
        assert torch.equal(seen.origins, archive.origins[rows])
        assert len(seen.timeline) == int(rows[-1]) + 1  # nothing after the split is in reach

    spy = SpyScore()
    preset(name, score=spy).fit(gaussian_control(horizon=2, num_origins=300))
    assert spy.seen[1] is None  # no validation split, no validation rows


def test_fitting_without_a_fit_split_raises():
    archive = gaussian_control(horizon=1, sizes={"calibration": 100, "test": None})
    with pytest.raises(ValueError, match="no 'fit' split"):
        SplitCP(score=SpyScore()).fit(archive)
    SplitCP().fit(archive)  # nothing to fit, nothing needed


@pytest.mark.parametrize("name", PRESETS)
def test_presets_rebuild_by_name_from_their_config(name):
    archive = variance_shift(horizon=2, num_origins=300)
    method = preset(name, alpha=0.2)
    rebuilt = from_config(json.loads(json.dumps(method.to_config())))
    assert type(rebuilt) is type(method)
    by_name = get_method(name, **method.get_config())
    for other in (rebuilt, by_name):
        assert_regions_equal(run(other, archive), run(method, archive))


@pytest.mark.parametrize("name", COUPLED)
def test_coupled_presets_keep_the_common_contracts(name):
    from dataclasses import replace

    from tscp.data import from_bootstrap_predictions
    from tscp.testing import synthetic_archive

    archive = synthetic_archive(horizon=1, num_origins=70, missing=0.05, seed=4)
    if name == "agaci":

        def factory():
            return get_method(name, endpoint_bounds=[-100, 100], gammas=[0, 0.01, 0.1])
    elif name == "enbpi":
        members = archive.forecasts[..., None].expand(-1, -1, -1, 3).clone()
        membership = torch.eye(3, dtype=torch.bool)[
            :, torch.arange(len(archive.splits["calibration"])) % 3
        ]
        archive = from_bootstrap_predictions(
            archive.targets(),
            members,
            membership,
            splits=archive.splits,
            origins=archive.origins,
            steps=archive.steps,
        )

        def factory():
            return get_method(name)
    elif name == "spci":
        if not has_module("sklearn_quantile"):
            pytest.skip("the forest extra is not installed")

        def factory():
            return get_method(name, lags=2, window_size=12, n_estimators=2)
    else:
        pytest.fail(f"registered preset {name!r} needs a supported contract scenario")
    # The same causality, split, persistence, reset and device checks apply to coupled
    # methods; their published algorithms support a narrower archive/score contract.
    check_method(
        factory, archives=[archive, replace(archive, observed=None, residuals=archive.residuals)]
    )


@pytest.mark.parametrize("name", COUPLED)
def test_coupled_presets_reject_arbitrary_scores(name):
    args = {"agaci": {"endpoint_bounds": [-100, 100]}, "spci": {"lags": 2, "window_size": 12}}
    with pytest.raises(TypeError, match="does not take"):
        get_method(name, **args.get(name, {}), score=AbsoluteScore())


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

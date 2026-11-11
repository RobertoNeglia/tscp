"""AgACI's endpoint aggregation, reference trace and prediction contracts."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from tscp.data import ResidualArchive
from tscp.methods.agaci import DEFAULT_GAMMAS, AgACI, _BernsteinAggregation
from tscp.sources import ExpandingSource, HoldoutSource
from tscp.testing import assert_regions_equal, check_method, synthetic_archive

REFERENCE = Path(__file__).parents[1] / "fixtures" / "agaci_reference.json"


def reference_archive():
    fixture = json.loads(REFERENCE.read_text())
    residuals = torch.tensor(fixture["calibration"] + fixture["residuals"], dtype=torch.float64)
    forecasts = torch.tensor(
        [0.0] * len(fixture["calibration"]) + fixture["forecasts"], dtype=torch.float64
    )
    archive = ResidualArchive.from_forecasts(
        residuals + forecasts,
        forecasts,
        sizes={"calibration": len(fixture["calibration"]), "test": None},
    )
    return fixture, archive


def test_agaci_matches_pinned_r_oracle():
    fixture, archive = reference_archive()
    method = AgACI(fixture["endpoint_bounds"], gammas=fixture["gammas"], alpha=fixture["alpha"])
    result = method.calibrate(archive).predict(archive).to_target(archive.split("test").forecasts)
    for name in ("lower", "upper"):
        expected = torch.tensor(fixture[name], dtype=torch.float64).reshape(-1, 1, 1)
        torch.testing.assert_close(getattr(result, name), expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("side", ["lower", "upper"])
def test_boa_weights_and_predictions_match_pinned_r_oracle(side):
    fixture, _ = reference_archive()
    tau = fixture["alpha"] / 2 if side == "lower" else 1 - fixture["alpha"] / 2
    learner = _BernsteinAggregation(len(fixture["gammas"]), tau, torch.device("cpu"))
    experts = torch.tensor(fixture[f"experts_{side}"], dtype=torch.float64)
    expected = torch.tensor(fixture[side], dtype=torch.float64)
    weights = torch.tensor(fixture[f"weights_{side}"], dtype=torch.float64)
    targets = torch.tensor(fixture["forecasts"], dtype=torch.float64) + torch.tensor(
        fixture["residuals"], dtype=torch.float64
    )
    for t, advice in enumerate(experts):
        torch.testing.assert_close(learner.weights(), weights[t], atol=1e-12, rtol=1e-12)
        prediction = learner.predict(advice)
        torch.testing.assert_close(prediction, expected[t], atol=1e-12, rtol=1e-12)
        learner.observe(advice, prediction, targets[t])


def test_default_expert_grid_includes_fixed_expert():
    assert len(DEFAULT_GAMMAS) == 30
    assert DEFAULT_GAMMAS[0] == 0
    assert DEFAULT_GAMMAS[-1] == 0.09


def test_single_fixed_expert_preserves_finite_endpoints_outside_replacements():
    _, archive = reference_archive()
    method = AgACI((-1, 1), gammas=[0], source=HoldoutSource(), alpha=0.2)
    region = method.calibrate(archive).predict(archive)
    torch.testing.assert_close(region.lower, torch.full_like(region.lower, -2.5))
    torch.testing.assert_close(region.upper, torch.full_like(region.upper, 2.5))
    assert bool((region.to_target(archive.split("test").forecasts).lower > 1).all())


def test_unbounded_expert_uses_target_coordinate_replacements():
    forecasts = torch.arange(5, dtype=torch.float64)
    archive = ResidualArchive.from_forecasts(
        forecasts, forecasts, sizes={"calibration": 1, "test": None}
    )
    region = (
        AgACI((-10, 10), gammas=[0], source=HoldoutSource()).calibrate(archive).predict(archive)
    )
    target = region.to_target(archive.split("test").forecasts)
    assert bool((target.lower == -10).all())
    assert bool((target.upper == 10).all())


def test_level_at_or_above_one_gives_the_papers_zero_radius():
    archive = ResidualArchive.from_forecasts(
        torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0]),
        torch.zeros(5),
        sizes={"calibration": 2, "test": None},
    )
    region = (
        AgACI((-100, 100), gammas=[1], source=HoldoutSource(), alpha=0.8)
        .calibrate(archive)
        .predict(archive)
    )
    assert float(region.upper[0]) == 1
    assert bool((region.lower[1:] == 0).all())
    assert bool((region.upper[1:] == 0).all())


def test_aggregation_can_be_asymmetric():
    fixture, archive = reference_archive()
    result = (
        AgACI(fixture["endpoint_bounds"], gammas=fixture["gammas"], alpha=fixture["alpha"])
        .calibrate(archive)
        .predict(archive)
    )
    assert bool((result.lower + result.upper).abs().gt(0.01).any())


def test_identical_experts_remain_uniform_with_zero_regret():
    learner = _BernsteinAggregation(4, 0.1, torch.device("cpu"))
    for value in (0.0, 0.0, 2.0, -1.0, 3.0):
        advice = torch.full((4,), value, dtype=torch.float64)
        prediction = learner.predict(advice)
        learner.observe(advice, prediction, prediction)
        torch.testing.assert_close(learner.weights(), torch.full((4,), 0.25, dtype=torch.float64))


def test_pinned_gradient_tie_uses_the_left_pinball_subgradient():
    learner = _BernsteinAggregation(2, 0.25, torch.device("cpu"))
    experts = torch.tensor([-1.0, 1.0], dtype=torch.float64)
    prediction = learner.predict(experts)
    learner.observe(experts, prediction, prediction)
    # At y = prediction, the OPERA gradient is -tau; it favors the larger endpoint.
    assert learner.weights()[1] > learner.weights()[0]


def test_method_contracts_and_missing_outcomes():
    archives = [
        synthetic_archive(horizon=1, num_origins=120, seed=3),
        synthetic_archive(horizon=1, num_origins=120, missing=0.15, seed=4),
    ]
    check_method(lambda: AgACI((-100, 100), gammas=[0, 0.005, 0.05]), archives=archives)


def test_repeated_prediction_resets_state_and_chunking_is_irrelevant():
    _, archive = reference_archive()
    method = AgACI((-10, 20), gammas=[0, 0.05, 0.6], alpha=0.2, chunk_size=1)
    method.calibrate(archive)
    before = {name: value.clone() for name, value in method.state_dict().items()}
    first = method.predict(archive)
    assert_regions_equal(method.predict(archive), first)
    after = method.state_dict()
    assert before.keys() == after.keys()
    assert all(torch.equal(value, after[name]) for name, value in before.items())
    other = AgACI((-10, 20), gammas=[0, 0.05, 0.6], alpha=0.2, chunk_size=100)
    assert_regions_equal(other.calibrate(archive).predict(archive), first)


def test_rolling_source_is_supported():
    archive = synthetic_archive(horizon=1, num_origins=120)
    check_method(
        lambda: AgACI((-100, 100), gammas=[0, 0.05], source=ExpandingSource(window_size=20)),
        archives=[archive],
    )


@pytest.mark.parametrize("bounds", [[], [0], [0, 1, 2], [1, 0], [0, 0], [-torch.inf, 1]])
def test_invalid_endpoint_replacements(bounds):
    with pytest.raises(ValueError, match="endpoint_bounds"):
        AgACI(bounds)


@pytest.mark.parametrize("gammas", [[], [-0.1], [torch.nan], [torch.inf]])
def test_invalid_expert_rates(gammas):
    with pytest.raises(ValueError, match="gammas"):
        AgACI((-1, 1), gammas=gammas)


@pytest.mark.parametrize("horizon,num_series", [(2, 1), (1, 2)])
def test_unsupported_shapes_are_rejected(horizon, num_series):
    archive = synthetic_archive(horizon=horizon, num_series=num_series)
    with pytest.raises(ValueError, match="one-step, univariate"):
        AgACI((-10, 10)).calibrate(archive)


def test_single_delayed_step_is_rejected():
    archive = ResidualArchive.from_forecasts(
        torch.zeros(20),
        torch.zeros(20),
        steps=torch.tensor([2]),
        sizes={"calibration": 8, "test": None},
    )
    with pytest.raises(ValueError, match="one-step, univariate"):
        AgACI((-10, 10)).calibrate(archive)


def test_forecasts_are_required_and_test_forecasts_must_be_finite():
    archive = synthetic_archive(horizon=1)
    with pytest.raises(ValueError, match="needs forecasts"):
        AgACI((-10, 10)).calibrate(replace(archive, forecasts=None))
    forecasts = archive.forecasts.clone()
    forecasts[archive.splits["test"][0]] = torch.nan
    bad = replace(archive, forecasts=forecasts)
    with pytest.raises(ValueError, match="finite forecasts"):
        AgACI((-10, 10)).calibrate(bad).predict(bad)


def test_arbitrary_scores_are_rejected():
    from tscp.scores import SignedScore

    with pytest.raises(TypeError):
        AgACI((-10, 10), score=SignedScore())

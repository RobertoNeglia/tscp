import json
from pathlib import Path

import pytest
import torch

from tscp.online.aci import ACIUpdate
from tscp.online.dtaci import DtACIUpdate
from tscp.quantile import EmpiricalDistribution
from tscp.testing import check_online_update

FIXTURE = Path(__file__).parents[1] / "fixtures" / "online_reference.json"


class ContinuousDistribution:
    def threshold(self, alpha):
        return 1 - alpha


def test_deterministic_algorithm_matches_independent_scalar_reference():
    data = json.loads(FIXTURE.read_text())["dtaci"]
    update = DtACIUpdate(gammas=data["gammas"], eta=data["eta"], sigma=data["sigma"])
    update.reset(torch.tensor(data["alpha"], dtype=torch.float64), torch.device("cpu"))
    for t, beta in enumerate(data["betas"]):
        torch.testing.assert_close(
            update.expert_levels,
            torch.tensor(data["expert_levels"][t], dtype=torch.float64),
            atol=1e-14,
            rtol=1e-13,
        )
        torch.testing.assert_close(
            update.probabilities,
            torch.tensor(data["probabilities"][t], dtype=torch.float64),
            atol=1e-14,
            rtol=1e-13,
        )
        threshold = update.threshold(t, ContinuousDistribution())
        assert float(threshold) == pytest.approx(1 - data["combined_levels"][t], abs=1e-13)
        update.observe(
            t,
            torch.tensor(1 - beta, dtype=torch.float64),
            threshold,
            torch.tensor(True),
            torch.tensor(beta, dtype=torch.float64),
        )


UPSTREAM = json.loads((FIXTURE.parent / "dtaci_reference.json").read_text())


@pytest.mark.parametrize("case", UPSTREAM["cases"], ids=lambda case: case["name"])
def test_deterministic_algorithm_matches_the_authors_r_code(case):
    assert UPSTREAM["reference_revision"] == "09ab1674bc579b46d0e54b3525059c0591be342f"
    update = DtACIUpdate(gammas=case["gammas"], eta=case["eta"], sigma=case["sigma"])
    update.reset(torch.tensor(case["alpha"], dtype=torch.float64), torch.device("cpu"))
    gammas = torch.tensor(case["gammas"], dtype=torch.float64)
    tolerance = UPSTREAM["tolerance"]
    for t, beta in enumerate(case["betas"]):
        mean_gamma = (update.probabilities * gammas).sum()
        assert float(mean_gamma) == pytest.approx(case["mean_gammas"][t], **tolerance_of(tolerance))
        threshold = update.threshold(t, ContinuousDistribution())
        level = 1 - float(threshold)
        assert level == pytest.approx(case["combined_levels"][t], **tolerance_of(tolerance))
        update.observe(
            t,
            torch.tensor(1 - beta, dtype=torch.float64),
            threshold,
            torch.tensor(True),
            torch.tensor(beta, dtype=torch.float64),
        )


def tolerance_of(tolerance):
    return {"abs": tolerance["atol"], "rel": tolerance["rtol"]}


def test_one_expert_is_aci():
    dtaci, aci = DtACIUpdate(gammas=[0.02]), ACIUpdate(gamma=0.02)
    for update in (dtaci, aci):
        update.reset(torch.tensor([0.1, 0.2]), torch.device("cpu"))
    distribution = EmpiricalDistribution(torch.arange(1, 31).double()[:, None].expand(-1, 2))
    for t in range(30):
        score = torch.tensor([35.0 if t % 3 else 0.0, 0.0 if t % 2 else 40.0])
        issued = [update.threshold(t, distribution) for update in (dtaci, aci)]
        assert torch.equal(*issued)
        for update, threshold in zip((dtaci, aci), issued, strict=True):
            update.observe(
                t, score, threshold, torch.ones(2, dtype=torch.bool), distribution.level(score)
            )


def test_expert_misses_use_issued_thresholds_at_empirical_boundary():
    update = DtACIUpdate(gammas=[0.1])
    update.reset(torch.tensor(0.5, dtype=torch.float64), torch.device("cpu"))
    distribution = EmpiricalDistribution(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
    threshold = update.threshold(0, distribution)
    score = torch.tensor(3.0, dtype=torch.float64)
    assert distribution.level(score) == 0.5
    assert score > threshold
    update.observe(0, score, threshold, torch.tensor(True), distribution.level(score))
    assert float(update.expert_levels[0]) == pytest.approx(0.45)


def test_delayed_release_updates_only_released_elements():
    update = DtACIUpdate(gammas=[0.01, 0.1], eta=2)
    update.reset(torch.tensor([0.1, 0.2]), torch.device("cpu"))
    first = update.threshold(0, ContinuousDistribution())
    second = update.threshold(1, ContinuousDistribution())
    assert torch.equal(first, second)
    original_levels = update.expert_levels.clone()
    original_weights = update.probabilities.clone()
    update.observe(0, torch.tensor([1.0, 1.0]), first, torch.tensor([True, False]), torch.zeros(2))
    assert torch.equal(update.expert_levels[:, 1], original_levels[:, 1])
    assert torch.equal(update.probabilities[:, 1], original_weights[:, 1])
    update.observe(1, torch.tensor([1.0, 1.0]), second, torch.tensor([True, False]), torch.zeros(2))
    expected = original_levels[:, 0] - torch.tensor([0.01, 0.1]).double() * 1.8
    torch.testing.assert_close(update.expert_levels[:, 0], expected)


def test_heuristic_is_per_target_level():
    update = DtACIUpdate()
    update.reset(torch.tensor([0.1, 0.05], dtype=torch.float64), torch.device("cpu"))
    assert float(update.learning_rate[0]) == pytest.approx(2.719912421004082)
    assert update.learning_rate[1] > update.learning_rate[0]


def test_completion_discards_experts_for_missing_outcomes():
    update = DtACIUpdate()
    update.reset(torch.tensor([0.1, 0.1]), torch.device("cpu"))
    threshold = update.threshold(0, ContinuousDistribution())
    update.observe(
        0, torch.tensor([1.0, 0.0]), threshold, torch.tensor([True, False]), torch.zeros(2)
    )
    assert 0 in update._issued
    update.finish_query(0)
    assert not update._issued


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gammas": []},
        {"gammas": [0]},
        {"gammas": [float("inf")]},
        {"eta": 0},
        {"sigma": 1},
        {"lookback": 0},
    ],
)
def test_bad_settings_raise(kwargs):
    with pytest.raises(ValueError):
        DtACIUpdate(**kwargs)


def test_dtaci_keeps_the_update_contract():
    check_online_update(lambda: DtACIUpdate(gammas=[0.01, 0.05]))

import pytest
import torch

from tscp.methods import ACI
from tscp.online import ACIUpdate
from tscp.sources import ExpandingSource, HoldoutSource
from tscp.testing import check_online_update


@pytest.mark.parametrize("gamma", [0.005, 0.05])
def test_aci_keeps_the_online_update_contract(gamma):
    check_online_update(lambda: ACIUpdate(gamma))


class Doubling:
    """A stand-in distribution whose threshold at level a is 2a."""

    def threshold(self, alpha):
        return 2 * alpha


def test_level_moves_only_where_an_outcome_is_released():
    update = ACIUpdate(gamma=0.1)
    update.reset(torch.full((3,), 0.1, dtype=torch.float64), torch.device("cpu"))
    update.observe(
        0,
        scores=torch.tensor([1.0, 0.0, 1.0]),
        thresholds=torch.tensor([0.5, 0.5, 0.5]),
        released=torch.tensor([True, True, False]),
    )
    expected = torch.tensor([0.1 + 0.1 * (0.1 - 1), 0.1 + 0.1 * 0.1, 0.1], dtype=torch.float64)
    assert torch.allclose(update.level, expected)
    assert torch.equal(update.threshold(1, Doubling()), 2 * update.level)


def test_each_element_moves_toward_its_own_target():
    update = ACIUpdate(gamma=0.1)
    update.reset(torch.tensor([0.05, 0.2], dtype=torch.float64), torch.device("cpu"))
    covered = torch.tensor([0.0, 0.0])
    update.observe(0, covered, torch.ones(2), torch.ones(2, dtype=torch.bool))
    assert torch.allclose(update.level, torch.tensor([0.055, 0.22], dtype=torch.float64))


def test_gamma_must_be_positive():
    with pytest.raises(ValueError, match="gamma"):
        ACIUpdate(gamma=0.0)


@pytest.mark.parametrize("source", [HoldoutSource, ExpandingSource])
@pytest.mark.parametrize("gamma", [0.01, 0.05])
def test_long_run_miss_rate_stays_within_the_deterministic_bound(make_archive, source, gamma):
    # Adversarial test period: blocks of 100 origins alternate between tiny and huge errors.
    num_calibration, num_test, alpha = 200, 2000, 0.1
    blocks = torch.tensor([0.01, 100.0]).repeat(num_test // 200).repeat_interleave(100)
    residuals = torch.cat([torch.randn(num_calibration), torch.randn(num_test) * blocks])
    archive = make_archive(residuals, {"calibration": num_calibration, "test": None})
    region = ACI(gamma=gamma, source=source(), alpha=alpha).calibrate(archive).predict(archive)
    miss_rate = 1 - region.contains(archive.split("test").residuals).double().mean().item()
    bound = (max(alpha, 1 - alpha) + gamma) / (gamma * num_test)
    assert abs(miss_rate - alpha) <= bound

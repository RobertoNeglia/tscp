import pytest
import torch

from tscp.methods import ACI
from tscp.online import ACIUpdate
from tscp.sources import ExpandingSource, HoldoutSource


def test_level_moves_only_where_an_outcome_is_released():
    update = ACIUpdate(gamma=0.1)
    update.reset(0.1, (3,), torch.device("cpu"))
    update.observe(
        scores=torch.tensor([1.0, 0.0, 1.0]),
        thresholds=torch.tensor([0.5, 0.5, 0.5]),
        released=torch.tensor([True, True, False]),
    )
    expected = torch.tensor([0.1 + 0.1 * (0.1 - 1), 0.1 + 0.1 * 0.1, 0.1], dtype=torch.float64)
    assert torch.allclose(update.level, expected)
    assert torch.equal(update.threshold(lambda level: 2 * level), 2 * update.level)


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

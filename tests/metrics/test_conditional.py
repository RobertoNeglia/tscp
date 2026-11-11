import math

import pytest
import torch

from tscp.metrics import grouped_coverage, rolling_coverage, worst_window_coverage


def test_rolling_and_worst_window():
    covered = torch.tensor([1, 1, 0, 0, 1, 1], dtype=torch.bool)
    rolling = rolling_coverage(covered, window=2)
    assert torch.allclose(rolling, torch.tensor([1.0, 0.5, 0.0, 0.5, 1.0], dtype=torch.float64))
    assert worst_window_coverage(covered, window=2).item() == 0.0


def test_unobserved_targets_are_skipped():
    covered = torch.tensor([1, 0, 0, 1], dtype=torch.bool)
    observed = torch.tensor([True, False, False, True])
    rolling = rolling_coverage(covered, window=2, observed=observed)
    assert rolling[0].item() == 1.0 and math.isnan(rolling[1].item()) and rolling[2].item() == 1.0
    assert worst_window_coverage(covered, window=2, observed=observed).item() == 1.0


def test_grouped_coverage_pools_elements_within_each_group():
    covered = torch.tensor([[1, 1], [0, 1], [1, 1], [0, 0]], dtype=torch.bool)
    labels, values = grouped_coverage(covered, torch.tensor([7, 3, 7, 3]))
    assert labels.tolist() == [3, 7]
    assert values.tolist() == [0.25, 1.0]


@pytest.mark.parametrize("window", [0, 5])
def test_window_must_fit(window):
    with pytest.raises(ValueError, match="window"):
        rolling_coverage(torch.ones(4, dtype=torch.bool), window)

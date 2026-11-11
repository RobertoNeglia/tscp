import math

import pytest
import torch

from tscp.data import ResidualArchive
from tscp.metrics import coverage, coverage_gap, evaluate, log_width, mean_width, winkler_score
from tscp.regions import Interval


def test_coverage_width_and_log_width():
    interval = Interval(torch.tensor([-1.0, -2.0, 0.0]), torch.tensor([1.0, 2.0, 0.5]))
    residuals = torch.tensor([0.5, 3.0, 0.25])
    assert coverage(interval, residuals).item() == pytest.approx(2 / 3)
    assert coverage_gap(interval, residuals, alpha=0.1).item() == pytest.approx(2 / 3 - 0.9)
    assert torch.allclose(log_width(interval), torch.log(torch.tensor([2.0, 4.0, 0.5])))
    assert mean_width(interval).item() == pytest.approx(6.5 / 3)
    observed = torch.tensor([True, False, True])
    assert coverage(interval, residuals, observed).item() == 1.0


def test_winkler_adds_the_scaled_miss_to_the_width():
    interval = Interval(torch.full((3,), -1.0), torch.full((3,), 1.0))
    scores = winkler_score(interval, torch.tensor([0.0, 2.0, -3.0]), alpha=0.1)
    assert torch.allclose(scores, torch.tensor([2.0, 2.0 + 20 * 1.0, 2.0 + 20 * 2.0]))


def test_unbounded_and_empty_regions_are_counted_not_averaged():
    lower = torch.tensor([-1.0, -torch.inf, torch.inf, -2.0]).reshape(4, 1, 1)
    upper = torch.tensor([1.0, torch.inf, -torch.inf, 2.0]).reshape(4, 1, 1)
    residuals = torch.tensor([0.0, 5.0, 0.0, 3.0]).reshape(4, 1, 1)
    archive = ResidualArchive(residuals=residuals, origins=torch.arange(4), steps=torch.tensor([1]))
    report = evaluate(Interval(lower, upper), archive, alpha=0.1)
    assert report["num_targets"] == 4
    assert report["coverage"] == 0.5  # the unbounded one covers, the empty one misses
    assert report["infinite_fraction"] == 0.25
    assert report["empty_fraction"] == 0.25
    assert report["log_volume_per_dim"] == pytest.approx((math.log(2) + math.log(4)) / 2)
    assert report["winkler"] == pytest.approx((2 + (4 + 20 * 1)) / 2)


def test_evaluate_reports_conditional_coverage_on_request():
    residuals = torch.tensor([0.0, 2.0, 0.0, 0.0]).reshape(4, 1, 1)
    archive = ResidualArchive(residuals=residuals, origins=torch.arange(4), steps=torch.tensor([1]))
    interval = Interval(torch.full((4, 1, 1), -1.0), torch.full((4, 1, 1), 1.0))
    report = evaluate(interval, archive, 0.1, window=2, groups=torch.tensor([0, 1, 0, 1]))
    assert report["worst_window_coverage"] == 0.5
    assert report["grouped_coverage"] == {0: 1.0, 1: 0.5}


def test_evaluate_checks_shapes():
    archive = ResidualArchive(residuals=torch.zeros(4, 1, 1), origins=torch.arange(4), steps=[1])
    with pytest.raises(ValueError, match="one element per residual"):
        evaluate(Interval(torch.zeros(3, 1, 1), torch.ones(3, 1, 1)), archive, 0.1)

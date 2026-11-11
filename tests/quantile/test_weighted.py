import math

import pytest
import torch

import tscp.quantile.weighted as weighted_module
from tscp.quantile import WeightedQuantile, uniform_quantile, weighted_quantile


def brute_force(scores, weights, alpha, test_weight=1.0):
    """The definition, literally: the smallest score whose weighted mass reaches 1 - alpha."""
    if 1 - alpha <= 0:
        return -math.inf
    total = weights.sum().item() + test_weight
    for candidate in sorted(set(scores.tolist())):
        if weights[scores <= candidate].sum().item() / total >= 1 - alpha - 1e-12:
            return candidate
    return math.inf


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.3, 0.5])
def test_weighted_matches_the_definition(alpha):
    scores = torch.rand(40, dtype=torch.float64)
    weights = torch.rand(6, 40, dtype=torch.float64)
    weights[weights < 0.2] = 0.0
    thresholds = weighted_quantile(scores, weights, alpha)
    for query in range(6):
        assert thresholds[query].item() == brute_force(scores, weights[query], alpha)


@pytest.mark.parametrize(
    ("num_scores", "alpha", "rank"), [(99, 0.1, 90), (19, 0.1, 18), (9, 0.1, 9)]
)
def test_uniform_threshold_is_the_conformal_order_statistic(num_scores, alpha, rank):
    scores = torch.randperm(num_scores).to(torch.float64) + 1  # the k-th smallest is k
    assert uniform_quantile(scores, alpha).item() == rank


def test_uniform_threshold_is_infinite_when_the_rank_exceeds_the_pool():
    # ceil(0.99 * 51) = 51 > 50 scores.
    assert uniform_quantile(torch.rand(50), 0.01).item() == math.inf


@pytest.mark.parametrize("num_scores", [1, 9, 19, 99, 100])
@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1, 0.2, 0.5, 0.9])
def test_uniform_and_weighted_paths_agree(num_scores, alpha):
    scores = torch.randn(num_scores)
    ones = torch.ones(1, num_scores)
    assert torch.equal(weighted_quantile(scores, ones, alpha)[0], uniform_quantile(scores, alpha))


@pytest.mark.parametrize("alpha", [1.0, 1.5])
def test_levels_at_or_above_one_give_empty_regions(alpha):
    scores = torch.rand(10)
    assert uniform_quantile(scores, alpha).item() == -math.inf
    assert weighted_quantile(scores, torch.ones(1, 10), alpha).item() == -math.inf


@pytest.mark.parametrize("alpha", [0.0, -0.2])
def test_levels_at_or_below_zero_give_unbounded_regions(alpha):
    scores = torch.rand(10)
    assert uniform_quantile(scores, alpha).item() == math.inf
    assert weighted_quantile(scores, torch.ones(1, 10), alpha).item() == math.inf


def test_too_little_pool_mass_gives_an_unbounded_region():
    scores = torch.rand(10)
    # Pool mass 10 * 0.01 = 0.1 against the test weight 1: never reaches 0.9.
    assert weighted_quantile(scores, torch.full((1, 10), 0.01), 0.1).item() == math.inf


def test_each_kept_element_is_its_own_problem():
    scores = torch.rand(30, 2, 3)
    alpha = torch.tensor([[0.1, 0.2, 0.3], [0.05, 0.5, 0.15]])
    weights = torch.rand(4, 30)
    together = weighted_quantile(scores, weights, alpha)
    uniform = uniform_quantile(scores, alpha)
    assert together.shape == (4, 2, 3) and uniform.shape == (2, 3)
    for step in range(2):
        for series in range(3):
            level = alpha[step, series].item()
            alone = weighted_quantile(scores[:, step, series], weights, level)
            assert torch.equal(together[:, step, series], alone)
            assert uniform[step, series] == uniform_quantile(scores[:, step, series], level)


def test_query_chunks_do_not_change_the_result(monkeypatch):
    scores, weights = torch.rand(50, 2), torch.rand(30, 50, 2)
    expected = weighted_quantile(scores, weights, 0.1)
    monkeypatch.setattr(weighted_module, "_MAX_ELEMENTS", 200)  # chunks of 2 queries
    assert torch.equal(weighted_quantile(scores, weights, 0.1), expected)


def test_component_dispatch_and_checks():
    model = WeightedQuantile()
    scores = torch.rand(20)
    assert torch.equal(model(scores, None, 0.1), uniform_quantile(scores, 0.1))
    with pytest.raises(ValueError, match="nonnegative"):
        model(scores, -torch.ones(1, 20), 0.1)
    with pytest.raises(ValueError, match="nonnegative"):
        WeightedQuantile(test_weight=-1.0)

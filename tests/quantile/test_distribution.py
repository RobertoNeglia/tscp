import pytest
import torch

from tscp.quantile import EmpiricalDistribution


def uniform(scores):
    return EmpiricalDistribution(torch.sort(scores, dim=0).values)


def weighted(scores, weights):
    ordered, order = torch.sort(scores, dim=0)
    return EmpiricalDistribution.weighted(ordered, order, weights)


def test_many_levels_at_once_equal_one_level_at_a_time():
    scores, weights = torch.randn(40, 3), torch.rand(5, 40)
    levels = torch.tensor([0.02, 0.1, 0.25, 0.6], dtype=torch.float64)
    for distribution in (uniform(scores), weighted(scores, weights)):
        batch = distribution.batch_shape
        together = distribution.threshold(levels.reshape(-1, *([1] * len(batch))))
        assert together.shape == (len(levels), *batch)
        for index, level in enumerate(levels.tolist()):
            assert torch.equal(together[index], distribution.threshold(level))


def test_uniform_level_is_the_conformal_p_value():
    scores = torch.randperm(19).double() + 1  # 1 .. 19
    distribution = uniform(scores)
    queries = torch.tensor([0.5, 1.0, 7.0, 7.5, 19.0, 30.0], dtype=torch.float64)
    at_least = (scores[None, :] >= queries[:, None]).sum(dim=1)
    expected = (at_least + 1) / (len(scores) + 1)
    assert torch.allclose(distribution.level(queries), expected.double())


@pytest.mark.parametrize("weights", [None, "random"])
def test_level_is_where_the_threshold_stops_covering(weights):
    scores = torch.randn(60, dtype=torch.float64)
    if weights is None:
        distribution = uniform(scores)
    else:
        distribution = weighted(scores, 0.2 + torch.rand(1, 60, dtype=torch.float64))
        distribution = distribution.select(0)
    for score in scores[torch.randperm(60)[:20]].tolist() + [-10.0, 10.0]:
        level = distribution.level(torch.tensor(score, dtype=torch.float64)).item()
        assert 0 < level <= 1
        if level - 1e-6 > 0:  # just below the level the score is still covered
            assert distribution.threshold(level - 1e-6).item() >= score
        if level + 1e-6 < 1:  # just above it, it is missed
            assert distribution.threshold(level + 1e-6).item() < score


def test_selecting_a_query_keeps_its_row():
    scores, weights = torch.randn(30, 2), torch.rand(4, 30, 2)
    distribution = weighted(scores, weights)
    assert distribution.batch_shape == (4, 2)
    for query in range(4):
        one = distribution.select(query)
        assert one.batch_shape == (2,)
        assert torch.equal(one.threshold(0.1), distribution.threshold(0.1)[query])
        realized = torch.randn(2)
        assert torch.equal(one.level(realized), distribution.level(realized.expand(4, 2))[query])


def test_weights_must_match_the_pool():
    scores = torch.randn(30)
    with pytest.raises(ValueError, match="P pool scores"):
        weighted(scores, torch.rand(2, 29))

import pytest
import torch

from tscp.data import ResidualArchive
from tscp.weights import ExponentialWeights, UniformWeights


def rows(origins) -> ResidualArchive:
    origins = torch.as_tensor(origins)
    return ResidualArchive(
        residuals=torch.zeros(len(origins), 1, 1), origins=origins, steps=torch.tensor([1])
    )


def test_weights_decay_with_age():
    weights = ExponentialWeights(0.9)(rows([0, 1, 2]), rows([3, 5]))
    expected = torch.tensor([[0.9**3, 0.9**2, 0.9], [0.9**5, 0.9**4, 0.9**3]], dtype=torch.float64)
    assert torch.allclose(weights, expected)


def test_rows_after_the_query_get_a_finite_weight():
    # The availability mask removes them; a weight above 1 could overflow before that.
    weights = ExponentialWeights(0.5)(rows([0, 10_000]), rows([1]))
    assert torch.isfinite(weights).all()


def test_decay_one_is_uniform():
    pool, query = rows([0, 1, 2, 3]), rows([4, 6])
    assert torch.equal(ExponentialWeights(1.0)(pool, query), UniformWeights()(pool, query))


@pytest.mark.parametrize("decay", [0.0, -0.5, 1.5])
def test_decay_must_lie_in_the_unit_interval(decay):
    with pytest.raises(ValueError, match="decay"):
        ExponentialWeights(decay)

import pytest
import torch

from tscp.imports import has_module
from tscp.quantile.forest import QuantileForest
from tscp.utils import from_config


def test_forest_config_records_effective_settings():
    forest = QuantileForest(n_estimators=3, max_depth=4, seed=9)
    rebuilt = from_config(forest.to_config())
    assert rebuilt.to_config() == forest.to_config()
    assert forest.get_config() == {
        "n_estimators": 3,
        "max_depth": 4,
        "min_samples_leaf": 1,
        "max_features": 1.0,
        "seed": 9,
        "n_jobs": 1,
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_estimators": 0},
        {"max_depth": 0},
        {"min_samples_leaf": 0},
        {"max_features": 0},
        {"seed": -1},
        {"n_jobs": 0},
    ],
)
def test_forest_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        QuantileForest(**kwargs)


def test_forest_rejects_integer_targets_and_empty_probability_vectors():
    features = torch.arange(8, dtype=torch.float64).reshape(4, 2)
    targets = torch.arange(4)
    forest = QuantileForest()
    with pytest.raises(ValueError, match="floating residuals"):
        forest(features, targets, features[-1], torch.tensor([0.5]))
    with pytest.raises(ValueError, match="nonempty"):
        forest(features, targets.double(), features[-1], torch.empty(0))


@pytest.mark.parametrize("name", ["features", "targets", "query", "probabilities"])
def test_forest_rejects_nonfloating_tensors_without_loading_the_backend(name):
    values = {
        "features": torch.arange(8, dtype=torch.float64).reshape(4, 2),
        "targets": torch.arange(4, dtype=torch.float64),
        "query": torch.tensor([1.0, 2.0]),
        "probabilities": torch.tensor([0.0, 1.0]),
    }
    values[name] = values[name].long()
    with pytest.raises(ValueError, match=f"{name} must hold floating"):
        QuantileForest()(**values)


@pytest.mark.skipif(not has_module("sklearn_quantile"), reason="needs the forest extra")
def test_forest_determinism_shape_and_runtime_state():
    features = torch.arange(20, dtype=torch.float64).reshape(10, 2)
    targets = torch.tensor([0.0, 1.0, 4.0, 1.0, 0.0, 2.0, 5.0, 3.0, 1.0, 4.0], dtype=torch.float64)
    query, probabilities = features[-1], torch.tensor([0.0, 0.2, 0.8, 1.0], dtype=torch.float64)
    forest = QuantileForest(n_estimators=3, seed=8)
    predicted = forest(features, targets, query, probabilities)
    assert predicted.shape == (4,) and predicted.dtype == targets.dtype
    assert (predicted.diff() >= 0).all()
    assert torch.equal(predicted, forest(features, targets, query, probabilities))
    assert not forest.state_dict()
    if torch.cuda.is_available():
        actual = forest(features.cuda(), targets.cuda(), query.cuda(), probabilities.cuda())
        assert actual.is_cuda
        torch.testing.assert_close(actual.cpu(), predicted)


@pytest.mark.skipif(not has_module("sklearn_quantile"), reason="needs the forest extra")
def test_forest_preserves_requested_quantile_order_and_duplicates():
    features = torch.arange(20, dtype=torch.float64).reshape(10, 2)
    targets = torch.tensor([0.0, 1.0, 4.0, 1.0, 0.0, 2.0, 5.0, 3.0, 1.0, 4.0], dtype=torch.float64)
    probabilities = torch.tensor([0.8, 0.0, 0.2, 0.8, 1.0, 0.2], dtype=torch.float64)
    forest = QuantileForest(n_estimators=3, seed=8)
    requested = forest(features, targets, features[-1], probabilities)
    ordered, order = probabilities.sort(stable=True)
    expected = forest(features, targets, features[-1], ordered)
    assert torch.equal(requested[order], expected)
    assert requested[0] == requested[3] and requested[2] == requested[5]

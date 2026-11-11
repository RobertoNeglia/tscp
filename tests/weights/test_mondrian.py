from dataclasses import replace

import pytest
import torch

from tscp.features import Calendar, ContextColumns, Phase
from tscp.methods import MondrianCP
from tscp.testing import check_method, check_weights, synthetic_archive
from tscp.weights import MondrianWeights


def test_mondrian_exact_tuple_match():
    archive = synthetic_archive(horizon=2, num_origins=100)
    archive = replace(archive, context=torch.tensor([[i % 3, i % 2] for i in range(100)]))
    weights = MondrianWeights(ContextColumns())
    pool, query = archive.head(20), archive.select(torch.arange(30, 40))
    expected = (query.context[:, None] == pool.context[None]).all(dim=-1)
    assert torch.equal(weights(pool, query), expected)


@pytest.mark.parametrize("groups", [lambda: Phase(3), lambda: Calendar(["hour"])])
def test_mondrian_contracts(groups):
    check_weights(lambda: MondrianWeights(groups()))
    check_method(lambda: MondrianCP(groups()))


def test_continuous_or_series_groups_are_rejected():
    archive = synthetic_archive(horizon=1, num_origins=100)
    weights = MondrianWeights(ContextColumns())
    with pytest.raises(ValueError, match="discrete"):
        weights(archive.head(20), archive.head(10))
    archive = replace(archive, context=torch.ones(100, 1, 2))
    with pytest.raises(ValueError, match="shape"):
        weights(archive.head(20), archive.head(10))


def test_unseen_group_is_unbounded():
    archive = synthetic_archive(horizon=1, num_origins=100)
    groups = torch.zeros(100, 1)
    groups[archive.splits["test"]] = 1
    archive = replace(archive, context=groups, context_names=("category",))
    region = MondrianCP(ContextColumns()).calibrate(archive).predict(archive)
    assert torch.isneginf(region.lower).all()
    assert torch.isposinf(region.upper).all()

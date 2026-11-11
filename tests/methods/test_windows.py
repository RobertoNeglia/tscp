import pytest
import torch

from tscp.data import ResidualArchive, SplitPlan
from tscp.methods import ConformalPredictor, WindowedCP
from tscp.quantile import Pool
from tscp.scores import AbsoluteScore, SignedScore
from tscp.sources import ExpandingSource
from tscp.testing import MaxAbsScore, assert_regions_equal, check_method, synthetic_archive


@pytest.mark.parametrize("score", [AbsoluteScore, SignedScore, MaxAbsScore])
def test_windowed_method_contracts(score):
    check_method(lambda: WindowedCP(40, score=score()))


def test_large_window_equals_expanding():
    archive = synthetic_archive(horizon=3, num_origins=200, missing=0.1)
    large = WindowedCP(1000).calibrate(archive).predict(archive)
    full = ConformalPredictor(AbsoluteScore(), source=ExpandingSource())
    assert_regions_equal(large, full.calibrate(archive).predict(archive))


def test_irregular_origins_window_counts_available_rows_before_missing_mask():
    origins = torch.tensor([0, 2, 5, 8, 12, 13, 18])
    steps = torch.tensor([1, 3])
    observed = torch.ones(7, 2, 1, dtype=torch.bool)
    observed[2, 0] = False
    archive = ResidualArchive(
        residuals=torch.arange(14).reshape(7, 2, 1).float() * observed,
        origins=origins,
        steps=steps,
        observed=observed,
        splits=SplitPlan({"calibration": torch.arange(3), "test": torch.arange(3, 7)}),
    )
    method = WindowedCP(2)
    rows = archive.select(torch.arange(7))
    pool = Pool(method.score(rows), rows, method.score.valid(rows))
    weights = method._weights(pool, archive.select(torch.tensor([3, 5])), steps, True)
    expected = torch.zeros(2, 7, 2, 1)
    expected[0, 1, 0] = 1  # row 2 is missing but consumes the second slot
    expected[0, 1:3, 1] = 1
    expected[1, 3:5, 0] = 1
    expected[1, 2:4, 1] = 1
    assert torch.equal(weights, expected)


def test_window_size_validation():
    for value in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="window_size"):
            ExpandingSource(value)


def test_chunk_size_does_not_change_window():
    archive = synthetic_archive(horizon=3, num_origins=200)
    method = WindowedCP(25, SignedScore()).calibrate(archive)
    first = method.predict(archive)
    method.chunk_size = 1
    assert_regions_equal(first, method.predict(archive))

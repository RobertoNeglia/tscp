import numpy as np
import pytest
import torch

from tscp.data import ResidualArchive
from tscp.testing import synthetic_archive


def test_an_archive_rebuilt_from_its_targets_and_forecasts_has_the_same_data():
    archive = synthetic_archive(horizon=3, num_series=2, num_origins=200, missing=0.1)
    rebuilt = ResidualArchive.from_forecasts(
        archive.targets(),
        archive.forecasts,
        origins=archive.origins,
        steps=archive.steps,
        splits=archive.splits,
        timestamps=archive.timestamps,
    )
    # truth - forecast after forecast + residual: equal up to float rounding.
    assert torch.allclose(rebuilt.residuals, archive.residuals, atol=1e-4)
    assert torch.equal(rebuilt.observed, archive.observed)
    assert torch.equal(rebuilt.timestamps, archive.timestamps)
    assert rebuilt.splits.sizes() == archive.splits.sizes()


@pytest.mark.parametrize(("shape", "cube"), [((40,), (40, 1, 1)), ((40, 3), (40, 3, 1))])
def test_one_series_and_one_step_shapes_are_accepted(shape, cube):
    truth = torch.randn(shape)
    archive = ResidualArchive.from_forecasts(truth, torch.zeros(shape))
    assert archive.residuals.shape == cube
    assert torch.equal(archive.residuals.reshape(shape), truth)
    assert archive.observed is None and archive.origins.tolist() == list(range(40))


def test_missing_truth_is_unobserved_and_missing_forecasts_must_match_it():
    truth, forecasts = torch.ones(10), torch.zeros(10)
    truth[2], forecasts[2] = torch.nan, torch.nan  # no target, no forecast: fine
    archive = ResidualArchive.from_forecasts(truth, forecasts)
    assert not archive.observed[2, 0, 0] and archive.residuals[2, 0, 0] == 0
    assert torch.isnan(archive.targets()[2, 0, 0])
    forecasts[4] = torch.nan  # a target without a forecast
    with pytest.raises(ValueError, match="finite wherever the truth is observed"):
        ResidualArchive.from_forecasts(truth, forecasts)


def test_splits_timestamps_and_their_checks():
    truth = torch.randn(30)
    times = np.arange("2026-01-05", "2026-02-04", dtype="datetime64[D]")
    archive = ResidualArchive.from_forecasts(
        truth, torch.zeros(30), sizes={"calibration": 10, "test": None}, timestamps=times
    )
    assert archive.splits.sizes() == {"calibration": 10, "test": 20}
    assert int(archive.timestamps[1] - archive.timestamps[0]) == 86_400 * 1_000_000_000
    with pytest.raises(ValueError, match="sizes or splits"):
        ResidualArchive.from_forecasts(
            truth, torch.zeros(30), sizes={"test": None}, splits=archive.splits
        )
    with pytest.raises(ValueError, match="same shape"):
        ResidualArchive.from_forecasts(truth, torch.zeros(29))
    with pytest.raises(ValueError, match="forecasts"):
        ResidualArchive(residuals=torch.zeros(3, 1, 1), origins=[0, 1, 2], steps=[1]).targets()

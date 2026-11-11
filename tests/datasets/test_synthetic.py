import pytest
import torch

from tscp.datasets import gaussian_control, variance_shift


def test_shapes_and_default_splits():
    archive = gaussian_control(horizon=3, num_origins=500, num_series=2)
    assert archive.residuals.shape == (500, 3, 2)
    assert archive.splits.names == ("fit", "calibration", "test")
    assert archive.splits.sizes()["fit"] == 100 and archive.splits.sizes()["calibration"] == 150
    assert archive.provenance["role"] == "control" and archive.provenance["horizon"] == 3


def test_multi_step_residuals_sum_the_next_innovations():
    archive = gaussian_control(horizon=3, num_origins=50)
    residuals = archive.residuals[..., 0]
    # r(s, 2) - r(s, 1) is the innovation at s + 2, which is also r(s + 1, 1).
    assert torch.allclose(residuals[:-1, 1] - residuals[:-1, 0], residuals[1:, 0], atol=1e-5)


def test_forecasts_plus_residuals_are_the_targets():
    archive = gaussian_control(horizon=2, num_origins=50)
    targets = archive.forecasts + archive.residuals
    # The last-value forecast at origin s + h is the target of (s, h).
    assert torch.allclose(targets[:-2, 1, 0], archive.forecasts[2:, 0, 0], atol=1e-5)


def test_seeds_reproduce_archives():
    first, second = gaussian_control(horizon=1, seed=4), gaussian_control(horizon=1, seed=4)
    assert first.fingerprint() == second.fingerprint()
    assert first.fingerprint() != gaussian_control(horizon=1, seed=5).fingerprint()


def test_variance_shift_changes_the_scale():
    archive = variance_shift(horizon=1, num_origins=4000, scale_after=3.0)
    residuals = archive.residuals[:, 0, 0]
    ratio = residuals[3100:].std() / residuals[:2900].std()
    assert 2.7 < ratio < 3.3


def test_horizon_must_be_given():
    with pytest.raises(TypeError):
        gaussian_control()
    with pytest.raises(ValueError, match="horizon"):
        gaussian_control(horizon=0)

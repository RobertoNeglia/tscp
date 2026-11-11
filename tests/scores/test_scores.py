from dataclasses import replace

import pytest
import torch

from tscp.datasets import gaussian_control, variance_shift
from tscp.methods import ACI, SplitCP
from tscp.quantile import uniform_quantile
from tscp.regions import Box
from tscp.scores import AbsoluteScore, CQRScore, NormalizedScore, Score, SignedScore


def split_cp(score, archive, alpha=0.1):
    return SplitCP(score=score, alpha=alpha).fit(archive).calibrate(archive).predict(archive)


def with_missing_targets(archive, fraction: float, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    observed = torch.rand(archive.residuals.shape, generator=generator) > fraction
    return replace(archive, residuals=archive.residuals * observed, observed=observed)


class MaxAbsScore(Score):
    """The largest absolute residual over the horizon: one box per series."""

    reduce_dims = ("step",)

    def forward(self, rows):
        return rows.residuals.abs().amax(dim=1)

    def region(self, threshold, rows):
        bound = threshold.unsqueeze(-1).expand(*threshold.shape, rows.horizon)
        return Box(-bound, bound, event_dims=("step",))


@pytest.mark.parametrize("share", [0.5, 0.2])
def test_signed_score_takes_each_side_at_its_share_of_alpha(share):
    archive = variance_shift(horizon=2, num_origins=400, num_series=2)
    calibration = archive.split("calibration").residuals
    region = split_cp(SignedScore(lower_share=share), archive, alpha=0.1)
    lower = uniform_quantile(-calibration, 0.1 * share)
    upper = uniform_quantile(calibration, 0.1 * (1 - share))
    assert torch.equal(region.lower, (-lower).expand_as(region.lower))
    assert torch.equal(region.upper, upper.expand_as(region.upper))


def test_signed_score_checks_its_share():
    with pytest.raises(ValueError, match="lower_share"):
        SignedScore(lower_share=1.0)
    assert SignedScore().kept_shape(3, 2) == (3, 2, 2)


def test_normalized_score_with_unit_scale_is_the_absolute_score():
    archive = variance_shift(horizon=2, num_origins=300)
    ones = replace(archive, outputs={"scale": torch.ones_like(archive.residuals)})
    normalized = split_cp(NormalizedScore("scale"), ones)
    assert torch.equal(normalized.upper, split_cp(AbsoluteScore(), archive).upper)


def test_normalized_score_scales_regions_by_the_output():
    archive = variance_shift(horizon=1, num_origins=300)
    scaled = replace(archive, outputs={"scale": 1 + torch.rand_like(archive.residuals)})
    region = split_cp(NormalizedScore("scale"), scaled)
    calibration, test = scaled.split("calibration"), scaled.split("test")
    threshold = uniform_quantile(calibration.residuals.abs() / calibration.outputs["scale"], 0.1)
    assert torch.equal(region.upper, threshold * test.outputs["scale"])
    negative = replace(archive, outputs={"scale": -scaled.outputs["scale"]})
    with pytest.raises(ValueError, match="positive"):
        split_cp(NormalizedScore("scale"), negative)
    with pytest.raises(KeyError, match="no output"):
        split_cp(NormalizedScore("spread"), scaled)


def test_fitted_scale_is_the_root_mean_square_of_the_fit_split():
    archive = variance_shift(horizon=2, num_origins=300, num_series=2)
    score = NormalizedScore()
    assert score.is_trainable and not NormalizedScore("scale").is_trainable
    with pytest.raises(RuntimeError, match="fit"):
        score(archive.split("calibration"))
    region = split_cp(score, archive)
    fit = archive.split("fit").residuals.double()
    assert torch.allclose(score.fitted_scale.double(), fit.square().mean(dim=0).sqrt())
    assert "fitted_scale" in score.state_dict()
    assert torch.isfinite(region.upper).all()


def test_fitted_scale_skips_unobserved_targets():
    archive = with_missing_targets(variance_shift(horizon=2, num_origins=300), fraction=0.3)
    score = NormalizedScore()
    SplitCP(score=score).fit(archive)
    fit = archive.split("fit")
    squares = (fit.residuals.double().square() * fit.observed).sum(dim=0)
    assert torch.allclose(score.fitted_scale.double(), (squares / fit.observed.sum(dim=0)).sqrt())


def test_cqr_with_a_zero_width_band_is_the_absolute_score():
    archive = variance_shift(horizon=2, num_origins=300)
    band = replace(archive, outputs={"lower": archive.forecasts, "upper": archive.forecasts})
    assert torch.equal(split_cp(CQRScore(), band).upper, split_cp(AbsoluteScore(), archive).upper)
    asymmetric, signed = split_cp(CQRScore(asymmetric=True), band), split_cp(SignedScore(), archive)
    assert torch.equal(asymmetric.lower, signed.lower)
    assert torch.equal(asymmetric.upper, signed.upper)


def test_cqr_widens_the_forecaster_band_by_the_conformal_correction():
    archive = variance_shift(horizon=1, num_origins=300)
    half = 0.5 + torch.rand_like(archive.residuals)
    band = replace(
        archive, outputs={"lower": archive.forecasts - half, "upper": archive.forecasts + half}
    )
    region = split_cp(CQRScore(), band)
    calibration, test = band.split("calibration"), band.split("test")
    low = calibration.outputs["lower"] - calibration.forecasts
    high = calibration.outputs["upper"] - calibration.forecasts
    residuals = calibration.residuals
    threshold = uniform_quantile(torch.maximum(low - residuals, residuals - high), 0.1)
    assert torch.allclose(region.lower, test.outputs["lower"] - test.forecasts - threshold)
    assert torch.allclose(region.upper, test.outputs["upper"] - test.forecasts + threshold)


def test_cqr_needs_forecasts_and_quantile_outputs():
    archive = variance_shift(horizon=1, num_origins=300)
    with pytest.raises(KeyError, match="no outputs"):
        split_cp(CQRScore(), archive)
    band = {"lower": archive.forecasts - 1, "upper": archive.forecasts + 1}
    with pytest.raises(ValueError, match="forecasts"):
        split_cp(CQRScore(), replace(archive, forecasts=None, outputs=band))


def test_joint_scores_calibrate_on_fully_observed_rows_only():
    archive = gaussian_control(horizon=3, num_origins=400, num_series=2)
    archive = with_missing_targets(archive, fraction=0.1)
    region = split_cp(MaxAbsScore(), archive)
    test = archive.split("test")
    assert isinstance(region, Box) and region.batch_shape == (len(test), 2)
    calibration = archive.split("calibration")
    complete = calibration.observed.all(dim=1)
    scores = calibration.residuals.abs().amax(dim=1)
    for series in range(2):
        expected = uniform_quantile(scores[complete[:, series], series], 0.1)
        assert torch.equal(region.upper[:, series, 0], expected.expand(len(test)))


def test_joint_scores_run_online_with_missing_targets():
    archive = with_missing_targets(gaussian_control(horizon=3, num_origins=400), fraction=0.1)
    region = ACI(gamma=0.01, score=MaxAbsScore()).calibrate(archive).predict(archive)
    assert isinstance(region, Box) and torch.isfinite(region.log_volume()).any()


def test_validity_reduces_over_joint_dimensions_and_spreads_over_channels():
    archive = with_missing_targets(gaussian_control(horizon=3, num_origins=50, num_series=2), 0.2)
    observed = archive.observed
    assert torch.equal(MaxAbsScore().valid(archive), observed.all(dim=1))
    signed = SignedScore().valid(archive)
    assert signed.shape == (*observed.shape, 2)
    assert torch.equal(signed[..., 0], observed) and torch.equal(signed[..., 1], observed)
    assert AbsoluteScore().valid(gaussian_control(horizon=1, num_origins=50)) is None

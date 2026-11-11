"""The checks catch components that break the contracts they check."""

import pytest
import torch

from tscp.features import FeatureMap
from tscp.methods import ConformalPredictor, SplitCP
from tscp.online import ACIUpdate
from tscp.quantile import EmpiricalDistribution, WeightedQuantile
from tscp.regions import Interval
from tscp.scores import AbsoluteScore, SignedScore
from tscp.sources import HoldoutSource
from tscp.testing import (
    check_feature_map,
    check_method,
    check_online_update,
    check_quantile_model,
    check_score,
    check_weights,
    synthetic_archive,
)
from tscp.weights import Weights

# -- scores ---------------------------------------------------------------------------------


class TooWideScore(AbsoluteScore):
    def region(self, threshold, rows):
        return Interval(-2 * threshold, 2 * threshold)


class PeekingScore(AbsoluteScore):
    """Covers exactly the right residuals, but builds its bounds from them."""

    def region(self, threshold, rows):
        magnitude = rows.residuals.abs()
        upper = torch.where(magnitude <= threshold, threshold, magnitude - 1e-3)
        return Interval(-threshold, upper)


class OverclaimingScore(AbsoluteScore):
    def valid(self, rows):
        return torch.ones_like(rows.residuals, dtype=torch.bool)


class GreedySignedScore(SignedScore):
    def levels(self, alpha, kept):
        return torch.full(kept, float(alpha), dtype=torch.float64)


class FlatScore(AbsoluteScore):
    def forward(self, rows):
        return rows.residuals.abs().mean(dim=2)


@pytest.mark.parametrize(
    ("score", "prop"),
    [
        (TooWideScore, "sublevel set"),
        (PeekingScore, "no residuals"),
        (OverclaimingScore, "validity"),
        (GreedySignedScore, "levels"),
        (FlatScore, "shape"),
    ],
)
def test_check_score_catches_broken_scores(score, prop):
    with pytest.raises(AssertionError, match=prop):
        check_score(score)


# -- weights --------------------------------------------------------------------------------


class FutureWeights(Weights):
    """Weights from the mean of every residual in the timeline, the future included."""

    def forward(self, pool, query):
        level = 1 + query.timeline.residuals.abs().mean()
        return torch.full((len(query), len(pool)), float(level), dtype=torch.float64)


class NegativeWeights(Weights):
    def forward(self, pool, query):
        return -torch.ones(len(query), len(pool), dtype=torch.float64)


@pytest.mark.parametrize(
    ("weights", "prop"), [(FutureWeights, "causality"), (NegativeWeights, "values")]
)
def test_check_weights_catches_broken_weights(weights, prop):
    with pytest.raises(AssertionError, match=prop):
        check_weights(weights)


# -- quantile models ------------------------------------------------------------------------


class ClampedDistribution(EmpiricalDistribution):
    def threshold(self, alpha):
        return super().threshold(alpha).clamp(-1e9, 1e9)


class ClampedQuantile(WeightedQuantile):
    def forward(self, pool, weights, query):
        distribution = super().forward(pool, weights, query)
        return ClampedDistribution(
            distribution.ordered, distribution.cumulative, distribution.test_weight
        )


def test_check_quantile_model_catches_finite_extreme_levels():
    with pytest.raises(AssertionError, match="conventions"):
        check_quantile_model(ClampedQuantile)


# -- online updates -------------------------------------------------------------------------


class StickyUpdate(ACIUpdate):
    """Keeps its level from one pass to the next."""

    def reset(self, alpha, device):
        if self.level is None:
            super().reset(alpha, device)


class DeafUpdate(ACIUpdate):
    def observe(self, t, scores, thresholds, released, levels=None):
        pass


@pytest.mark.parametrize(("update", "prop"), [(StickyUpdate, "reset"), (DeafUpdate, "response")])
def test_check_online_update_catches_broken_updates(update, prop):
    with pytest.raises(AssertionError, match=prop):
        check_online_update(lambda: update(gamma=0.02))


# -- methods --------------------------------------------------------------------------------


class NoisyWeights(Weights):
    def forward(self, pool, query):
        return torch.rand(len(query), len(pool), dtype=torch.float64)


class CenteredScore(AbsoluteScore):
    """``|r - m|``, with the center ``m`` fitted as the mean residual."""

    def __init__(self):
        super().__init__()
        self.register_buffer("center", None)

    def fit(self, train, validation=None):
        self.center = train.residuals.mean(dim=0)
        return self

    def forward(self, rows):
        return (rows.residuals - self.center).abs()

    def region(self, threshold, rows):
        return Interval(self.center - threshold, self.center + threshold)


class CalibrationFitted(ConformalPredictor):
    """Fits its score on the calibration split instead of the fit split."""

    def fit(self, archive):
        self.score.fit(archive.split("calibration"))
        return self


class FutureScore(AbsoluteScore):
    """Widens every region by the mean of the whole timeline."""

    def region(self, threshold, rows):
        extra = rows.timeline.residuals.abs().mean()
        return Interval(-threshold - extra, threshold + extra)


def altered_after_construction():
    method = SplitCP()
    method.alpha = 0.3  # not part of the recorded config
    return method


@pytest.mark.parametrize(
    ("factory", "prop"),
    [
        (lambda: ConformalPredictor(AbsoluteScore(), weights=NoisyWeights()), "determinism"),
        (lambda: SplitCP(score=FutureScore()), "causality"),
        (lambda: CalibrationFitted(CenteredScore(), source=HoldoutSource()), "split discipline"),
        (altered_after_construction, "config"),
    ],
)
def test_check_method_catches_broken_methods(factory, prop):
    with pytest.raises(AssertionError, match=prop):
        check_method(factory)


def test_a_correctly_fitted_center_passes():
    check_method(lambda: SplitCP(score=CenteredScore()))


# -- feature maps ---------------------------------------------------------------------------


class PeekingFeatures(FeatureMap):
    """The rows' own residuals, whose targets come after their origins."""

    def forward(self, rows):
        return rows.residuals.reshape(len(rows), -1)


def test_check_feature_map_catches_a_look_ahead():
    with pytest.raises(AssertionError, match="causality"):
        check_feature_map(PeekingFeatures)


# -- fixtures -------------------------------------------------------------------------------


def test_synthetic_archives_have_every_output_and_the_asked_missing_fraction():
    archive = synthetic_archive(horizon=3, num_series=2, num_origins=200, missing=0.2, seed=5)
    assert (archive.horizon, archive.num_series, archive.num_origins) == (3, 2, 200)
    assert set(archive.outputs) == {"scale", "lower", "upper"}
    assert (archive.outputs["scale"] > 0).all() and archive.forecasts is not None
    assert abs(1 - archive.observed.double().mean().item() - 0.2) < 0.05
    again = synthetic_archive(horizon=3, num_series=2, num_origins=200, missing=0.2, seed=5)
    assert again.fingerprint() == archive.fingerprint()
    assert synthetic_archive(horizon=1).observed is None

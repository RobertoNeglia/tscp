"""Sequential prediction intervals from conditional residual quantiles."""

import torch
from torch import Tensor

from tscp.data import ResidualArchive
from tscp.methods.enbpi import _ResidualIntervalMethod
from tscp.quantile.forest import QuantileForest
from tscp.quantile.width import WidthOptimal
from tscp.regions import Interval
from tscp.registry import register


@register("method", "spci")
class SPCI(_ResidualIntervalMethod):
    """Sequential predictive conformal inference (Xu & Xie, ICML 2023, Algorithm 1).

    Refit a quantile regression forest at each origin using lagged signed residuals
    within the current rolling window. Predict the next residual's conditional quantiles
    from the latest ``lags`` observations, and choose the narrowest interval over five
    tail allocations. Test labels enter training only after their target times pass.

    The point forecaster is supplied through the archive; it is never retrained.
    A fresh forest is fitted on every eligible query, so no external estimator is saved
    or pickled. Recalibration and deterministic replay reconstruct prediction state.

    Deviation:
        Only one-step univariate forecasts are supported. All windows use the exact
        ``sklearn-quantile`` forest, omitting the reference's sampled approximation on
        large data. Each refit uses an explicit seed. Missing labels are not imputed:
        training pairs with missing lags or targets are excluded, and an unavailable
        query lag or fewer than two valid training pairs gives an unbounded interval.
        Lags live on the origin grid, whose spacing is the greatest common divisor of
        the calibration and test origin spacings. An origin absent from both splits,
        such as a row between them, is a missing slot, so a lag vector never joins
        residuals that are not adjacent in time.

    Args:
        lags: Number of consecutive origin-grid slots used as each feature vector.
        window_size: Number of latest origin-grid slots from which to construct both
            training lag vectors and targets. Missing observations and absent origins
            consume slots.
        alpha: Target miscoverage level.
        n_estimators: Number of trees in each exact quantile forest.
        max_depth: Maximum tree depth, or None for unrestricted growth.
        min_samples_leaf: Minimum samples per tree leaf.
        max_features: Fraction of features considered at each tree split.
        seed: Random seed reused for each fresh refit.
        n_jobs: Number of CPU forest workers.
    """

    def __init__(
        self,
        lags: int,
        window_size: int,
        alpha: float = 0.1,
        n_estimators: int = 10,
        max_depth: int | None = 2,
        min_samples_leaf: int = 1,
        max_features: float = 1.0,
        seed: int = 0,
        n_jobs: int = 1,
    ):
        super().__init__(window_size, alpha)
        if isinstance(lags, bool) or not isinstance(lags, int) or lags < 1:
            raise ValueError("lags must be a positive integer")
        if window_size is None or window_size < lags + 2:
            raise ValueError("window_size must allow lags plus at least two training pairs")
        self.lags = lags
        self.forest = QuantileForest(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            seed=seed,
            n_jobs=n_jobs,
        )
        self.interval = WidthOptimal(bins=5)

    def _training_pairs(self, history: Tensor, valid: Tensor) -> tuple[Tensor, Tensor] | None:
        if len(history) < self.lags + 2 or not bool(valid[-self.lags :].all()):
            return None
        windows = history.unfold(0, self.lags + 1, 1)
        eligible = valid.unfold(0, self.lags + 1, 1).all(dim=-1)
        if int(eligible.sum()) < 2:
            return None
        pairs = windows[eligible]
        return pairs[:, :-1], pairs[:, -1]

    def predict(self, archive: ResidualArchive) -> Interval:
        """Fit rolling conditional quantiles and return residual-coordinate intervals."""
        test = self._test_rows(archive)
        self.forest.check_dependency()
        self.reset_online()
        bounds = test.residuals.new_empty((len(test), 2))
        probabilities = self.interval.probabilities(self.alpha, device=test.device).flatten()
        for u, history, valid in self._windows(test, grid=True):
            pairs = self._training_pairs(history, valid)
            if pairs is None:
                bounds[u] = bounds.new_tensor([-torch.inf, torch.inf])
                continue
            features, targets = pairs
            quantiles = self.forest(features, targets, history[-self.lags :], probabilities)
            bounds[u] = self.interval.select(quantiles.reshape(2, self.interval.bins))
        return Interval(bounds[:, 0, None, None], bounds[:, 1, None, None])


__all__ = ["SPCI"]

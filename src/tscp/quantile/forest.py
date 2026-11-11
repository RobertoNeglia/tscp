"""A CPU adapter for the exact sklearn-quantile forest."""

import torch
from torch import Tensor

from tscp.imports import require
from tscp.utils import Component


class QuantileForest(Component):
    """Fit and query the conditional quantile forest used by SPCI.

    This adapter uses ``sklearn_quantile.RandomForestQuantileRegressor`` (Roebroek),
    the backend used by Xu & Xie, ICML 2023. Inputs and outputs are torch tensors;
    conversion to CPU NumPy arrays is confined to this optional backend boundary.
    Each call fits a fresh exact forest and returns quantiles of one feature vector.
    The working estimator is not fitted reusable state and is not saved or pickled.

    Deviation:
        The sampled-forest shortcut used by the author code on large training sets is
        not used. A fixed explicit seed replaces the reference's ambient random state.

    Args:
        n_estimators: Number of trees.
        max_depth: Maximum tree depth, or None for unrestricted growth.
        min_samples_leaf: Minimum samples in a leaf.
        max_features: Fraction of features considered at each split.
        seed: Seed used for each fresh refit.
        n_jobs: Number of CPU workers; 1 makes small rolling fits inexpensive.
    """

    adapts_online = True

    def __init__(
        self,
        n_estimators: int = 10,
        max_depth: int | None = 2,
        min_samples_leaf: int = 1,
        max_features: float = 1.0,
        seed: int = 0,
        n_jobs: int = 1,
    ):
        super().__init__()
        for name, value in (("n_estimators", n_estimators), ("min_samples_leaf", min_samples_leaf)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if max_depth is not None and (
            isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 1
        ):
            raise ValueError("max_depth must be a positive integer or None")
        if not 0 < max_features <= 1:
            raise ValueError("max_features must lie in (0, 1]")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs < 1:
            raise ValueError("n_jobs must be a positive integer")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = float(max_features)
        self.seed = seed
        self.n_jobs = n_jobs

    def check_dependency(self) -> None:
        """Check that the optional backend is installed, with an actionable error."""
        require("sklearn_quantile", "forest")

    def forward(
        self, features: Tensor, targets: Tensor, query: Tensor, probabilities: Tensor
    ) -> Tensor:
        """Return ``[L]`` conditional quantiles from ``[M, F]`` training features.

        Args:
            features: Floating ``[M, F]`` causal lag vectors.
            targets: Floating ``[M]`` observed residuals to predict from the lag vectors.
            query: Floating ``[F]`` causal lag vector of the current forecast.
            probabilities: Floating ``[L]`` quantile probabilities in ``[0, 1]``. Their order
                and duplicates are preserved in the returned predictions.
        """
        if (
            features.ndim != 2
            or features.shape[0] < 1
            or features.shape[1] < 1
            or targets.shape != features.shape[:1]
            or query.shape != features.shape[1:]
        ):
            raise ValueError("features, targets and query must have shapes [M, F], [M] and [F]")
        if not targets.is_floating_point():
            raise ValueError("targets must hold floating residuals")
        for name, values in (
            ("features", features),
            ("query", query),
            ("probabilities", probabilities),
        ):
            if not values.is_floating_point():
                raise ValueError(f"{name} must hold floating values")
        if probabilities.ndim != 1 or not probabilities.numel():
            raise ValueError("probabilities must be a nonempty vector")
        outside = (probabilities < 0) | (probabilities > 1)
        if not torch.isfinite(probabilities).all() or outside.any():
            raise ValueError("probabilities must lie in [0, 1]")
        if not all(torch.isfinite(value).all() for value in (features, targets, query)):
            raise ValueError("forest features and targets must be finite")
        ordered, order = probabilities.sort(stable=True)
        backend = require("sklearn_quantile", "forest")
        estimator = backend.RandomForestQuantileRegressor(
            n_estimators=self.n_estimators,
            q=ordered.detach().cpu().double().numpy(),
            criterion="squared_error",
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=True,
            random_state=self.seed,
            n_jobs=self.n_jobs,
        )
        estimator.fit(features.detach().cpu().numpy(), targets.detach().cpu().numpy())
        values = estimator.predict(query.detach().cpu().numpy().reshape(1, -1))
        values = torch.as_tensor(values, device=targets.device, dtype=targets.dtype).reshape(-1)
        result = torch.empty_like(values)
        result[order.to(targets.device)] = values
        return result


__all__ = ["QuantileForest"]

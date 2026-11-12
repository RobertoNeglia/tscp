"""Ensemble batch prediction intervals from cached out-of-bag forecasts."""

import math
from collections import deque
from collections.abc import Iterator
from typing import Self

import torch
from torch import Tensor

from tscp.data import ResidualArchive
from tscp.data.oob import OOB_CONVENTION
from tscp.methods.base import ConformalPredictor
from tscp.quantile.width import WidthOptimal
from tscp.regions import Interval
from tscp.registry import register
from tscp.scores import SignedScore


class _ResidualIntervalMethod(ConformalPredictor):
    """Shared lifecycle for coupled, one-step signed-residual interval methods."""

    supports_arbitrary_scores = False
    adapts_online = True

    def __init__(self, window_size: int | None, alpha: float):
        super().__init__(score=SignedScore(), alpha=alpha)
        if window_size is not None and (
            isinstance(window_size, bool) or not isinstance(window_size, int) or window_size < 1
        ):
            raise ValueError("window_size must be a positive integer or None")
        self.window_size = window_size
        self._calibration_fingerprint: str | None = None

    def _check_archive(self, archive: ResidualArchive) -> None:
        if archive.horizon != 1 or archive.num_series != 1 or int(archive.steps[0]) != 1:
            raise ValueError(f"{type(self).__name__} supports one step (steps=[1]) and one series")
        if archive.splits is None or "calibration" not in archive.splits:
            raise ValueError("this method needs a calibration split")

    def fit(self, archive: ResidualArchive) -> Self:
        """Validate the archive; the fixed forecaster is supplied by the caller.

        There is no offline fitted state. Online conformal estimation happens only from
        the calibration history and outcomes released during prediction.
        """
        self._check_archive(archive)
        return self

    def calibrate(self, archive: ResidualArchive) -> Self:
        """Remember the initial signed residual history, without fitting a forecaster."""
        self._check_archive(archive)
        calibration = archive.split("calibration")
        self.pool_scores = calibration.residuals.clone()
        self.pool_origins = calibration.origins.clone()
        self.pool_valid = (
            torch.ones_like(calibration.residuals, dtype=torch.bool)
            if calibration.observed is None
            else calibration.observed.clone()
        )
        self._calibration_fingerprint = calibration.fingerprint()
        return self

    def _test_rows(self, archive: ResidualArchive) -> ResidualArchive:
        self._check_archive(archive)
        if self.pool_scores is None:
            raise RuntimeError("calibrate() must run before predict()")
        if archive.splits is None or "test" not in archive.splits:
            raise ValueError("predict() needs an archive with a 'test' split")
        if archive.split("calibration").fingerprint() != self._calibration_fingerprint:
            raise ValueError("predict() must get the calibration archive that calibrate() got")
        if archive.device != self.pool_scores.device:
            raise ValueError("move the calibrated method and archive to the same device")
        return archive.split("test")

    def reset_online(self) -> None:
        """Reset replay state; this method keeps its history local to each prediction."""

    def _windows(
        self, test: ResidualArchive, *, grid: bool = False
    ) -> Iterator[tuple[int, Tensor, Tensor]]:
        """Yield each query with the residual history released before it.

        Without ``grid``, the history holds the latest released rows. With ``grid``, it
        holds the latest slots of the origin grid, whose spacing is the greatest common
        divisor of the calibration and test origin spacings: an origin absent from both
        splits, such as a row between them, becomes a missing slot, so adjacent history
        entries are always adjacent in time. Each query's history then ends at the slot
        just before its origin.
        """
        from tscp.methods.engine import release_schedule

        size = self.window_size if self.window_size is not None else len(self.pool_scores)
        history: deque[Tensor] = deque(maxlen=size)
        observed: deque[Tensor] = deque(maxlen=size)
        empty = test.residuals.new_zeros(())
        absent = torch.zeros((), dtype=torch.bool, device=test.device)
        present = torch.ones((), dtype=torch.bool, device=test.device)
        origins = torch.cat((self.pool_origins, test.origins)).tolist()
        cadence = math.gcd(*(b - a for a, b in zip(origins, origins[1:], strict=False))) or 1
        last: int | None = None

        def pad(through: int) -> None:
            # Absent origins up to and including ``through`` become missing slots.
            nonlocal last
            if grid and last is not None and through > last:
                for _ in range(min((through - last) // cadence, size)):
                    history.append(empty)
                    observed.append(absent)
                last = through

        def append(origin: int, value: Tensor, valid: Tensor) -> None:
            nonlocal last
            pad(origin - cadence)
            history.append(value)
            observed.append(valid)
            last = origin

        for origin, value, valid in zip(
            self.pool_origins.tolist(),
            self.pool_scores[:, 0, 0].unbind(),
            self.pool_valid[:, 0, 0].unbind(),
            strict=True,
        ):
            append(origin, value, valid)
        test_origins = test.origins.tolist()
        for u, released in release_schedule(test.origins, test.steps):
            for t, _ in released:
                # Test labels enter the history only at their target's release time.
                append(
                    test_origins[t],
                    test.residuals[t, 0, 0],
                    present if test.observed is None else test.observed[t, 0, 0],
                )
            pad(test_origins[u] - cadence)
            yield u, torch.stack(tuple(history)), torch.stack(tuple(observed))


@register("method", "enbpi")
class EnbPI(_ResidualIntervalMethod):
    """Journal EnbPI (Xu & Xie, TPAMI 2023, Algorithm 1).

    Uses signed out-of-bag residuals, a rolling calibration window and the narrowest
    interval on the reference's grid of five tail allocations. Empirical percentiles
    use upward-rounded percentage levels and linear interpolation. Returned intervals
    are in residual coordinates; their centers are the archive's aggregate forecasts.

    Build the archive with :func:`~tscp.data.from_bootstrap_predictions`, which performs
    the journal's mean aggregation of cached bootstrap predictions.

    Deviation:
        Forecasters are supplied, never trained here. Only one-step univariate forecasts
        are supported, with feedback released before each query rather than in batches.
        Missing residuals consume window slots and are excluded from quantiles; they
        are not imputed. The builder rejects rows with no out-of-bag predictions instead
        of using the reference implementation's zero-prediction fallback.

    Args:
        window_size: Number of latest released rows, or None for the calibration size.
            Missing observations consume slots, without backfilling older observations.
        alpha: Target miscoverage level.
    """

    def __init__(self, window_size: int | None = None, alpha: float = 0.1):
        super().__init__(window_size, alpha)
        self.interval = WidthOptimal(bins=5, round_percentiles=True)

    def _check_archive(self, archive: ResidualArchive) -> None:
        super()._check_archive(archive)
        metadata = archive.provenance.get("oob", {})
        counts = archive.outputs.get("oob_count")
        if (
            not isinstance(metadata, dict)
            or metadata.get("convention") != OOB_CONVENTION
            or counts is None
            or counts.shape != archive.residuals.shape
            or counts.is_floating_point()
            or counts.dtype == torch.bool
            or archive.forecasts is None
        ):
            raise ValueError("EnbPI needs an archive built with from_bootstrap_predictions")
        if (counts[archive.splits["calibration"]] <= 0).any():
            raise ValueError("every calibration row needs an out-of-bag prediction")
        initial_rows = torch.zeros_like(counts, dtype=torch.bool)
        initial_rows[archive.splits["calibration"]] = True
        if not torch.equal(counts > 0, initial_rows):
            raise ValueError("the calibration split must match the out-of-bag aggregation rows")

    def predict(self, archive: ResidualArchive) -> Interval:
        """Replay the test origins and return one residual interval per origin."""
        test = self._test_rows(archive)
        self.reset_online()
        bounds = test.residuals.new_empty((len(test), 2))
        for u, history, valid in self._windows(test):
            bounds[u] = self.interval(history[valid], self.alpha)
        return Interval(bounds[:, 0, None, None], bounds[:, 1, None, None])


__all__ = ["EnbPI"]

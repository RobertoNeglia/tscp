"""Residual archives from cached bootstrap predictions."""

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from tscp.data.archive import ResidualArchive, _as_cube
from tscp.data.splits import SplitPlan

OOB_CONVENTION = "enbpi_2023_mean"


def from_bootstrap_predictions(
    truth,
    member_forecasts,
    in_bag,
    *,
    splits: SplitPlan | Mapping[str, Tensor],
    origins=None,
    steps=None,
    context=None,
    context_names: tuple[str, ...] = (),
    timestamps=None,
    series: tuple[str, ...] = (),
    provenance: Mapping[str, Any] | None = None,
) -> ResidualArchive:
    """Aggregate cached bootstrap forecasts using the journal EnbPI convention.

    At calibration row ``i``, the forecast is the mean of members that did not train
    on ``i``. At other rows it is the mean of those calibration-row-specific ensembles,
    as in Xu & Xie, *Conformal prediction for time series*, TPAMI 2023, Algorithm 1.
    Residuals are always truth minus the aggregate forecast, in original units.

    No forecaster is fitted here. The caller supplies predictions from already fitted
    bootstrap members and records how their training samples were obtained in provenance.

    Args:
        truth: Targets ``[T, H, N]``, or ``[T]`` for a single step and series.
        member_forecasts: Cached predictions ``[T, H, N, B]``, or ``[T, B]`` for a
            single step and series. All members must have finite predictions.
        in_bag: Bool ``[B, C]``, True if member ``b`` trained on calibration row ``c``.
            Columns follow the order of the ``C`` rows in the calibration split.
        splits: Chronological split plan, including a calibration split.
        origins: ``[T]`` forecast origins; consecutive integers by default.
        steps: ``[H]`` horizon steps; ``1 .. H`` by default.
        context: Optional origin-known covariates.
        context_names: Names of the context columns.
        timestamps: Optional origin timestamps.
        series: Optional series names.
        provenance: JSON metadata describing the data, forecasters and bootstrap design.

    Returns:
        An ordinary archive with aggregate forecasts, signed residuals, an ``oob_count``
        output (positive on calibration rows, zero elsewhere), and aggregation metadata.

    Raises:
        ValueError: Shapes disagree, membership is not bool, or a calibration row has
            no out-of-bag members. Rejecting that last case differs from the reference
            implementation's zero-prediction fallback.
    """
    truth = _as_cube(truth, "truth")
    forecasts = torch.as_tensor(member_forecasts, device=truth.device)
    if forecasts.ndim == 2 and truth.shape[1:] == (1, 1):
        forecasts = forecasts[:, None, None, :]
    if forecasts.ndim != 4 or forecasts.shape[:3] != truth.shape or forecasts.shape[-1] < 1:
        raise ValueError("member_forecasts must have shape [T, H, N, B] with B positive")
    if not forecasts.is_floating_point() or not torch.isfinite(forecasts).all():
        raise ValueError("member_forecasts must hold finite floating predictions")
    plan = splits if isinstance(splits, SplitPlan) else SplitPlan(splits)
    plan = plan.to(truth.device)
    if "calibration" not in plan:
        raise ValueError("bootstrap predictions need a calibration split")
    rows = plan["calibration"]
    membership = torch.as_tensor(in_bag, device=truth.device)
    if membership.dtype != torch.bool or membership.shape != (forecasts.shape[-1], len(rows)):
        raise ValueError("in_bag must be bool [B, C], with C calibration rows")
    if rows.min() < 0 or rows.max() >= len(truth):
        raise ValueError("calibration rows are outside the forecasts")
    oob = ~membership.T  # [C, B]
    counts = oob.sum(dim=-1)
    if (counts == 0).any():
        raise ValueError("every calibration row needs at least one out-of-bag member")
    coefficients = oob.double() / counts[:, None]
    # The mean over C leave-one-out ensembles is a weighted mean over B members.
    predictions = forecasts.double()
    aggregate = (predictions * coefficients.mean(dim=0)).sum(dim=-1)
    aggregate[rows] = (predictions[rows] * coefficients[:, None, None, :]).sum(dim=-1)
    count_output = torch.zeros_like(truth, dtype=torch.int64)
    count_output[rows] = counts[:, None, None]
    metadata = dict(provenance or {})
    if "oob" in metadata:
        raise ValueError("provenance key 'oob' is reserved for the aggregation convention")
    metadata["oob"] = {
        "convention": OOB_CONVENTION,
        "num_members": forecasts.shape[-1],
        "calibration_size": len(rows),
    }
    return ResidualArchive.from_forecasts(
        truth,
        aggregate.to(truth.dtype),
        origins=origins,
        steps=steps,
        splits=plan,
        outputs={"oob_count": count_output},
        context=context,
        context_names=context_names,
        timestamps=timestamps,
        series=series,
        provenance=metadata,
    )


__all__ = ["OOB_CONVENTION", "from_bootstrap_predictions"]

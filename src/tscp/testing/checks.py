"""Checks that methods and components keep tscp's contracts.

Each check takes a factory, a callable that returns a fresh instance, and raises an
AssertionError that names the property that failed. The checks need no test framework::

    from tscp.testing import check_method

    check_method(lambda: MyMethod(alpha=0.1))

tscp runs its own presets and components through these checks, and a component written
elsewhere can be checked the same way.
"""

import json
import math
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

import torch
from torch import Tensor

from tscp.data import ResidualArchive
from tscp.features import FeatureMap
from tscp.methods import ConformalPredictor, load_method, save_method
from tscp.metrics import Report, evaluate
from tscp.online import OnlineUpdate
from tscp.quantile import Pool, QuantileModel, ScoreDistribution
from tscp.regions import RESIDUAL_DIMS, Region, restore_event_dims
from tscp.scores import AbsoluteScore, Score, SignedScore
from tscp.sources import HoldoutSource
from tscp.testing.archives import synthetic_archive
from tscp.utils import Component, ConfigMixin, from_config
from tscp.weights import Weights


def _check(condition, prop: str, message: str) -> None:
    if not bool(condition):
        raise AssertionError(f"{prop}: {message}")


def _devices(devices: Sequence[str] | None) -> list[str]:
    if devices is not None:
        return list(devices)
    return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def assert_regions_equal(
    first: Region, second: Region, atol: float = 0.0, rtol: float = 0.0
) -> None:
    """Assert that two batches of regions are the same, up to ``atol + rtol * |value|``."""
    if type(first) is not type(second):
        raise AssertionError(
            f"regions differ in type: {type(first).__name__} and {type(second).__name__}"
        )
    if first.event_dims != second.event_dims:
        raise AssertionError(
            f"regions cover other dimensions jointly: {first.event_dims} and {second.event_dims}"
        )
    tensors, others = first.tensors(), second.tensors()
    if tensors.keys() != others.keys():
        raise AssertionError(f"regions have other tensors: {sorted(tensors)}, {sorted(others)}")
    for name, value in tensors.items():
        other = others[name].to(value.device)
        if value.shape != other.shape:
            raise AssertionError(
                f"tensor {name!r} has shapes {tuple(value.shape)} and {tuple(other.shape)}"
            )
        if atol == 0 and rtol == 0:
            same = torch.equal(value, other)
        else:
            same = torch.allclose(value, other, rtol=rtol, atol=atol)
        if not same:
            finite = torch.isfinite(value) & torch.isfinite(other)
            gap = (value - other)[finite].abs().max().item() if finite.any() else 0.0
            raise AssertionError(
                f"tensor {name!r} differs: largest finite difference {gap:.3g}, "
                "or a difference in where it is infinite"
            )


def _same_regions(
    first: Region, second: Region, prop: str, message: str, atol=0.0, rtol=0.0
) -> None:
    try:
        assert_regions_equal(first, second, atol, rtol)
    except AssertionError as error:
        raise AssertionError(f"{prop}: {message} ({error})") from None


def _run(method: ConformalPredictor, archive: ResidualArchive) -> Region:
    return method.fit(archive).calibrate(archive).predict(archive)


def _fitted(component: Component, archive: ResidualArchive) -> Component:
    """Fit ``component`` the way the composer does, if it is trainable."""
    if component.is_trainable:
        plan = archive.splits
        validation = archive.split("validation") if "validation" in plan else None
        component.fit(archive.split("fit"), validation)
    return component


def _scramble(
    archive: ResidualArchive, targets: Tensor, rows: Tensor, seed: int
) -> ResidualArchive:
    """Replace the residuals at ``targets`` and every other field at ``rows`` with noise.

    Args:
        archive: A whole archive.
        targets: ``[T, H]`` bool, the residuals to change. Unobserved ones stay 0.
        rows: ``[T]`` bool, the rows whose forecasts, outputs and context change. Outputs
            are rescaled by positive factors, so positive outputs stay positive.
        seed: Seed of the noise.
    """
    generator = torch.Generator().manual_seed(seed)

    def noise(like: Tensor) -> Tensor:
        return torch.randn(like.shape, generator=generator).to(like.device, like.dtype)

    def at(values: Tensor) -> Tensor:
        return rows.reshape(-1, *([1] * (values.ndim - 1)))

    changed = targets[..., None].expand_as(archive.residuals)
    if archive.observed is not None:
        changed = changed & archive.observed
    fields = {"residuals": torch.where(changed, 10 * noise(archive.residuals), archive.residuals)}
    if archive.forecasts is not None:
        forecasts = archive.forecasts
        fields["forecasts"] = torch.where(at(forecasts), forecasts + noise(forecasts), forecasts)
    outputs = {}
    for name, value in archive.outputs.items():
        if value.is_floating_point():
            value = torch.where(at(value), value * (1.5 + noise(value).tanh()), value)
        outputs[name] = value
    fields["outputs"] = outputs
    if archive.context is not None and archive.context.is_floating_point():
        context = archive.context
        fields["context"] = torch.where(at(context), context + noise(context), context)
    return replace(archive, **fields)


def _default_archives() -> tuple[ResidualArchive, ...]:
    return (
        synthetic_archive(horizon=1, num_origins=240, seed=1),
        synthetic_archive(horizon=3, num_series=2, num_origins=240, missing=0.05, seed=2),
    )


# -- methods --------------------------------------------------------------------------------


def check_method(
    factory: Callable[[], ConformalPredictor],
    *,
    archives: Sequence[ResidualArchive] | None = None,
    devices: Sequence[str] | None = None,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> None:
    """Check that a method keeps tscp's contracts.

    On each archive:

    - **runs:** fit, calibrate and predict give one region per test origin, which
      :func:`~tscp.metrics.evaluate` accepts;
    - **determinism:** two fresh instances give the same regions;
    - **causality:** changing what is not yet known at a test origin (residuals whose
      target time is later, and the forecasts, outputs and context of later rows) leaves
      the regions up to that origin unchanged;
    - **split discipline:** fitting on a copy whose rows after the fitting splits are
      scrambled, then calibrating and predicting on the original, gives the same regions;
    - **config:** the config survives a JSON round trip and rebuilds the same method;
    - **save and load:** a saved, loaded and recalibrated method gives the same regions;
    - **devices:** the regions on every device agree with the CPU's, within float32
      rounding (``atol`` and ``rtol``): devices reduce in other orders.

    Args:
        factory: Returns a fresh, unfitted method.
        archives: Archives to check on. By default, two synthetic archives: one step, and
            three steps over two series with missing targets.
        devices: Devices to check; CPU, and CUDA when it is available, by default.
        atol: Absolute tolerance for comparing regions across devices.
        rtol: Relative tolerance for comparing regions across devices.
    """
    for archive in archives if archives is not None else _default_archives():
        _check_method_on(factory, archive, _devices(devices), atol, rtol)


def _check_method_on(factory, archive: ResidualArchive, devices: list[str], atol, rtol):
    method = factory()
    _check(
        isinstance(method, ConfigMixin),
        "config",
        "a method must record its config: subclass ConformalPredictor or Component",
    )
    method.fit(archive).calibrate(archive)
    fitted_state = {name: value.clone() for name, value in method.state_dict().items()}
    region = method.predict(archive)
    state = method.state_dict()
    _check(
        fitted_state.keys() == state.keys()
        and all(torch.equal(value, state[name]) for name, value in fitted_state.items()),
        "fitted state",
        "prediction changed persistent fitted state",
    )
    test = archive.split("test")
    report = evaluate(region, test, getattr(method, "alpha", 0.1))
    _check(isinstance(report, Report), "runs", "evaluate() must accept the regions")

    _same_regions(_run(factory(), archive), region, "determinism", "fresh instances disagree")
    _same_regions(method.predict(archive), region, "reset", "repeated prediction disagrees")

    for position in sorted({len(test) // 4, (3 * len(test)) // 4}):
        origin = int(test.origins[position])
        later = _scramble(
            archive, archive.target_times > origin, archive.origins > origin, seed=position
        )
        _same_regions(
            _run(factory(), later)[: position + 1],
            region[: position + 1],
            "causality",
            f"regions up to test origin {origin} change with information that comes later",
        )

    plan = archive.splits
    fitting = [plan[name] for name in ("fit", "validation") if name in plan]
    if fitting:
        last = max(int(rows[-1]) for rows in fitting)
        after = torch.arange(len(archive), device=archive.device) > last
        scrambled = _scramble(archive, after[:, None].expand(-1, archive.horizon), after, seed=11)
        other = factory().fit(scrambled).calibrate(archive).predict(archive)
        _same_regions(
            other, region, "split discipline", "fitting reads rows after the fitting splits"
        )

    config = json.loads(json.dumps(method.to_config()))
    rebuilt = from_config(config)
    _check(
        type(rebuilt) is type(method) and rebuilt.to_config() == method.to_config(),
        "config",
        "the config does not rebuild the same method",
    )
    _same_regions(_run(rebuilt, archive), region, "config", "a rebuilt method disagrees")

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "method.npz"
        save_method(method, path)
        loaded = load_method(path, archive)
    _same_regions(
        loaded.calibrate(archive).predict(archive),
        region,
        "save and load",
        "a saved, loaded and recalibrated method gives other regions",
    )

    for device in devices:
        if device == "cpu":
            continue
        moved = _run(factory().to(device), archive.to(device)).to("cpu")
        _same_regions(moved, region, f"device {device}", "regions disagree with CPU", atol, rtol)


# -- scores ---------------------------------------------------------------------------------


def check_score(
    factory: Callable[[], Score],
    *,
    archive: ResidualArchive | None = None,
    alpha: float = 0.1,
    devices: Sequence[str] | None = None,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> None:
    """Check that a score keeps the score contract.

    - **shape:** scores are floating, of shape ``[T, *kept]`` with ``kept`` from
      :meth:`~tscp.scores.Score.kept_shape`;
    - **levels:** each lies in ``[0, alpha]``, and the channels of an element share
      ``alpha``: their levels sum to at most ``alpha``;
    - **validity:** a score is never valid where a residual it combines is unobserved,
      and valid scores are finite;
    - **sublevel set:** for random thresholds ``q``, ``region(q)`` contains exactly the
      residuals whose scores are at most ``q`` in every channel;
    - **no residuals:** a region does not change when the query residuals do;
    - **config** and **devices**, as in :func:`check_method`.

    Args:
        factory: Returns a fresh score; a trainable one is fitted on the fit split.
        archive: Archive to check on; by default a synthetic archive over three steps and
            two series, with forecasts, outputs and missing targets.
        alpha: Miscoverage level for :meth:`~tscp.scores.Score.levels`.
        devices: Devices to check; CPU, and CUDA when it is available, by default.
        atol: Absolute tolerance for comparing scores across devices.
        rtol: Relative tolerance for comparing scores across devices.
    """
    if archive is None:
        archive = synthetic_archive(horizon=3, num_series=2, num_origins=240, missing=0.05, seed=3)
    score = _fitted(factory(), archive)
    calibration, query = archive.split("calibration"), archive.split("test")
    kept = score.kept_shape(archive.horizon, archive.num_series)

    values = score(calibration)
    expected_shape = (len(calibration), *kept)
    _check(values.is_floating_point(), "shape", "scores must be floating")
    _check(
        tuple(values.shape) == expected_shape,
        "shape",
        f"scores have shape {tuple(values.shape)}, not {expected_shape}",
    )

    levels = score.levels(alpha, kept)
    _check(
        levels.dtype == torch.float64 and tuple(levels.shape) == kept,
        "levels",
        f"levels must be float64 of shape {kept}",
    )
    _check(
        ((levels >= 0) & (levels <= alpha + 1e-12)).all(),
        "levels",
        "each level must lie in [0, alpha]",
    )
    if score.channels:
        _check(
            (levels.sum(dim=-1) <= alpha + 1e-12).all(),
            "levels",
            "the channels of an element must share alpha, but their levels sum to more",
        )

    valid = score.valid(calibration)
    implied = Score.valid(score, calibration)  # every residual the score combines observed
    if valid is not None:
        _check(
            valid.dtype == torch.bool and valid.shape == values.shape,
            "validity",
            "valid() must return a bool mask shaped like the scores, or None",
        )
        if implied is not None:
            _check(
                not (valid & ~implied).any(),
                "validity",
                "a score is valid where a residual it combines is unobserved",
            )
    defined = valid if valid is not None else torch.ones_like(values, dtype=torch.bool)
    _check(torch.isfinite(values[defined]).all(), "validity", "valid scores must be finite")

    query_values = score(query)
    generator = torch.Generator().manual_seed(0)
    finite = query_values[torch.isfinite(query_values)]
    spread = float(finite.std()) if finite.numel() > 1 else 1.0
    jitter = torch.randn(query_values.shape, generator=generator).to(query_values)
    thresholds = query_values + max(spread, 1e-3) * jitter
    region = score.region(thresholds, query)
    expected = query_values <= thresholds
    query_valid = score.valid(query)
    if score.channels:
        expected = expected.all(dim=-1)
        if query_valid is not None:
            query_valid = query_valid.all(dim=-1)
    _check(
        tuple(region.batch_shape) == tuple(expected.shape),
        "sublevel set",
        f"regions have batch shape {tuple(region.batch_shape)}, not {tuple(expected.shape)}",
    )
    agree = region.contains(query.residuals) == expected
    if query_valid is not None:
        agree = agree | ~query_valid
    _check(
        agree.all(),
        "sublevel set",
        f"region(q) must contain exactly the residuals whose scores are at most q; "
        f"{int((~agree).sum())} of {agree.numel()} disagree",
    )

    blind = replace(query, residuals=_scramble_residuals(query, seed=1))
    _same_regions(
        score.region(thresholds, blind),
        region,
        "no residuals",
        "a region must not read the residuals of its query rows",
    )

    rebuilt = _fitted(from_config(json.loads(json.dumps(score.to_config()))), archive)
    _check(type(rebuilt) is type(score), "config", "the config does not rebuild the score")
    _check(
        torch.equal(rebuilt(calibration), values),
        "config",
        "a score rebuilt from its config gives other scores",
    )

    for device in _devices(devices):
        if device == "cpu":
            continue
        moved = _fitted(factory().to(device), archive.to(device))
        on_device = moved(calibration.to(device)).to("cpu")
        _check(
            torch.allclose(on_device[defined], values[defined], rtol=rtol, atol=atol),
            f"device {device}",
            "scores disagree with CPU",
        )


def _scramble_residuals(rows: ResidualArchive, seed: int) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(rows.residuals.shape, generator=generator).to(rows.residuals)
    if rows.observed is None:
        return noise
    return noise * rows.observed


# -- weights --------------------------------------------------------------------------------


def check_weights(
    factory: Callable[[], Weights],
    *,
    archive: ResidualArchive | None = None,
) -> None:
    """Check that a weight scheme keeps the weights contract.

    - **shape:** weights are floating, ``[Q, P]`` or ``[Q, P, ...]`` for ``Q`` query rows and
      ``P`` pool rows;
    - **values:** finite and nonnegative;
    - **causality:** changing what is not yet known at a query leaves its weights unchanged;
    - **config:** a weight scheme rebuilt from its config gives the same weights.

    Args:
        factory: Returns fresh weights; trainable ones are fitted on the fit split.
        archive: Archive to check on; a synthetic one by default.
    """
    if archive is None:
        archive = synthetic_archive(horizon=2, num_series=2, num_origins=240, seed=4)

    def weights_of(component, source):
        return component(source.split("calibration"), source.split("test"))

    weights = _fitted(factory(), archive)
    values = weights_of(weights, archive)
    pool, query = archive.split("calibration"), archive.split("test")
    _check(
        values.is_floating_point() and values.ndim >= 2,
        "shape",
        "weights must be a floating tensor [Q, P] or [Q, P, ...]",
    )
    _check(
        tuple(values.shape[:2]) == (len(query), len(pool)),
        "shape",
        f"weights start with shape {tuple(values.shape[:2])}, not {(len(query), len(pool))}",
    )
    _check(
        (torch.isfinite(values) & (values >= 0)).all(),
        "values",
        "weights must be finite and nonnegative",
    )
    position = len(query) // 2
    origin = int(query.origins[position])
    later = _scramble(archive, archive.target_times > origin, archive.origins > origin, seed=5)
    changed = weights_of(_fitted(factory(), later), later)
    _check(
        torch.equal(changed[: position + 1], values[: position + 1]),
        "causality",
        f"weights of queries up to origin {origin} change with information that comes later",
    )
    rebuilt = _fitted(from_config(json.loads(json.dumps(weights.to_config()))), archive)
    _check(
        torch.equal(weights_of(rebuilt, archive), values),
        "config",
        "weights rebuilt from their config differ",
    )


# -- quantile models ------------------------------------------------------------------------


def check_quantile_model(
    factory: Callable[[], QuantileModel],
    *,
    archive: ResidualArchive | None = None,
    num_queries: int = 8,
) -> None:
    """Check that a quantile model keeps the quantile-model contract.

    With equal and with random weights:

    - **distribution:** the model returns a :class:`~tscp.quantile.ScoreDistribution` of
      batch shape ``[Q, *kept]``, or ``[*kept]`` when shared by every query;
    - **levels:** several levels at once equal one level at a time;
    - **monotone:** thresholds never increase with ``alpha``;
    - **conventions:** ``alpha <= 0`` gives ``+inf`` and ``alpha >= 1`` gives ``-inf``, which
      the long-run guarantee of the ACI family relies on;
    - **level:** ``level(s)`` is where the threshold stops covering ``s``;
    - **select:** ``select(i)`` is the distribution of query ``i``.

    Args:
        factory: Returns a fresh quantile model; a trainable one is fitted on the fit split.
        archive: Archive to check on; a synthetic one by default.
        num_queries: Number of query rows.
    """
    if archive is None:
        archive = synthetic_archive(horizon=2, num_series=2, num_origins=240, seed=6)
    model = _fitted(factory(), archive)
    rows = archive.split("calibration")
    query = archive.split("test").select(torch.arange(num_queries, device=archive.device))
    scores = AbsoluteScore()(rows)
    pool = Pool(scores, rows)
    kept = tuple(scores.shape[1:])
    generator = torch.Generator().manual_seed(0)
    random_weights = 0.1 + torch.rand((num_queries, len(pool)), generator=generator)
    alphas = torch.tensor([0.02, 0.1, 0.3, 0.6], dtype=torch.float64)
    for label, weights in (("equal weights", None), ("random weights", random_weights.double())):
        distribution = model(pool, weights, query)
        _check(
            isinstance(distribution, ScoreDistribution),
            "distribution",
            f"with {label}, the model must return a ScoreDistribution",
        )
        batch = tuple(distribution.batch_shape)
        _check(
            batch in (kept, (num_queries, *kept)),
            "distribution",
            f"with {label}, batch shape {batch} is neither {kept} nor {(num_queries, *kept)}",
        )
        grid = alphas.reshape(-1, *([1] * len(batch)))
        together = distribution.threshold(grid)
        _check(
            tuple(together.shape) == (len(alphas), *batch),
            "levels",
            f"with {label}, thresholds at {len(alphas)} levels have shape {tuple(together.shape)}",
        )
        for index, alpha in enumerate(alphas.tolist()):
            _check(
                torch.equal(together[index], distribution.threshold(alpha)),
                "levels",
                f"with {label}, several levels at once disagree with one level at a time",
            )
        _check(
            (together[:-1] >= together[1:]).all(),
            "monotone",
            f"with {label}, thresholds must not increase with alpha",
        )
        _check(
            torch.isposinf(distribution.threshold(0.0)).all(),
            "conventions",
            f"with {label}, alpha <= 0 must give +inf (an unbounded region)",
        )
        _check(
            torch.isneginf(distribution.threshold(1.0)).all(),
            "conventions",
            f"with {label}, alpha >= 1 must give -inf (an empty region)",
        )
        for row in torch.randperm(len(pool), generator=generator)[:5].tolist():
            probe = scores[row].double()
            level = distribution.level(probe)
            _check(
                ((level > 0) & (level <= 1)).all(),
                "level",
                f"with {label}, levels must lie in (0, 1]",
            )
            below, above = level - 1e-6, level + 1e-6
            covered = distribution.threshold(below.clamp(min=1e-9)) >= probe
            missed = distribution.threshold(above.clamp(max=1 - 1e-9)) < probe
            _check(
                (covered | (below <= 0)).all() and (missed | (above >= 1)).all(),
                "level",
                f"with {label}, level(s) must be where the threshold stops covering s",
            )
        if len(batch) == len(kept) + 1:
            for index in (0, num_queries - 1):
                one = distribution.select(index)
                _check(
                    tuple(one.batch_shape) == kept
                    and torch.equal(one.threshold(0.1), distribution.threshold(0.1)[index]),
                    "select",
                    f"with {label}, select({index}) must be the distribution of query {index}",
                )


# -- online updates -------------------------------------------------------------------------


def check_online_update(
    factory: Callable[[], OnlineUpdate],
    *,
    archives: Sequence[ResidualArchive] | None = None,
    devices: Sequence[str] | None = None,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> None:
    """Check that an online update keeps the online-update contract.

    - **method contracts:** :func:`check_method` passes for the update inside a
      :class:`~tscp.methods.ConformalPredictor`, with an absolute score and with a signed
      score whose two channels have their own levels;
    - **reset:** predicting twice with the same instance gives the same regions;
    - **response:** a miss changes later thresholds, and never earlier ones.

    Args:
        factory: Returns a fresh online update.
        archives: Archives for :func:`check_method`.
        devices: Devices for :func:`check_method`.
        atol: Absolute tolerance for comparing regions across devices.
        rtol: Relative tolerance for comparing regions across devices.
    """

    def absolute():
        return ConformalPredictor(AbsoluteScore(), source=HoldoutSource(), update=factory())

    def signed():
        return ConformalPredictor(SignedScore(), source=HoldoutSource(), update=factory())

    for method in (absolute, signed):
        check_method(method, archives=archives, devices=devices, atol=atol, rtol=rtol)

    archive = synthetic_archive(horizon=2, num_origins=240, seed=7)
    method = absolute()
    first = _run(method, archive)
    _same_regions(method.predict(archive), first, "reset", "reset() must clear the state")

    test_rows = archive.splits["test"]
    spike = len(test_rows) // 3
    residuals = archive.residuals.clone()
    residuals[test_rows[spike]] = 1e6
    spiked = _run(absolute(), replace(archive, residuals=residuals))
    differs = torch.zeros(len(test_rows), dtype=torch.bool)
    for name, value in first.tensors().items():
        differs |= (value != spiked.tensors()[name]).reshape(len(test_rows), -1).any(dim=1)
    _check(
        not differs[: spike + 1].any(),
        "causality",
        "a miss changes thresholds before its outcome is known",
    )
    _check(
        differs[spike + 1 :].any(),
        "response",
        "a miss never changes a later threshold; an online update must react to its outcomes",
    )


# -- regions --------------------------------------------------------------------------------


def check_region(
    factory: Callable[[], Region],
    *,
    residual_shape: tuple[int, int, int] = (6, 3, 2),
    enclosing: tuple[Tensor, Tensor] | None = None,
    samples: int = 4000,
    devices: Sequence[str] | None = None,
) -> None:
    """Check that a region type keeps the region contract.

    - **shapes:** the batch shape is ``[Q, *kept]`` for residuals of ``residual_shape``,
      :attr:`~tscp.regions.Region.dimension` is the number of coordinates covered jointly,
      and :meth:`~tscp.regions.Region.contains` and
      :meth:`~tscp.regions.Region.log_volume` answer per region;
    - **shift:** :meth:`~tscp.regions.Region.to_target` keeps membership and volume;
    - **indexing:** selecting regions selects their tensors along the batch dimensions;
    - **devices:** the regions move to every device and back unchanged;
    - **volume:** with ``enclosing`` bounds, a Monte Carlo estimate of each bounded region's
      volume agrees with ``log_volume`` within five standard errors.

    Args:
        factory: Returns the regions to check.
        residual_shape: ``(Q, H, N)`` of the residuals the regions are for.
        enclosing: ``(lower, upper)`` of boxes that contain the regions, each
            ``[*batch, *event]``, for the volume check.
        samples: Monte Carlo samples per region.
        devices: Devices to check; CPU, and CUDA when it is available, by default.
    """
    region = factory()
    num_queries, horizon, num_series = residual_shape
    sizes = dict(zip(RESIDUAL_DIMS, (horizon, num_series), strict=True))
    kept = [sizes[name] for name in RESIDUAL_DIMS if name not in region.event_dims]
    _check(
        tuple(region.batch_shape) == (num_queries, *kept),
        "shapes",
        f"batch shape {tuple(region.batch_shape)} is not {(num_queries, *kept)}",
    )
    joint = math.prod(sizes[name] for name in region.event_dims)
    _check(region.dimension == joint, "shapes", f"dimension {region.dimension} is not {joint}")
    generator = torch.Generator().manual_seed(0)
    points = torch.randn(residual_shape, generator=generator)
    covered = region.contains(points)
    _check(
        covered.dtype == torch.bool and covered.shape == region.batch_shape,
        "shapes",
        "contains() must return a bool tensor of the batch shape",
    )
    log_volume = region.log_volume()
    _check(
        log_volume.shape == region.batch_shape and not log_volume.isnan().any(),
        "shapes",
        "log_volume() must return a tensor of the batch shape without NaN",
    )

    forecasts = 10 * torch.randn(residual_shape, generator=generator)
    shifted = region.to_target(forecasts)
    _check(
        type(shifted) is type(region) and shifted.event_dims == region.event_dims,
        "shift",
        "to_target() must return a region of the same type",
    )
    _check(
        torch.equal(shifted.contains(points + forecasts), covered),
        "shift",
        "a shifted region must cover the shifted points",
    )
    finite = torch.isfinite(log_volume)
    _check(
        torch.allclose(shifted.log_volume()[finite], log_volume[finite], atol=1e-4)
        and torch.equal(shifted.log_volume()[~finite], log_volume[~finite]),
        "shift",
        "shifting must not change the volume",
    )

    tensors = region.tensors()
    mask = torch.arange(num_queries) % 2 == 0
    for index in (1, slice(1, 3), mask):
        selected = region[index]
        _check(
            type(selected) is type(region)
            and all(torch.equal(v, tensors[k][index]) for k, v in selected.tensors().items()),
            "indexing",
            f"region[{index!r}] must select the tensors along the batch dimensions",
        )

    for device in _devices(devices):
        back = region.to(device).to("cpu")
        _check(
            all(torch.equal(v, tensors[k]) for k, v in back.tensors().items()),
            f"device {device}",
            "moving the regions changed them",
        )

    if enclosing is not None:
        lower, upper = enclosing
        uniform = torch.rand(samples, *lower.shape, generator=generator).to(lower)
        drawn = restore_event_dims(lower + uniform * (upper - lower), region.event_dims)
        inside = torch.stack([region.contains(point) for point in drawn]).double().mean(dim=0)
        width = (upper - lower).reshape(*region.batch_shape, -1).double()
        enclosing_volume = width.log().sum(dim=-1).exp()
        estimate = inside * enclosing_volume
        error = (inside * (1 - inside) / samples).sqrt() * enclosing_volume
        exact = log_volume.double().exp()
        _check(
            ((estimate - exact).abs() <= 5 * error + 1e-9)[finite].all(),
            "volume",
            "Monte Carlo volumes disagree with log_volume()",
        )


# -- feature maps ---------------------------------------------------------------------------


def check_feature_map(
    factory: Callable[[], FeatureMap],
    *,
    archive: ResidualArchive | None = None,
    devices: Sequence[str] | None = None,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> None:
    """Check that a feature map keeps the feature-map contract.

    - **shape:** features are floating, ``[T, *element, F]`` with element ``()``, ``(N,)``
      or ``(H, N)``;
    - **values:** finite, with what is unknown filled in and marked invalid;
    - **validity:** ``valid`` is None or a bool mask ``[T, *element]``;
    - **causality:** changing what is not yet known at an origin leaves the features of the
      rows up to that origin unchanged;
    - **config** and **devices**, as in :func:`check_method`.

    Args:
        factory: Returns a fresh feature map; a trainable one is fitted on the fit split.
        archive: Archive to check on; by default a synthetic archive over three steps and
            two series, with context, timestamps and missing targets.
        devices: Devices to check; CPU, and CUDA when it is available, by default.
        atol: Absolute tolerance for comparing features across devices.
        rtol: Relative tolerance for comparing features across devices.
    """
    if archive is None:
        archive = synthetic_archive(horizon=3, num_series=2, num_origins=240, missing=0.05, seed=8)
    feature_map = _fitted(factory(), archive)
    rows = archive.split("test")
    values = feature_map(rows)
    allowed = [(), (archive.num_series,), (archive.horizon, archive.num_series)]
    element = tuple(values.shape[1:-1])
    _check(
        values.is_floating_point() and values.ndim >= 2 and values.shape[0] == len(rows),
        "shape",
        f"features must be a floating tensor [T, *element, F], not {tuple(values.shape)}",
    )
    _check(element in allowed, "shape", f"element {element} is none of {allowed}")
    _check(
        torch.isfinite(values).all(),
        "values",
        "features must be finite: fill what is unknown and mark it invalid",
    )
    valid = feature_map.valid(rows)
    if valid is not None:
        _check(
            valid.dtype == torch.bool and tuple(valid.shape) == (len(rows), *element),
            "validity",
            f"valid() must return None or a bool mask of shape {(len(rows), *element)}",
        )

    position = len(rows) // 2
    origin = int(rows.origins[position])
    later = _scramble(archive, archive.target_times > origin, archive.origins > origin, seed=9)
    changed = _fitted(factory(), later)(later.split("test"))
    _check(
        torch.equal(changed[: position + 1], values[: position + 1]),
        "causality",
        f"features of rows up to origin {origin} change with information that comes later",
    )

    rebuilt = _fitted(from_config(json.loads(json.dumps(feature_map.to_config()))), archive)
    _check(torch.equal(rebuilt(rows), values), "config", "a rebuilt feature map differs")

    for device in _devices(devices):
        if device == "cpu":
            continue
        moved = _fitted(factory().to(device), archive.to(device))(rows.to(device)).to("cpu")
        _check(
            torch.allclose(moved, values, rtol=rtol, atol=atol),
            f"device {device}",
            "features disagree with CPU",
        )


__all__ = [
    "assert_regions_equal",
    "check_feature_map",
    "check_method",
    "check_online_update",
    "check_quantile_model",
    "check_region",
    "check_score",
    "check_weights",
]

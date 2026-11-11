# Concepts

tscp is post-hoc: it starts from the forecasts of a model that is already trained, and it
never trains the forecaster. A conformal method in tscp is one choice per *axis*, run
through a fixed lifecycle on a *residual archive*.

## The residual archive

A {class}`~tscp.data.ResidualArchive` holds the residuals of one forecaster on one
dataset, with what methods need beside them.

- **Residuals** are truth minus forecast, in the target's original units, with shape
  `[T, H, N]`: forecast origins, horizon steps and series. tscp never standardizes them.
- **Origins and steps** count base time steps. The target of origin `s` at step `h` is at
  time `s + h`.
- **Splits** are named row sets in time order: `fit` and an optional `validation` for
  components that are trained, `calibration` for conformal scores, and `test`. No two
  splits share a target time.
- **Optional fields:** point forecasts, other forecaster outputs (such as predicted
  quantiles, in target units), context known at each origin, an observed mask for missing
  targets, timestamps, series names and a graph over the series.

{meth}`~tscp.data.ResidualArchive.from_forecasts` builds an archive from a backtest's
arrays, and {func}`~tscp.data.save_archive` writes it to a self-contained `.npz` file,
checked against a fingerprint when it is read back.

## Knowing only the past

The residual of origin `s` at step `h` is known once its target time has passed: at origin
`t` it is usable only if `s + h <= t`. Everything that grows or adapts during the test
period follows this rule: expanding calibration pools, online updates and features.

A row subset, made by `select` or `split`, keeps the archive it came from as its
`timeline`. Components read the history before a row from there, through
{meth}`~tscp.data.ResidualArchive.known_counts`. `split` cuts the timeline after the
split's last row, so a component fitted on the fit split cannot reach anything later.
{func}`tscp.testing.check_method` checks both properties for any method.

## One choice per axis

| Axis | Base class | What it decides |
|---|---|---|
| Residual source | {class}`~tscp.sources.Source` | which rows calibrate, and whether test residuals join them once known |
| Score | {class}`~tscp.scores.Score` | how an error becomes a number, and the region a threshold on it defines |
| Weights | {class}`~tscp.weights.Weights` | which past errors count, and how much |
| Quantile model | {class}`~tscp.quantile.QuantileModel` | the conformal distribution of the score, read at any level |
| Online update | {class}`~tscp.online.OnlineUpdate` | how the threshold reacts to its own misses |
| Region | {class}`~tscp.regions.Region` | what the user gets: a membership test and a size |

A score keeps the dimensions of `[H, N]` it does not reduce: an interval per step and
series, or one joint region over the horizon. A score may also split an element into
channels, such as the two sides of an interval, and share the miscoverage among them.
Feature maps ({class}`~tscp.features.FeatureMap`) turn what is known at each origin
into vectors for weights and quantile models.

## The lifecycle

```python
method.fit(archive)  # trainable components see the fit and validation splits only
method.calibrate(archive)  # scores of the calibration pool
region = method.predict(archive)  # one region per test origin
report = evaluate(region, archive.split("test"), alpha)
```

Without an online update, `predict` reads every threshold at once. With one, it answers the
test origins in time order and hands the update each outcome once its target time has
passed. {func}`~tscp.metrics.evaluate` returns a {class}`~tscp.metrics.Report`: coverage,
coverage per step and over the horizon, region size as log-volume per dimension, and
conditional coverage on request.

A fitted method is saved with {func}`~tscp.methods.save_method` and loaded with
{func}`~tscp.methods.load_method`; calibration is cheap, so a loaded method is
calibrated again before it predicts.

## Regions

Regions live in residual coordinates: a target `y` with forecast `f` is covered when
`y - f` lies in the region. Coverage and volume do not change under this shift, so
evaluation never needs the forecasts, and `region.to_target(forecasts)` maps regions back
to target units. Region size is reported as log-volume per dimension, never as the mean of
raw volumes, which the widest regions would dominate.

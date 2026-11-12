# Calibration methods

All methods consume forecasts or residual archives from a model trained outside tscp.
Residuals are truth minus prediction in the original target units. Fit any initial
learned components on the fitting split, choose settings on fitting/validation data,
then calibrate and predict chronologically.

## Composable presets

| Registry name | Preset | Main arguments |
|---|---|---|
| `split` | `SplitCP` | `score`, `alpha` |
| `nexcp` | `NexCP` | `decay`, `score`, `source`, `alpha` |
| `aci` | `ACI` | `gamma`, `score`, `source`, `alpha` |
| `windowed` | `WindowedCP` | required `window_size`, `score`, `alpha` |
| `mondrian` | `MondrianCP` | required `groups`, `score`, `source`, `alpha` |
| `dtaci` | `DtACI` | `gammas`, `eta`, `sigma`, `score`, `source`, `alpha` |
| `pi` | `ConformalPI` | `lr`, `csat`, `ki`, `warm_start`, `score`, `alpha` |
| `pid` | `ConformalPID` | PI settings, `scorecaster`, `score`, `alpha` |

Use `SplitCP(score=SignedScore())` for two-sided calibration,
`SplitCP(score=NormalizedScore(...))` for normalized residuals, and
`SplitCP(score=CQRScore(...))` for externally supplied quantile forecasts.

`ExpandingSource(window_size=W)` retains the latest W available calibration/test
rows for each release group. Availability is computed before missing-score masks:
missing targets consume slots and older rows are not substituted. This limits
calibration support, not total archive memory.

Mondrian calibration matches tuples of discrete origin-level features, such as
`Phase(24)`, integer `Calendar(["hour"])` or integer-valued context columns. It does
not infer bins. Unseen groups give unbounded intervals; small groups may also give
unbounded intervals because the test-point mass is retained. Its finite-sample
category-conditional guarantee requires exchangeability within each fixed group.
Rolling windows alone do not provide that guarantee on arbitrary dependent data.

```{literalinclude} ../../examples/calibration.py
:language: python
```

Run this example with `--config examples/configs/calibration.yaml`.

## Online adaptation

Initial fitting and parameter selection use fitting/validation splits. During
prediction, explicitly online components may update from calibration observations
and test outcomes whose target time has arrived. Every prediction starts a fresh
replay. Runtime histories and online estimators are not saved as fitted state.

Adaptive weights and quantile components opt in with `adapts_online=True` and clear
runtime state in `reset_online()`. Their inputs hide future origins and mark
unreleased targets as missing. Their distributions are constructed after releases.
Threshold updates receive query times through `prepare_query`; a nested
`Scorecaster` receives available calibration history and released scores.

DtACI uses the deterministic level-aggregation algorithm from Gibbs and Candès
(2024, Algorithm 2). Expert misses use the thresholds issued at the original query;
expert weighting uses that query's realized level. The default fixed learning and
mixing rates do not imply exact asymptotic target coverage.

PI combines quantile tracking and the tangent error integrator. PID adds a scorecast.
`ThetaScorecaster` refits a CPU Theta model from released score history and returns
torch outputs on the method's device. Each fit uses one contiguous run of observed
history: while the run after a gap, such as the split gap, is still short, forecasts
continue from the earlier run. It returns zero during warm-up, so PID then behaves as
PI. Install the `pid` extra for Theta; ordinary PI needs no optional package.
The controller's documented indexing and initialization match its reference fixture.

By default PI and PID start from a zero threshold, as in the reference
implementation, so the first test intervals are narrow until the tracker catches up.
`warm_start=True` makes the first threshold the split-conformal quantile of the
calibration scores and scales the first learning rates by their range; the tracker
absorbs the first scorecast. The integrator still counts test errors only.

From a source checkout, install optional backends with
`pip install -e ".[forest,pid]"`. The forest extra pins the author estimator and bounds
scikit-learn below 1.9 because that estimator uses an internal bootstrap helper whose
signature changed in 1.9.

## Specialized intervals

These presets initially accept one horizon step (`steps=[1]`) and one series.
They expose the same fit/calibrate/predict lifecycle and return residual-coordinate
intervals, with dedicated interval construction rather than arbitrary score choices.

**AgACI** combines ACI experts with separate lower/upper Bernstein aggregators.
Each expert adapts its whole interval at alpha. Construct it with required
`endpoint_bounds=(lower, upper)` in target coordinates and provide archive forecasts.
These values replace nonfinite expert endpoints; finite endpoints are retained.

**EnbPI** implements the 2023 journal construction. Supply predictions from an
externally trained bootstrap ensemble to `tscp.data.from_bootstrap_predictions`:
`member_forecasts` has shape `[T,H,N,B]`, and bool `in_bag [B,C]` states membership
for calibration rows in their split order. The builder computes row-specific OOB
calibration forecasts and the journal test-forecast aggregation. Rows with zero OOB
members are rejected. Membership consistency can be checked; model training remains
the caller's responsibility.

`EnbPI(window_size=None)` uses the calibration row count as its rolling window.
It searches five tail allocations using upward-rounded percentile levels and linear
interpolation, without the split-conformal test-point correction. Missing rows consume
slots and are excluded from quantiles.

**SPCI** requires explicit `lags` and `window_size`. At each query it fits the exact
author `sklearn-quantile` estimator on valid lag/target pairs inside the rolling window.
The forecaster remains fixed. Lags and the window are slots on the origin grid, whose
spacing is the greatest common divisor of the calibration and test origin spacings:
an absent origin, such as a row between the two splits, is a missing slot, so lag
vectors never join residuals that are not adjacent in time. Missing query lags or fewer
than two valid training pairs give an unbounded interval. Install the `forest` extra. The large-sample sampled-forest
shortcut is omitted, and every refit uses a recorded seed.

`QuantileForest` and `WidthOptimal` are helpers for signed-residual interval methods.
Width optimization is separate from `ScoreDistribution.level`, since the selected
intervals need not be nested as alpha changes. Optional forest and Theta backends
convert to CPU/NumPy internally and return torch tensors.

## Reference checks

Independent fixtures under `tests/fixtures` record source revisions, inputs,
parameters, numerical conventions, tolerances and deviations. Regeneration scripts
run separately from ordinary tests. The original golden-region fixture continues to
test regression against earlier tscp behavior.

PID, EnbPI and SPCI run the authors' pinned code, DtACI the authors' pinned R code,
and AgACI's endpoint aggregation the OPERA package. Split, signed, normalized, CQR,
NexCP, ACI, windowed and Mondrian calibration, and AgACI's experts, are checked against
independent evaluations of their equations.

The fixtures verify conformal estimation conditional on supplied forecasts, not the
forecast-training experiments in the papers. Method docstrings give citations and
method-specific deviations. All supported presets also share causality, fitting-split,
config, persistence, replay-reset and device checks.

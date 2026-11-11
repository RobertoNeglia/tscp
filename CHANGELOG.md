# Changelog

All notable changes to tscp are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Package layout under `src/tscp/`: one subpackage per design axis (`sources`, `scores`,
  `weights`, `quantile`, `online`, `regions`), plus `data`, `methods`, `metrics`,
  `features`, `datasets` and `utils`.
- `ResidualArchive`, `SplitPlan` with `chronological_split`, the availability rule, and
  self-contained `.npz` archives with fingerprints.
- Row subsets that keep the archive they came from as their `timeline`, with
  `known_counts` for reading the history before a row causally. `split()` cuts the
  timeline after the split's last row; `head()` returns the first rows.
- The `ConformalPredictor` composer, and the presets `SplitCP`, `NexCP` and `ACI`,
  registered in `method_map` and built with `get_method`. Trainable components are fitted
  on the `fit` split, with the `validation` split when there is one.
- Scores that read whole rows (forecasts, outputs, context), with channels and
  per-element miscoverage levels: `AbsoluteScore`, `SignedScore`, `NormalizedScore`
  (forecaster scale or fitted scale) and `CQRScore` (symmetric or per side). Scores over
  the horizon or across series are calibrated on rows whose targets are all observed.
- Quantile models return a `ScoreDistribution`: thresholds at any number of levels from
  one sort, and the conformal p-value of any score. `WeightedQuantile` returns an
  `EmpiricalDistribution`.
- Online updates answer queries keyed by their position and observe outcomes as their
  targets arrive, with the realized levels on request: `ACIUpdate`, run by the online
  engine in `tscp.methods.engine`.
- Other components: `HoldoutSource`, `ExpandingSource`, `UniformWeights`,
  `ExponentialWeights`.
- Regions: `Interval`, and `Box` over steps, series or both. Every region exposes its
  tensors, so indexing and device moves work for every type.
- Metrics: coverage, log-width, Winkler score, rolling, worst-window and grouped coverage,
  and `evaluate`, which returns a `Report` with per-step and joint coverage for any
  region type.
- Synthetic archives: a Gaussian control and a variance shift.
- Feature maps, computed causally from a row's timeline: `ResidualHistory` (the last
  residuals known at each origin, per step), `Phase` (position in a cycle of base
  steps), `Calendar` (minute to day of year, from timestamps), `ContextColumns` and
  `Concat`.
- Optional archive `timestamps` (int64 nanoseconds, hashed). Archive files are schema
  version 2; version 1 files still load.
- `tscp.nn` for learned components: `pinball_loss`, `interval_score`, `RowBatches` of
  archive rows with their features, `train` (seeded, with early stopping on the
  validation split) and `seeded`. `examples/learned_score.py` fits a score whose scale is
  a small network on the context.
- `context_scale`: a synthetic control whose error scale follows a known covariate.
- Batch production: `ResidualArchive.from_forecasts` builds an archive from a backtest's
  arrays (NaN truth means unobserved; timestamps as int64 nanoseconds or `datetime64`),
  and `targets()` gives the truth back. `save_method` and `load_method` keep a fitted
  method in one `.npz` file, without pickle; components whose fitted state lives outside
  torch set `refit_on_load`. Fresh components load fitted state whose shape depends on
  the data. `examples/production.py` runs the whole workflow.
- A `py.typed` marker, for type checkers.
- A release workflow that publishes to PyPI through trusted publishing when a version
  tag is pushed.
- `tscp.testing`: checks that a method or component keeps tscp's contracts
  (`check_method`, `check_score`, `check_weights`, `check_quantile_model`,
  `check_online_update`, `check_region`), with `synthetic_archive` and `MaxAbsScore`.
  tscp checks its own presets and components with them.
- `tscp.registry`: components and presets registered by kind and short name. `register`
  adds new ones; a taken name raises unless `replace=True`. Plugins register through
  the `tscp.plugins` entry-point group and are loaded when a name or a config's class is
  not found. `compose` builds a method from a spec of short names, such as one read from
  YAML; `examples/plugin` is an installable example plugin.
- Recorded constructor configs (`to_config`, `from_config`) and config hashes.
- Sphinx docs with a concepts page and tutorials (composing a method, writing a component,
  a learned component and a plugin, batch production) whose code lives in `examples/`
  and runs in the tests; CI that builds them with warnings as errors; pre-commit.

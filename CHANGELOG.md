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
- Recorded constructor configs (`to_config`, `from_config`) and config hashes.
- Sphinx docs, CI, pre-commit and a quickstart example.

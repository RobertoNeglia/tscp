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
- The `ConformalPredictor` composer, and the presets `SplitCP`, `NexCP` and `ACI`,
  registered in `method_map` and built with `get_method`.
- Components: `AbsoluteScore`, `HoldoutSource`, `ExpandingSource`, `UniformWeights`,
  `ExponentialWeights`, `WeightedQuantile`, `ACIUpdate`, `Interval`.
- Metrics: coverage, log-width, Winkler score, rolling, worst-window and grouped coverage,
  and `evaluate`.
- Synthetic archives: a Gaussian control and a variance shift.
- Recorded constructor configs (`to_config`, `from_config`) and config hashes.
- Sphinx docs, CI, pre-commit and a quickstart example.

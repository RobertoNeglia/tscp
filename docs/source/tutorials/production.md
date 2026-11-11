# Batch production

tscp works on a forecaster's backtest: its forecasts and the truth, as plain arrays. A
production job fits a method once and saves it; later jobs load it, calibrate it on the
latest archive and predict.

```{literalinclude} ../../../examples/production.py
:language: python
:pyobject: main
:start-at: archive = ResidualArchive.from_forecasts
```

- {meth}`~tscp.data.ResidualArchive.from_forecasts` takes `[T]`, `[T, H]` or `[T, H, N]`
  arrays; a NaN in the truth marks a target that was not observed. Timestamps can be
  int64 nanoseconds or a `numpy.datetime64` array.
- {func}`~tscp.methods.save_method` writes one `.npz` file, without pickle, with the
  method's config and fitted state; existing files are never overwritten.
  {func}`~tscp.methods.load_method` checks the state against its fingerprint. Components
  whose fitted state lives outside torch are fitted again on the archive's fit split.
- `region.to_target(forecasts)` gives the regions in the target's units.

```bash
python examples/production.py --config examples/configs/production.yaml
```

The method here is composed from the config; any preset or composed method works the same
way.

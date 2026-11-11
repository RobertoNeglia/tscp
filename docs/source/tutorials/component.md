# Write a component

A component is one choice on one axis. Writing one means subclassing the axis's base
class and implementing a method or two; everything else (fitting, masks for unavailable
and missing data, online release, devices, configs) is the composer's job.

This weight scheme counts only the past errors at the same point of a cycle, such as the
same hour of the day, and lets older cycles count less:

```{literalinclude} ../../../examples/custom_component.py
:language: python
:pyobject: PeriodicWeights
```

Its constructor takes only JSON-friendly arguments, so its config records and rebuilds
it. `forward` returns nonnegative, unnormalized weights of each pool row for each query;
the composer multiplies them by the availability and observed masks, so the component
never handles either.

## Check it

{mod}`tscp.testing` checks that a component keeps its axis's contract, and that a method
keeps tscp's: causality, split discipline, config round trips, saving and loading, and
devices. A check raises an `AssertionError` that names the first property that fails.

```{literalinclude} ../../../examples/custom_component.py
:language: python
:pyobject: main
:start-at: def periodic_method
:end-at: print("PeriodicWeights
```

There is a check per axis: {func}`~tscp.testing.check_score`,
{func}`~tscp.testing.check_weights`, {func}`~tscp.testing.check_quantile_model`,
{func}`~tscp.testing.check_online_update`, {func}`~tscp.testing.check_region` and
{func}`~tscp.testing.check_feature_map`.

## Use it

```bash
python examples/custom_component.py --config examples/configs/custom_component.yaml
```

On errors whose size depends on the hour, split conformal over-covers some hours and
under-covers others; the periodic weights keep every hour near the target. To make the
component buildable by name in specs, register it ({doc}`plugin`).

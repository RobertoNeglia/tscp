# Compose a method

A method is one component per axis, so a new method is often a new combination of
components that exist. {func}`~tscp.methods.compose` builds one from a spec of short
names, the kind of thing a YAML file holds. A spec either names a preset with its
arguments, or gives one component per axis:

```{literalinclude} ../../../examples/configs/compose.yaml
:language: yaml
```

A component is given by its short name, as a mapping with a `name` and its arguments, or
as a full config. The registered names are listed by {func}`tscp.registry.names`:

```python
from tscp import registry

registry.names("weights")  # ['exponential', 'uniform']
```

The example builds and evaluates every method of the config on the same archive, whose
error scale jumps during the test period:

```{literalinclude} ../../../examples/compose.py
:language: python
:pyobject: main
```

```bash
python examples/compose.py --config examples/configs/compose.yaml
```

Split conformal under-covers after the jump; the methods that weigh recent errors more, or
move their level after misses, recover. Any composed method can be saved, and rebuilt from
its config with {func}`~tscp.utils.from_config`.

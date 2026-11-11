# Write a plugin

A plugin is a package that adds components or methods to tscp, so that specs and configs
can name them. It registers each class under a kind and a short name with
{func}`tscp.registry.register`:

```{literalinclude} ../../../examples/plugin/src/tscp_example_plugin/__init__.py
:language: python
:start-at: "@register(\"weights\""
:end-before: "@register(\"method\""
```

and declares the module that does the registering as an entry point in the group
`tscp.plugins`:

```{literalinclude} ../../../examples/plugin/pyproject.toml
:language: toml
:start-at: "[project.entry-points"
:end-at: "example ="
```

Once the plugin is installed (`pip install -e examples/plugin`), tscp imports it when a
spec or config names something it does not know:

```{literalinclude} ../../../examples/plugin/spec.yaml
:language: yaml
```

```python
import yaml

from tscp.methods import compose, get_method

method = compose(yaml.safe_load(open("examples/plugin/spec.yaml")))
method = get_method("linear_decay", window=200)
```

Registering a name that is already taken raises an error, so a plugin never silently
replaces another plugin's components or tscp's own; `replace=True` takes a name on
purpose. Check a plugin's components with {mod}`tscp.testing` like any other.

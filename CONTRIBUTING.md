# Contributing to tscp

A conformal method in tscp is one component per axis (residual source, score, calibration
weights, quantile model, online update, region), run by `tscp.methods.ConformalPredictor`.
This guide covers setup, adding a component or a method, and style.

## Setup

```bash
pip install -e ".[dev]"
pre-commit install
```

Run the tests with `python -m pytest`, and lint with `ruff check .` and `ruff format .`.

## Adding a component

A component is one choice on one axis: a score, a weight scheme, a quantile model, an
online update, a residual source or a region type.

1. Put it in its own file in the axis subpackage, for example
   `src/tscp/weights/sliding_window.py`, subclassing that axis's base class.
2. Export it in the subpackage's `__init__.py` and `__all__`.
3. Give it only JSON-friendly constructor arguments (numbers, strings, lists, dicts or
   other components), so its config records and rebuilds.
4. Register fitted tensors as parameters or buffers. If it has a fit step, override
   `fit(archive)`: it receives only the `fit` split.
5. Add tests at the mirrored path, for example `tests/weights/test_sliding_window.py`.
   Test exact properties where the theory gives them.
6. List it in `CHANGELOG.md` under "Unreleased".

## Adding a method

A method is a preset: one component per axis.

1. Subclass `tscp.methods.ConformalPredictor` in `src/tscp/methods/<name>.py` and wire the
   components in `__init__`, exposing only the method's own hyperparameters.
2. Register it in `method_map` in `src/tscp/methods/__init__.py`. The generic tests in
   `tests/methods/test_presets.py` then run on it: causality, split discipline, config
   round trip and device.
3. Check it against the authors' code or published numbers. The docstring cites the
   paper and lists every deviation under "Deviation".

## Style

- Google-style docstrings that give tensor shapes, such as `[T, H, N]`.
- Torch tensors throughout; NumPy only for files.
- Optional dependencies go through `tscp.imports.require` and an extra in `pyproject.toml`.
- Settings such as the horizon come from configs, never from code, labels or paths.

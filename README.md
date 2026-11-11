# tscp

[![CI](https://github.com/RobertoNeglia/tscp/actions/workflows/ci.yaml/badge.svg)](https://github.com/RobertoNeglia/tscp/actions/workflows/ci.yaml)
[![Documentation](https://readthedocs.org/projects/tscp/badge/?version=latest)](https://tscp.readthedocs.io/en/latest/)

Conformal prediction for time series, in PyTorch: the published methods behind one interface,
from one-step intervals to joint regions over a forecast horizon and across series, with one
evaluation protocol.

**Status:** early development. The data contract, the component interfaces, contract
checks for new components, a registry with plugins, feature maps, helpers for learned
components and saved methods are in place, with split CP, NexCP and ACI and the absolute,
signed, normalized and CQR scores; more methods will follow.

## Install

tscp is in early development; its pre-releases install with

```bash
pip install --pre tscp
```

For development, clone the repository and install it with `pip install -e ".[dev]"`.
Documentation: https://tscp.readthedocs.io

## Quickstart

```python
from tscp.datasets import variance_shift
from tscp.methods import get_method
from tscp.metrics import evaluate

archive = variance_shift(horizon=1)  # a synthetic residual archive
method = get_method("nexcp", decay=0.99, alpha=0.1)
region = method.fit(archive).calibrate(archive).predict(archive)
print(evaluate(region, archive.split("test"), alpha=0.1, window=200))
```

`examples/quickstart.py` compares the three methods on two synthetic archives.

## Build your own method

A method is one component per axis: residual source, score, calibration weights, quantile
model, online update. `compose` builds one from short names, such as a spec read from YAML:

```python
from tscp.methods import compose

method = compose(
    {
        "score": "signed",
        "source": "expanding",
        "weights": {"name": "exponential", "decay": 0.99},
        "update": {"name": "aci", "gamma": 0.005},
    }
)
```

A new component subclasses its axis's base class, and `tscp.testing` checks that it keeps
the contracts every method relies on, such as causality and split discipline:

```python
from tscp.testing import check_method

check_method(lambda: compose({"score": "signed", "update": "aci"}))
```

Packages can register components under short names as plugins. The tutorials in `docs/`
cover composing methods, writing components, learned components, plugins and batch
production; the code they show is in `examples/` and runs in the tests.

## Batch production

```python
from tscp.data import ResidualArchive
from tscp.methods import load_method, save_method

archive = ResidualArchive.from_forecasts(
    truth, forecasts, sizes={"fit": 0.2, "calibration": 0.3, "test": None}
)
save_method(method.fit(archive), "method.npz")  # one file, no pickle
region = load_method("method.npz").calibrate(archive).predict(archive)
intervals = region.to_target(archive.split("test").forecasts)
```

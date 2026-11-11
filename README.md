# tscp

Conformal prediction for time series, in PyTorch: the published methods behind one interface,
from one-step intervals to joint regions over a forecast horizon and across series, with one
evaluation protocol.

**Status:** early development. The data contract, the component interfaces, and split CP,
NexCP and ACI are in place; more methods will follow.

## Install

```bash
pip install -e ".[dev]"
```

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

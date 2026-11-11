# tscp

Conformal prediction for time series, in PyTorch: the published methods behind one
interface, from one-step intervals to joint regions over a forecast horizon and across
series, with one evaluation protocol.

```python
from tscp.datasets import variance_shift
from tscp.methods import get_method
from tscp.metrics import evaluate

archive = variance_shift(horizon=1)
method = get_method("nexcp", decay=0.99, alpha=0.1)
region = method.fit(archive).calibrate(archive).predict(archive)
print(evaluate(region, archive.split("test"), alpha=0.1, window=200))
```

```{toctree}
:maxdepth: 2

concepts
tutorials/compose
tutorials/component
tutorials/learned
tutorials/plugin
tutorials/production
api
```

# Write a learned component

A learned component has torch parameters and fits them in its `fit(train, validation)`.
The composer calls `fit` with the fit split, and the validation split when the plan has
one; both are row subsets whose timeline ends where the split ends. {mod}`tscp.nn` holds
the pieces: {class}`~tscp.nn.RowBatches` batches the rows with their features,
{func}`~tscp.nn.train` fits a module with early stopping on the validation split, and
{func}`~tscp.nn.seeded` makes the initial weights the same on every run.

This score divides each error by a scale that a small network predicts from the context:

```{literalinclude} ../../../examples/learned_score.py
:language: python
:pyobject: ContextScaleScore
```

The network is built inside `seeded`, so two fresh scores are identical, which
{func}`~tscp.testing.check_method` checks. Its parameters live in `state_dict()`, so a
saved method keeps them.

```bash
python examples/learned_score.py --config examples/configs/learned_score.yaml
```

On {func}`~tscp.datasets.context_scale`, whose error scale follows a covariate in the
context, the learned score covers as often as the absolute score with narrower intervals.

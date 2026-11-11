import importlib.util
from pathlib import Path

import pytest
import torch

from tscp.datasets import context_scale
from tscp.features import ResidualHistory
from tscp.methods import SplitCP
from tscp.metrics import evaluate, winkler_score
from tscp.nn import RowBatches, interval_score, pinball_loss, seeded, train
from tscp.regions import Interval
from tscp.scores import AbsoluteScore
from tscp.testing import check_method, check_score, synthetic_archive

EXAMPLE = Path(__file__).parents[2] / "examples" / "learned_score.py"


def load_example():
    spec = importlib.util.spec_from_file_location("learned_score_example", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pinball_loss_is_minimized_by_the_quantile():
    targets = torch.randn(4000, dtype=torch.float64)
    grid = torch.linspace(-3, 3, 601, dtype=torch.float64)
    losses = torch.stack([pinball_loss(q, targets, 0.9) for q in grid])
    assert abs(grid[losses.argmin()].item() - torch.quantile(targets, 0.9).item()) < 0.02
    assert pinball_loss(torch.tensor(0.0), torch.tensor([1.0, -1.0]), 0.9).item() == pytest.approx(
        (0.9 * 1.0 + 0.1 * 1.0) / 2
    )
    weights = torch.tensor([1.0, 0.0])
    assert pinball_loss(torch.zeros(2), torch.tensor([1.0, -5.0]), 0.5, weights).item() == 0.5


def test_interval_score_is_the_mean_winkler_score():
    lower, upper = -torch.rand(10), torch.rand(10)
    targets = 2 * torch.randn(10)
    expected = winkler_score(Interval(lower, upper), targets, 0.1).mean()
    assert torch.allclose(interval_score(lower, upper, targets, 0.1), expected)


def test_seeded_blocks_repeat_without_touching_the_callers_random_state():
    torch.manual_seed(5)
    expected = torch.rand(3)
    torch.manual_seed(5)
    with seeded(0):
        first = torch.nn.Linear(4, 2).weight.detach().clone()
    with seeded(0):
        second = torch.nn.Linear(4, 2).weight.detach().clone()
    assert torch.equal(first, second)
    assert torch.equal(torch.rand(3), expected)  # the caller's stream went on unchanged


def fit_line(seed=0, epochs=30, patience=None, validation=True):
    """Fit y = 2x + noise with a linear model; validation targets flip sign to force overfit."""
    generator = torch.Generator().manual_seed(1)
    x = torch.randn(64, 1, generator=generator)
    y = 2 * x + 0.1 * torch.randn(64, 1, generator=generator)
    batches = [(x[i : i + 16], y[i : i + 16]) for i in range(0, 64, 16)]
    held_out = [(x, -y)] if validation else None
    with seeded(seed):
        model = torch.nn.Linear(1, 1)

    def loss(module, batch):
        inputs, targets = batch
        return (module(inputs) - targets).square().mean()

    optimizer = lambda params: torch.optim.SGD(params, lr=0.1)  # noqa: E731
    result = train(
        model,
        loss,
        batches,
        validation=held_out,
        epochs=epochs,
        patience=patience,
        optimizer=optimizer,
        seed=seed,
    )
    return model, result


def test_training_is_reproducible():
    first, first_result = fit_line(validation=False)
    second, second_result = fit_line(validation=False)
    assert first_result.losses == second_result.losses
    assert torch.equal(first.weight, second.weight)
    assert first_result.losses[-1] < first_result.losses[0] and first_result.best_epoch is None


def test_early_stopping_keeps_the_best_validation_state():
    model, result = fit_line(epochs=30, patience=3)
    best = result.best_epoch
    assert len(result.validation_losses) == best + 1 + 3  # stopped 3 epochs after the best
    assert result.validation_losses[best] == min(result.validation_losses)
    # The kept state is the best epoch's, not the last one's.
    rerun, _ = fit_line(epochs=best + 1, validation=False)
    assert torch.allclose(model.weight, rerun.weight) and torch.allclose(model.bias, rerun.bias)


def test_training_checks_its_inputs():
    model = torch.nn.Linear(1, 1)
    with pytest.raises(ValueError, match="epochs"):
        train(model, lambda m, b: m.weight.sum(), [None], epochs=0)
    with pytest.raises(ValueError, match="empty"):
        train(model, lambda m, b: m.weight.sum(), [], epochs=1)
    with pytest.raises(ValueError, match="no parameters"):
        train(torch.nn.ReLU(), lambda m, b: torch.tensor(0.0), [None])


def test_row_batches_cover_every_row_once_per_pass():
    archive = synthetic_archive(horizon=2, num_origins=200, missing=0.1)
    rows = archive.split("calibration")
    batches = RowBatches(rows, ResidualHistory(2), batch_size=16, shuffle=True, seed=3)
    passes = [torch.cat([batch["index"] for batch in batches]) for _ in range(2)]
    for order in passes:
        assert torch.equal(order.sort().values, torch.arange(len(rows)))
    assert not torch.equal(passes[0], passes[1])  # a new order on every pass
    again = RowBatches(rows, ResidualHistory(2), batch_size=16, shuffle=True, seed=3)
    assert torch.equal(torch.cat([batch["index"] for batch in again]), passes[0])
    batch = next(iter(RowBatches(rows, ResidualHistory(2), batch_size=8)))
    assert batch["features"].shape == (8, 2, 1, 2) and batch["features_valid"].shape == (8, 2, 1)
    assert torch.equal(batch["observed"], rows.observed[:8])
    assert len(batches) == -(-len(rows) // 16)


def test_the_learned_score_example_keeps_the_contracts():
    ContextScaleScore = load_example().ContextScaleScore

    def factory():
        return ContextScaleScore(inputs=2, hidden=8, epochs=15, patience=5)

    check_score(factory)
    check_method(lambda: SplitCP(score=factory()))


def test_the_learned_score_narrows_intervals_where_the_context_explains_the_scale():
    ContextScaleScore = load_example().ContextScaleScore
    sizes = {"fit": 0.3, "validation": 0.1, "calibration": 0.2, "test": None}
    archive = context_scale(horizon=1, num_origins=3000, sizes=sizes, seed=4)
    test = archive.split("test")
    reports = {}
    for name, score in (
        ("absolute", AbsoluteScore()),
        ("learned", ContextScaleScore(inputs=1, epochs=100, patience=10)),
    ):
        method = SplitCP(score=score)
        reports[name] = evaluate(method.fit(archive).calibrate(archive).predict(archive), test, 0.1)
    assert abs(reports["learned"].coverage - 0.9) < 0.03
    assert reports["learned"].log_volume_per_dim < reports["absolute"].log_volume_per_dim - 0.3

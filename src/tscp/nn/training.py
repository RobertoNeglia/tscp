"""A small, reproducible training loop for learned components."""

import contextlib
import math
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@contextlib.contextmanager
def seeded(seed: int, device: torch.device | str | None = None) -> Iterator[None]:
    """Run a block with the random state seeded, and restore the caller's afterwards.

    Building a network inside ``seeded(seed)`` makes its initial weights the same on every
    run, without changing the random numbers the rest of the program draws.

    Args:
        seed: The seed.
        device: A CUDA device whose generator to fork and seed as well.
    """
    devices = [device] if device is not None and torch.device(device).type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        yield


@dataclass(frozen=True)
class TrainingResult:
    """What :func:`train` did.

    Attributes:
        losses: Mean training loss of each epoch.
        validation_losses: Validation loss after each epoch; empty without validation.
        best_epoch: The epoch whose state the module was left in, with validation.
    """

    losses: tuple[float, ...]
    validation_losses: tuple[float, ...]
    best_epoch: int | None


def train(
    module: torch.nn.Module,
    loss: Callable[[torch.nn.Module, Any], Tensor],
    batches: Iterable[Any],
    *,
    validation: Iterable[Any] | None = None,
    epochs: int = 100,
    optimizer: Callable[[list[torch.nn.Parameter]], torch.optim.Optimizer] | None = None,
    patience: int | None = None,
    seed: int = 0,
) -> TrainingResult:
    """Fit ``module`` by minimizing ``loss(module, batch)`` over the batches.

    Training is reproducible without touching the caller's random state: it runs inside
    :func:`seeded`, so dropout and shuffles draw the same numbers on every run.

    With ``validation``, the mean loss over its batches is computed after every epoch, in
    eval mode and without gradients. The module ends in the state of the epoch with the
    lowest validation loss, and with ``patience``, training stops after that many epochs
    without improvement.

    Args:
        module: The module to train; its parameters with ``requires_grad`` are updated.
        loss: Maps the module and a batch to a scalar loss.
        batches: Training batches; iterated once per epoch, so it must be re-iterable,
            like :class:`tscp.nn.RowBatches`.
        validation: Validation batches, also re-iterable.
        epochs: Most passes over the training batches.
        optimizer: Builds the optimizer from the parameters; Adam with learning rate
            ``1e-3`` by default.
        patience: Epochs without a better validation loss before stopping.
        seed: Seed of the random state during training.
    """
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if patience is not None and patience < 1:
        raise ValueError("patience must be positive")
    parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("the module has no parameters to train")
    build = optimizer or (lambda params: torch.optim.Adam(params, lr=1e-3))
    losses: list[float] = []
    validation_losses: list[float] = []
    best, best_epoch, best_state, waited = math.inf, None, None, 0
    with seeded(seed, parameters[0].device):
        step = build(parameters)
        for epoch in range(epochs):
            module.train()
            total, count = 0.0, 0
            for batch in batches:
                step.zero_grad()
                value = loss(module, batch)
                value.backward()
                step.step()
                total, count = total + float(value.detach()), count + 1
            if count == 0:
                raise ValueError("the training batches are empty")
            losses.append(total / count)
            if validation is None:
                continue
            module.eval()
            with torch.no_grad():
                values = [float(loss(module, batch)) for batch in validation]
            if not values:
                raise ValueError("the validation batches are empty")
            current = sum(values) / len(values)
            validation_losses.append(current)
            if current < best:
                best, best_epoch, waited = current, epoch, 0
                best_state = {k: v.detach().clone() for k, v in module.state_dict().items()}
            else:
                waited += 1
                if patience is not None and waited >= patience:
                    break
    if best_state is not None:
        module.load_state_dict(best_state)
    module.eval()
    return TrainingResult(tuple(losses), tuple(validation_losses), best_epoch)


__all__ = ["TrainingResult", "seeded", "train"]

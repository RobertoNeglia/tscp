"""A learned score: the scale of each error, predicted from the context by a small network.

    python examples/learned_score.py --config examples/configs/learned_score.yaml

The network is fitted on the fit split with tscp.nn, stopping early on the validation
split, and split conformal calibrates it on the calibration split. On an archive whose
error scale follows a covariate, it covers as often as the absolute score with narrower
intervals. Nothing is written to disk.
"""

import argparse

import torch
from torch import Tensor

from tscp.data import ResidualArchive
from tscp.datasets import context_scale
from tscp.features import ContextColumns
from tscp.methods import SplitCP
from tscp.metrics import evaluate
from tscp.nn import RowBatches, seeded, train
from tscp.regions import Interval
from tscp.scores import AbsoluteScore, Score


class ContextScaleScore(Score):
    """``|r| / sigma(x)``, with ``sigma`` a small network on the context ``x``.

    The network maps a row's context to the log of its error scale, shared by every step
    and series. Its fit minimizes the Gaussian negative log-likelihood of the observed
    residuals of the fit split, with early stopping on the validation split.

    Args:
        inputs: Number of context columns.
        hidden: Width of the hidden layer.
        epochs: Most training epochs.
        patience: Epochs without a better validation loss before stopping.
        seed: Seed of the initial weights and of training.
    """

    def __init__(
        self, inputs: int, hidden: int = 16, epochs: int = 200, patience: int = 20, seed: int = 0
    ):
        super().__init__()
        self.features = ContextColumns()
        self.epochs, self.patience, self.seed = int(epochs), int(patience), int(seed)
        with seeded(seed):  # the same initial weights on every run
            self.network = torch.nn.Sequential(
                torch.nn.Linear(inputs, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, 1)
            )

    def fit(self, train_rows: ResidualArchive, validation: ResidualArchive | None = None):
        def loss(network, batch):
            log_scale = network(batch["features"])[:, :, None]  # [B, 1, 1]
            nll = log_scale + 0.5 * (batch["residuals"] / log_scale.exp()).square()
            observed = batch["observed"].to(nll.dtype)
            return (nll * observed).sum() / observed.sum().clamp(min=1)

        batches = RowBatches(
            train_rows, self.features, batch_size=128, shuffle=True, seed=self.seed
        )
        held_out = None
        if validation is not None:
            held_out = RowBatches(validation, self.features, batch_size=4096)
        train(
            self.network,
            loss,
            batches,
            validation=held_out,
            epochs=self.epochs,
            patience=self.patience if held_out is not None else None,
            seed=self.seed,
        )
        return self

    def _scale(self, rows: ResidualArchive) -> Tensor:
        with torch.no_grad():
            return self.network(self.features(rows)).exp()[:, :, None]  # [T, 1, 1]

    def forward(self, rows: ResidualArchive) -> Tensor:
        return rows.residuals.abs() / self._scale(rows)

    def region(self, threshold: Tensor, rows: ResidualArchive) -> Interval:
        width = threshold * self._scale(rows)
        return Interval(-width, width)


def main() -> None:
    import yaml

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="YAML file with the settings")
    args = parser.parse_args()
    with open(args.config) as handle:
        config = yaml.safe_load(handle)

    archive = context_scale(
        horizon=config["horizon"],
        num_origins=config["num_origins"],
        strength=config["strength"],
        sizes=config["sizes"],
        seed=config["seed"],
    )
    test = archive.split("test")
    alpha = config["alpha"]
    scores = {
        "absolute": AbsoluteScore(),
        "learned": ContextScaleScore(inputs=archive.context.shape[-1], **config["network"]),
    }
    print(f"{'score':9} {'coverage':>8} {'log-width':>9}")
    for name, score in scores.items():
        method = SplitCP(score=score, alpha=alpha)
        report = evaluate(method.fit(archive).calibrate(archive).predict(archive), test, alpha)
        print(f"{name:9} {report.coverage:8.3f} {report.log_volume_per_dim:9.3f}")


if __name__ == "__main__":
    main()

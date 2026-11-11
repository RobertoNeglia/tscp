"""Batch production: from a forecaster's backtest to intervals in target units.

    python examples/production.py --config examples/configs/production.yaml

A forecaster's backtest gives forecasts and the truth as plain arrays. They become a
residual archive; a method composed from the config is fitted, saved and loaded again,
as a later job would; the loaded method is recalibrated, predicts, and its regions are
mapped back to target units. The saved file goes to a temporary directory.
"""

import argparse
import math
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml

from tscp.data import ResidualArchive
from tscp.methods import compose, load_method, save_method
from tscp.metrics import evaluate


def seasonal_backtest(num_hours: int, horizon: int, seed: int):
    """A daily cycle with noise that grows in the afternoon, and a seasonal naive forecaster.

    The forecast of the value at hour ``s`` is the value at ``s - 24``, so it is known at
    every origin up to 24 hours ahead.
    """
    generator = torch.Generator().manual_seed(seed)
    hours = torch.arange(num_hours + horizon + 24)
    noise = (0.5 + (hours % 24 >= 12).double()) * torch.randn(len(hours), generator=generator)
    values = 10 + 3 * torch.sin(2 * math.pi * hours / 24) + noise
    origins = torch.arange(24, 24 + num_hours)
    steps = torch.arange(1, horizon + 1)
    targets = origins[:, None] + steps[None, :]
    return origins, steps, values[targets], values[targets - 24]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="YAML file with the settings")
    args = parser.parse_args()
    with open(args.config) as handle:
        config = yaml.safe_load(handle)

    origins, steps, truth, forecasts = seasonal_backtest(
        config["num_hours"], config["horizon"], config["seed"]
    )
    start = np.datetime64(config["start"], "h")
    archive = ResidualArchive.from_forecasts(
        truth,
        forecasts,
        origins=origins,
        steps=steps,
        sizes=config["sizes"],
        timestamps=start + origins.numpy().astype("timedelta64[h]"),
        provenance={"forecaster": "seasonal naive", "horizon": config["horizon"]},
    )

    method = compose(config["method"]).fit(archive)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "method.npz"
        save_method(method, path)
        method = load_method(path, archive)  # as a later job would

    region = method.calibrate(archive).predict(archive)
    test = archive.split("test")
    report = evaluate(region, test, method.alpha)
    steps_text = " ".join(f"{c:.3f}" for c in report.coverage_by_step)
    print(
        f"coverage {report.coverage:.3f} (by step: {steps_text}), joint {report.joint_coverage:.3f}"
    )

    intervals = region.to_target(test.forecasts)  # in the target's units
    print(f"{'origin':16} {'step':>4} {'forecast':>8} {'lower':>7} {'upper':>7} {'truth':>7}")
    for row in range(len(test) - 3, len(test)):
        when = np.datetime64(int(test.timestamps[row]), "ns").astype("datetime64[h]")
        for step in range(archive.horizon):
            print(
                f"{str(when):16} {int(archive.steps[step]):>4} "
                f"{float(test.forecasts[row, step, 0]):8.2f} "
                f"{float(intervals.lower[row, step, 0]):7.2f} "
                f"{float(intervals.upper[row, step, 0]):7.2f} "
                f"{float(test.targets()[row, step, 0]):7.2f}"
            )


if __name__ == "__main__":
    main()

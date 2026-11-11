"""Build an archive from cached ensemble predictions and compare residual intervals.

    python examples/bootstrap_intervals.py --config examples/configs/enbpi.yaml

The synthetic member forecasts illustrate the array contract. Replace them and the
membership array with exports from fitted bootstrap forecasters for an application.
The SPCI configuration additionally needs the ``forest`` extra. Nothing is written.
"""

import argparse

import torch
import yaml

from tscp.data import from_bootstrap_predictions
from tscp.datasets import variance_shift
from tscp.methods import compose
from tscp.metrics import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="YAML file with the settings")
    args = parser.parse_args()
    with open(args.config) as handle:
        config = yaml.safe_load(handle)
    archive = variance_shift(
        horizon=config["horizon"], num_origins=config["num_origins"], seed=config["seed"]
    )
    generator = torch.Generator().manual_seed(config["seed"])
    count, members = len(archive.splits["calibration"]), config["members"]
    in_bag = torch.rand(members, count, generator=generator) < config["in_bag_fraction"]
    in_bag[torch.arange(count) % members, torch.arange(count)] = False
    offsets = config["member_spread"] * torch.randn(members, generator=generator)
    member_forecasts = archive.forecasts[..., None] + offsets
    archive = from_bootstrap_predictions(
        archive.targets(),
        member_forecasts,
        in_bag,
        origins=archive.origins,
        steps=archive.steps,
        splits=archive.splits,
        provenance={"dataset": "variance_shift", "forecaster": "synthetic_cached_ensemble"},
    )
    print(f"{'method':14} {'coverage':>8} {'log-width':>9}")
    for name, spec in config["methods"].items():
        method = compose({**spec, "alpha": config["alpha"]})
        region = method.fit(archive).calibrate(archive).predict(archive)
        report = evaluate(region, archive.split("test"), method.alpha)
        print(f"{name:14} {report.coverage:8.3f} {report.log_volume_per_dim:9.3f}")


if __name__ == "__main__":
    main()

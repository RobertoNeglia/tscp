"""Run split CP, NexCP and ACI on two synthetic archives and print their reports.

    python examples/quickstart.py --config examples/configs/quickstart.yaml

Nothing is written to disk. The Gaussian control is exchangeable, so every method should
cover about 1 - alpha; on the variance shift, split CP under-covers after the jump while
NexCP and ACI recover.
"""

import argparse

import yaml

from tscp.datasets import synthetic
from tscp.methods import get_method
from tscp.metrics import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="YAML file with the settings")
    args = parser.parse_args()
    with open(args.config) as handle:
        config = yaml.safe_load(handle)

    alpha = config["alpha"]
    print(f"{'dataset':18} {'H':>2} {'method':6} {'coverage':>8} {'worst':>6} "
          f"{'log-width':>9} {'unbounded':>9}")  # fmt: skip
    for dataset in config["datasets"]:
        archive = getattr(synthetic, dataset)(
            horizon=config["horizon"],
            num_origins=config["num_origins"],
            sizes=config["sizes"],
            seed=config["seed"],
        )
        test = archive.split("test")
        for name, arguments in config["methods"].items():
            method = get_method(name, alpha=alpha, **arguments)
            region = method.fit(archive).calibrate(archive).predict(archive)
            report = evaluate(region, test, alpha, window=config["window"])
            print(
                f"{dataset:18} {archive.horizon:>2} {name:6} {report['coverage']:8.3f} "
                f"{report['worst_window_coverage']:6.3f} {report['log_volume_per_dim']:9.3f} "
                f"{report['infinite_fraction']:9.3f}"
            )


if __name__ == "__main__":
    main()

"""Compose methods from specs: the same building blocks, combined in different ways.

    python examples/compose.py --config examples/configs/compose.yaml

Each entry under ``methods`` in the config is a spec of short names, which
``tscp.methods.compose`` builds; the script evaluates every method on the same archive.
Nothing is written to disk.
"""

import argparse

import yaml

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
    test = archive.split("test")
    print(f"{'method':14} {'coverage':>8} {'worst':>6} {'log-width':>9}")
    for name, spec in config["methods"].items():
        method = compose({**spec, "alpha": config["alpha"]})
        region = method.fit(archive).calibrate(archive).predict(archive)
        report = evaluate(region, test, method.alpha, window=config["window"])
        print(
            f"{name:14} {report.coverage:8.3f} {report.worst_window_coverage:6.3f} "
            f"{report.log_volume_per_dim:9.3f}"
        )


if __name__ == "__main__":
    main()

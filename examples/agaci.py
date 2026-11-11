"""Aggregate adaptive interval endpoints on a synthetic variance shift.

python examples/agaci.py --config examples/configs/agaci.yaml
"""

import argparse

import yaml

from tscp.datasets import variance_shift
from tscp.methods.agaci import AgACI
from tscp.metrics import evaluate
from tscp.sources import ExpandingSource


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="YAML file with the settings")
    args = parser.parse_args()
    with open(args.config) as handle:
        config = yaml.safe_load(handle)
    archive = variance_shift(
        horizon=config["horizon"], num_origins=config["num_origins"], seed=config["seed"]
    )
    method = AgACI(
        endpoint_bounds=config["endpoint_bounds"],
        alpha=config["alpha"],
        source=ExpandingSource(window_size=config["calibration_window"]),
    )
    region = method.fit(archive).calibrate(archive).predict(archive)
    print(evaluate(region, archive.split("test"), method.alpha, window=config["report_window"]))


if __name__ == "__main__":
    main()

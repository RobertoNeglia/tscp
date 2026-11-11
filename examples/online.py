"""Compare online controllers on a synthetic residual archive from YAML settings.

Run with ``--config examples/configs/online.yaml``. The optional
``online_pid.yaml`` configuration also exercises Theta scorecasting and requires
the ``pid`` extra. No files are written.
"""

import argparse
import json

import yaml

from tscp.datasets import synthetic
from tscp.methods import get_method
from tscp.metrics import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as handle:
        config = yaml.safe_load(handle)
    archive = getattr(synthetic, config["dataset"])(
        horizon=config["horizon"],
        num_origins=config["num_origins"],
        sizes=config["sizes"],
        seed=config["seed"],
    )
    for name, settings in config["methods"].items():
        method = get_method(name, alpha=config["alpha"], **settings)
        method.fit(archive)
        regions = method.calibrate(archive).predict(archive)
        report = evaluate(regions, archive.split("test"), config["alpha"], window=config["window"])
        print(json.dumps({"method": name, "horizon": archive.horizon, "report": report.to_dict()}))


if __name__ == "__main__":
    main()

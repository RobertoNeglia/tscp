"""Compare pooled, rolling and category-specific residual calibration."""

import argparse

import yaml

from tscp.datasets import variance_shift
from tscp.methods import compose
from tscp.metrics import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as handle:
        settings = yaml.safe_load(handle)
    archive = variance_shift(
        horizon=settings["horizon"], num_origins=settings["num_origins"], seed=settings["seed"]
    )
    for name, spec in settings["methods"].items():
        method = compose({**spec, "alpha": settings["alpha"]})
        region = method.fit(archive).calibrate(archive).predict(archive)
        report = evaluate(region, archive.split("test"), method.alpha)
        print(name, report.to_dict())


if __name__ == "__main__":
    main()

"""Write a component, check it with tscp.testing, and use it in a method.

    python examples/custom_component.py --config examples/configs/custom_component.yaml

``PeriodicWeights`` counts only the past errors at the same point of a cycle, such as the
same hour of the day, and lets older cycles count less. On errors whose size depends on
the hour, it keeps the coverage of every hour near the target, where split conformal
over-covers some hours and under-covers others. Nothing is written to disk.
"""

import argparse

import torch
import yaml
from torch import Tensor

from tscp.data import ResidualArchive
from tscp.methods import ConformalPredictor, SplitCP
from tscp.metrics import evaluate
from tscp.scores import AbsoluteScore
from tscp.sources import ExpandingSource
from tscp.testing import check_method, check_weights
from tscp.weights import Weights


class PeriodicWeights(Weights):
    """Weights for past residuals in the same phase of a cycle, decaying cycle by cycle.

    A pool residual whose origin lies ``k`` whole cycles before the query's gets weight
    ``decay ** k``; residuals at other points of the cycle get 0.

    Args:
        period: Length of the cycle, in base time steps.
        decay: Weight ratio per cycle, in (0, 1].
    """

    def __init__(self, period: int = 24, decay: float = 0.9):
        super().__init__()
        if period < 1 or not 0 < decay <= 1:
            raise ValueError("period must be positive and decay must lie in (0, 1]")
        self.period = int(period)
        self.decay = float(decay)

    def forward(self, pool: ResidualArchive, query: ResidualArchive) -> Tensor:
        age = (query.origins[:, None] - pool.origins[None, :]).clamp(min=0)
        same_phase = age % self.period == 0
        weights = torch.tensor(self.decay, dtype=torch.float64) ** (age // self.period)
        return torch.where(same_phase, weights, torch.zeros((), dtype=torch.float64))


def hourly_errors(num_origins: int, seed: int) -> ResidualArchive:
    """Errors twice as large in the second half of every day, as plain arrays."""
    generator = torch.Generator().manual_seed(seed)
    hours = torch.arange(num_origins)
    scale = 0.5 + (hours % 24 >= 12).float()
    errors = scale * torch.randn(num_origins, generator=generator)
    sizes = {"calibration": 0.4, "test": None}
    return ResidualArchive.from_forecasts(errors, torch.zeros(num_origins), sizes=sizes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="YAML file with the settings")
    args = parser.parse_args()
    with open(args.config) as handle:
        config = yaml.safe_load(handle)
    period, decay, alpha = config["period"], config["decay"], config["alpha"]

    def periodic_method():
        weights = PeriodicWeights(period, decay)
        return ConformalPredictor(
            AbsoluteScore(), source=ExpandingSource(), weights=weights, alpha=alpha
        )

    # The checks raise an AssertionError naming the first contract a component breaks.
    check_weights(lambda: PeriodicWeights(period, decay))
    check_method(periodic_method)
    print("PeriodicWeights keeps the weights and method contracts")

    archive = hourly_errors(config["num_origins"], config["seed"])
    test = archive.split("test")
    hours = test.origins % period
    for name, method in (("split", SplitCP(alpha=alpha)), ("periodic", periodic_method())):
        region = method.calibrate(archive).predict(archive)
        report = evaluate(region, test, alpha, groups=hours)
        by_hour = report.grouped_coverage.values()
        print(
            f"{name:9} coverage {report.coverage:.3f}, "
            f"by hour from {min(by_hour):.3f} to {max(by_hour):.3f}"
        )


if __name__ == "__main__":
    main()

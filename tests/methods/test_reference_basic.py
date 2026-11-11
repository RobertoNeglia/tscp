"""Fixed scalar references evaluated independently of tscp's implementation."""

import json
from pathlib import Path

import pytest
import torch

from tscp.data import ResidualArchive, SplitPlan
from tscp.methods import ACI, NexCP, SplitCP
from tscp.scores import CQRScore, NormalizedScore, SignedScore

FIXTURE = json.loads((Path(__file__).parents[1] / "fixtures/references/basic.json").read_text())


@pytest.mark.parametrize("name", sorted(FIXTURE["expected"]))
def test_against_scalar_published_equations(name):
    inputs, parameters = FIXTURE["inputs"], FIXTURE["parameters"]
    residuals = torch.tensor(inputs["calibration"] + inputs["test"], dtype=torch.float64)
    forecasts = torch.arange(len(residuals), dtype=torch.float64)[:, None, None] + 10
    archive = ResidualArchive(
        residuals=residuals[:, None, None],
        origins=torch.tensor(inputs["origins"]),
        steps=torch.tensor([1]),
        forecasts=forecasts,
        outputs={
            "scale": torch.tensor(inputs["scales"], dtype=torch.float64)[:, None, None],
            "lower": forecasts + torch.tensor(inputs["lower"], dtype=torch.float64)[:, None, None],
            "upper": forecasts + torch.tensor(inputs["upper"], dtype=torch.float64)[:, None, None],
        },
        splits=SplitPlan({"calibration": torch.arange(9), "test": torch.arange(9, 14)}),
    )
    alpha = parameters["alpha"]
    scores = {
        "signed": lambda: SignedScore(),
        "normalized": lambda: NormalizedScore("scale"),
        "normalized_fitted": NormalizedScore,
        "cqr": CQRScore,
        "cqr_asymmetric": lambda: CQRScore(asymmetric=True),
    }
    if name == "nexcp":
        method = NexCP(decay=parameters["decay"], alpha=alpha)
    elif name == "aci":
        method = ACI(gamma=parameters["gamma"], alpha=alpha)
    else:
        method = SplitCP(score=scores[name]() if name in scores else None, alpha=alpha)
    if name == "normalized_fitted":
        fitting = ResidualArchive(
            residuals=residuals[:9, None, None],
            origins=torch.arange(-20, -11),
            steps=torch.tensor([1]),
            splits=SplitPlan({"fit": torch.arange(9)}),
        )
        method.score.fit(fitting.split("fit"))
    region = method.calibrate(archive).predict(archive)
    expected = torch.tensor(
        [[float(v) for v in pair] for pair in FIXTURE["expected"][name]], dtype=torch.float64
    )
    actual = torch.stack((region.lower.flatten(), region.upper.flatten()), dim=-1)
    torch.testing.assert_close(actual, expected, **FIXTURE["provenance"]["tolerance"])

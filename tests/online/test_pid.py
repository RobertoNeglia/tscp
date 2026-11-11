import json
import math
from pathlib import Path

import pytest
import torch

from tscp.online.pid import PIDUpdate
from tscp.online.scorecaster import Scorecaster
from tscp.testing import check_online_update

FIXTURE = Path(__file__).parents[1] / "fixtures" / "online_reference.json"


class FixedScorecaster(Scorecaster):
    def __init__(self, values):
        super().__init__()
        self.values = values
        self.register_buffer("target", None, persistent=False)

    def reset(self, alpha, device):
        self.target = alpha.to(device)

    def predict(self, t, *, origin, target_times):
        return torch.full_like(self.target, self.values[t])

    def observe(self, t, scores, released):
        pass


def test_controller_matches_pinned_authors_code_with_and_without_scorecasts():
    data = json.loads(FIXTURE.read_text())["pid"]
    assert data["provenance"]["revision"] == "b729c3f5ff633bfc43f0f7ca08199b549c2573ac"
    for case in data["cases"]:
        args = case["settings"]
        caster = FixedScorecaster(data["scorecasts"]) if case["scorecast"] else None
        update = PIDUpdate(
            lr=args["lr"],
            csat=args["Csat"],
            ki=args["KI"],
            proportional_lr=case["proportional_lr"],
            lr_window=args["T_burnin"],
            scorecaster=caster,
        )
        update.reset(torch.tensor(args["alpha"], dtype=torch.float64), torch.device("cpu"))
        for t, score in enumerate(data["scores"]):
            update.prepare_query(t, origin=t, target_times=torch.tensor(t + 1))
            threshold = update.threshold(t, None)
            assert float(threshold) == pytest.approx(case["thresholds"][t], abs=1e-12)
            update.observe(
                t, torch.tensor(score, dtype=torch.float64), threshold, torch.tensor(True)
            )


def test_zero_integral_is_exact_quantile_tracking_and_masked():
    update = PIDUpdate(lr=0.5, ki=0, proportional_lr=False)
    update.reset(torch.tensor([0.1, 0.2], dtype=torch.float64), torch.device("cpu"))
    assert torch.equal(update.threshold(0, None), torch.zeros(2, dtype=torch.float64))
    update.observe(0, torch.tensor([1.0, 1.0]), torch.zeros(2), torch.tensor([True, False]))
    torch.testing.assert_close(update.threshold(1, None), torch.tensor([0.45, 0.0]).double())
    assert torch.equal(update.count, torch.tensor([1.0, 0.0]).double())


@pytest.mark.parametrize("sign", [-1, 1])
def test_tangent_saturation_preserves_unbounded_thresholds(sign):
    update = PIDUpdate(lr=0, ki=1, csat=0.01, proportional_lr=False)
    update.reset(torch.tensor(0.5), torch.device("cpu"))
    update.count.fill_(10)
    update.error_sum.fill_(sign * 100)
    update.observe(0, torch.tensor(0.0), torch.tensor(0.0), torch.tensor(True))
    assert float(update.threshold(1, None)) == sign * math.inf


def test_missing_release_does_not_change_controller_history():
    update = PIDUpdate(proportional_lr=True)
    update.reset(torch.tensor([0.1]), torch.device("cpu"))
    update.observe(0, torch.tensor([1e9]), torch.tensor([0.0]), torch.tensor([False]))
    assert update._history == [[]]
    assert update.count.item() == 0


def test_pi_keeps_the_update_contract():
    check_online_update(lambda: PIDUpdate(lr=0.1, proportional_lr=False))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lr": -1},
        {"ki": -1},
        {"csat": 0},
        {"lr_window": 0},
        {"lr": float("nan")},
    ],
)
def test_bad_settings_raise(kwargs):
    with pytest.raises(ValueError):
        PIDUpdate(**kwargs)

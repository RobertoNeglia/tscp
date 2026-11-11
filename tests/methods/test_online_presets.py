import pytest
import torch

from tscp.methods.dtaci import DtACI
from tscp.methods.pid import ConformalPI, ConformalPID
from tscp.online.scorecaster import Scorecaster
from tscp.scores import AbsoluteScore, CQRScore, SignedScore
from tscp.testing import check_method


class LastReleasedScorecaster(Scorecaster):
    """A deterministic scorecaster that exercises the controller's history interface."""

    def __init__(self):
        super().__init__()
        self.register_buffer("last", None, persistent=False)

    def reset(self, alpha, device):
        self.last = torch.zeros_like(alpha, device=device)

    def initialize(self, scores, origins, target_times, valid):
        for i, row in enumerate(scores):
            mask = torch.ones_like(row, dtype=torch.bool) if valid is None else valid[i]
            self.last = torch.where(mask, row.to(self.last), self.last)

    def predict(self, t, *, origin, target_times):
        return self.last.clone()

    def observe(self, t, scores, released):
        self.last = torch.where(released, scores.to(self.last), self.last)


@pytest.mark.parametrize("score", [AbsoluteScore, SignedScore, CQRScore])
@pytest.mark.parametrize("method", ["dtaci", "pi", "pid"])
def test_presets_keep_contracts_with_channel_scores(method, score):
    def factory():
        if method == "dtaci":
            return DtACI(score=score(), gammas=[0.01, 0.05])
        if method == "pi":
            return ConformalPI(score=score(), proportional_lr=False)
        return ConformalPID(
            score=score(), proportional_lr=False, scorecaster=LastReleasedScorecaster()
        )

    check_method(factory)


def test_calibration_initializes_only_the_scorecaster(make_archive):
    archive = make_archive([1, 2, 3, 4, 5, 6, 7, 8], {"calibration": 4, "test": None})
    method = ConformalPID(scorecaster=LastReleasedScorecaster(), proportional_lr=False)
    region = method.calibrate(archive).predict(archive)
    assert region.upper[0].item() == 4
    again = method.predict(archive)
    assert torch.equal(again.upper, region.upper)

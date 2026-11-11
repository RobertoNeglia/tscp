from types import SimpleNamespace

import pytest
import torch

from tscp.imports import has_module
from tscp.online.scorecaster import ThetaScorecaster
from tscp.utils import from_config


def test_refit_schedule_missing_history_and_forecast_leads(monkeypatch):
    fitted = []
    leads = []

    class FakeTheta:
        def __init__(self, values, **kwargs):
            self.values = [float(value) for value in values]

        def fit(self):
            fitted.append(self.values)
            return self

        def forecast(self, steps):
            leads.append(steps)
            return SimpleNamespace(iloc=[self.values[-1] + steps])

    monkeypatch.setattr(
        "tscp.online.scorecaster.require", lambda *args: SimpleNamespace(ThetaModel=FakeTheta)
    )
    caster = ThetaScorecaster(min_history=3, refit_every=2)
    caster.reset(torch.tensor([0.1, 0.1]), torch.device("cpu"))
    scores = torch.tensor([[10, 100], [11, 101], [12, 102], [13, 103]], dtype=torch.float64)
    valid = torch.ones_like(scores, dtype=torch.bool)
    valid[2, 1] = False
    caster.initialize(scores, torch.arange(4), torch.arange(1, 5)[:, None], valid)
    prediction = caster.predict(0, origin=4, target_times=torch.tensor([5, 5]))
    assert prediction.tolist() == [14, 0]
    assert fitted == [[10, 11, 12, 13]]
    caster.observe(0, torch.tensor([14, 999]), torch.tensor([True, False]))
    prediction = caster.predict(1, origin=5, target_times=torch.tensor([6, 6]))
    assert prediction.tolist() == [15, 0]
    assert len(fitted) == 1
    assert leads == [1, 2]
    caster.observe(1, torch.tensor([15, 999]), torch.tensor([True, False]))
    assert caster.predict(2, origin=6, target_times=torch.tensor([7, 7])).tolist() == [16, 0]
    assert len(fitted) == 2
    assert caster._values[1] == [100, 101, 103]


def test_reset_and_config_remove_replay_history():
    caster = ThetaScorecaster(min_history=3)
    caster.reset(torch.tensor([0.1]), torch.device("cpu"))
    caster.initialize(torch.ones(4, 1), torch.arange(4), torch.arange(1, 5)[:, None], None)
    rebuilt = from_config(caster.to_config())
    assert rebuilt.to_config() == caster.to_config()
    assert not caster.state_dict()
    caster.reset(torch.tensor([0.1]), torch.device("cpu"))
    assert caster._values == [[]]
    assert caster.predict(0, origin=5, target_times=torch.tensor([6])).item() == 0


def test_completion_discards_metadata_for_missing_outcomes():
    caster = ThetaScorecaster()
    caster.reset(torch.tensor([0.1]), torch.device("cpu"))
    caster.predict(0, origin=0, target_times=torch.tensor([1]))
    caster.observe(0, torch.tensor([0.0]), torch.tensor([False]))
    assert 0 in caster._pending
    caster.finish_query(0)
    assert not caster._pending
    assert caster._values == [[]]


@pytest.mark.skipif(not has_module("statsmodels"), reason="requires the optional pid extra")
def test_theta_matches_direct_backend_forecast():
    from statsmodels.tsa.forecasting.theta import ThetaModel

    scores = (torch.linspace(0.0, 2.0, 30, dtype=torch.float64).sin() + 2)[:, None]
    caster = ThetaScorecaster(min_history=20)
    caster.reset(torch.tensor([0.1]), torch.device("cpu"))
    caster.initialize(scores, torch.arange(30), torch.arange(1, 31)[:, None], None)
    actual = caster.predict(0, origin=30, target_times=torch.tensor([31]))
    expected = ThetaModel(scores[:, 0].numpy(), period=1, deseasonalize=False).fit().forecast(1)
    assert actual.item() == pytest.approx(float(expected.iloc[-1]), abs=1e-12)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"period": 0},
        {"min_history": 1},
        {"window": 2},
        {"refit_every": 0},
    ],
)
def test_bad_settings_raise(kwargs):
    with pytest.raises(ValueError):
        ThetaScorecaster(**kwargs)

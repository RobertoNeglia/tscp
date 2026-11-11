"""The online engine: what an update is asked, what it is told, and when."""

import pytest
import torch

from tscp.datasets import variance_shift
from tscp.methods import ConformalPredictor
from tscp.online import OnlineUpdate
from tscp.scores import AbsoluteScore, SignedScore
from tscp.sources import ExpandingSource, HoldoutSource
from tscp.weights import ExponentialWeights


class ExpertSpy(OnlineUpdate):
    """Asks for three levels at once, answers with the middle one, and records events."""

    needs_levels = True

    def reset(self, alpha, device):
        self.alpha = alpha
        self.answered = 0
        self.events = []  # (t, released, levels, queries answered when it arrived)

    def threshold(self, t, distribution):
        assert t == self.answered  # queries come in time order
        levels = torch.stack([self.alpha / 2, self.alpha, 2 * self.alpha])
        thresholds = distribution.threshold(levels)
        assert thresholds.shape == levels.shape
        self.answered += 1
        return thresholds[1]

    def observe(self, t, scores, thresholds, released, levels=None):
        self.events.append((t, released.clone(), levels.clone(), self.answered))


SETUPS = {
    "holdout": lambda: {"source": HoldoutSource()},
    "expanding": lambda: {"source": ExpandingSource()},
    "expanding_weighted": lambda: {"source": ExpandingSource(), "weights": ExponentialWeights()},
}


@pytest.mark.parametrize("setup", sorted(SETUPS))
@pytest.mark.parametrize("score", [AbsoluteScore, SignedScore])
def test_a_neutral_update_reproduces_the_offline_regions(setup, score):
    archive = variance_shift(horizon=3, num_origins=300, num_series=2)
    offline = ConformalPredictor(score(), **SETUPS[setup]())
    online = ConformalPredictor(score(), update=ExpertSpy(), chunk_size=7, **SETUPS[setup]())
    expected = offline.calibrate(archive).predict(archive)
    region = online.calibrate(archive).predict(archive)
    for name, value in region.tensors().items():
        assert torch.equal(value, expected.tensors()[name])


def test_outcomes_arrive_once_their_target_time_has_passed():
    archive = variance_shift(horizon=3, num_origins=200)
    spy = ExpertSpy()
    ConformalPredictor(AbsoluteScore(), update=spy).calibrate(archive).predict(archive)
    num_queries = len(archive.split("test"))
    for t, released, _, answered in spy.events:
        step = int(released.reshape(3, -1).any(dim=1).nonzero())
        # Step h of query t is known h base steps later: just before query t + h is answered.
        assert answered == t + step + 1
    # Step h of query t arrives with query t + h, so step h has num_queries - h outcomes.
    assert len(spy.events) == sum(num_queries - h for h in (1, 2, 3))


def test_realized_levels_are_conformal_p_values():
    archive = variance_shift(horizon=1, num_origins=300)
    spy = ExpertSpy()
    ConformalPredictor(AbsoluteScore(), update=spy).calibrate(archive).predict(archive)
    pool = archive.split("calibration").residuals.abs().double().flatten()
    scores = archive.split("test").residuals.abs().double().flatten()
    for t, _, levels, _ in spy.events:
        expected = ((pool >= scores[t]).sum() + 1).double() / (len(pool) + 1)
        assert torch.allclose(levels.flatten(), expected.reshape(1))

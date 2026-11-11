"""Online feedback hides other targets and completes missing-data queries."""

from dataclasses import replace

import pytest
import torch

from tscp.methods import ConformalPredictor
from tscp.online import OnlineUpdate
from tscp.online.dtaci import DtACIUpdate
from tscp.online.pid import PIDUpdate
from tscp.online.scorecaster import ThetaScorecaster
from tscp.scores import AbsoluteScore, SignedScore
from tscp.testing import MaxAbsScore, synthetic_archive


class FeedbackSpy(OnlineUpdate):
    needs_levels = True

    def reset(self, alpha, device):
        self.alpha = alpha
        self.answered = 0
        self.events = []
        self.completed = []
        self.counts = {}

    def threshold(self, t, distribution):
        self.answered += 1
        self.counts[t] = 0
        return distribution.threshold(self.alpha)

    def observe(self, t, scores, thresholds, released, levels=None):
        assert not bool(scores[~released].any())
        assert levels is not None and not bool(levels[~released].any())
        self.counts[t] += 1
        self.events.append((t, scores.clone(), released.clone()))

    def finish_query(self, t):
        assert t not in {event[0] for event in self.completed}
        self.completed.append((t, self.answered, self.counts.pop(t)))


def missing_archive(mode):
    archive = synthetic_archive(horizon=3, num_origins=80, num_series=2)
    observed = torch.ones_like(archive.residuals, dtype=torch.bool)
    test = archive.splits["test"]
    if mode == "all":
        observed[test] = False
    elif mode == "some":
        observed[test, 1, 0] = False
        observed[test[::2], :, 1] = False
    return replace(archive, observed=observed, residuals=archive.residuals * observed)


@pytest.mark.parametrize("missing", ["none", "some", "all"])
@pytest.mark.parametrize("score", [AbsoluteScore, SignedScore])
def test_step_feedback_is_masked_and_every_elapsed_query_finishes(missing, score):
    archive = missing_archive(missing)
    spy = FeedbackSpy()
    method = ConformalPredictor(score(), update=spy).calibrate(archive)
    method.predict(archive)
    queries = len(archive.splits["test"])
    assert spy.completed == [(t, t + 3, 3) for t in range(queries - 3)]
    expected_scores = score()(archive.split("test"))
    for t, actual, released in spy.events:
        torch.testing.assert_close(actual[released], expected_scores[t][released])
    assert set(spy.counts) == set(range(queries - 3, queries))


def test_joint_scores_complete_after_the_last_target_even_if_invalid():
    archive = missing_archive("some")
    spy = FeedbackSpy()
    ConformalPredictor(MaxAbsScore(), update=spy).calibrate(archive).predict(archive)
    queries = len(archive.splits["test"])
    assert spy.completed == [(t, t + 3, 1) for t in range(queries - 3)]


@pytest.mark.parametrize("missing", ["some", "all"])
def test_dtaci_retains_only_queries_with_unelapsed_target_times(missing):
    archive = missing_archive(missing)
    update = DtACIUpdate(gammas=[0.01, 0.1])
    ConformalPredictor(AbsoluteScore(), update=update).calibrate(archive).predict(archive)
    queries = len(archive.splits["test"])
    assert set(update._issued) == set(range(queries - 3, queries))


@pytest.mark.parametrize("missing", ["some", "all"])
def test_pid_completes_scorecaster_queries_without_fabricating_observations(missing):
    archive = missing_archive(missing)
    # The warm-up is deliberately longer than this archive; no optional model is fitted.
    caster = ThetaScorecaster(min_history=1000)
    update = PIDUpdate(scorecaster=caster)
    ConformalPredictor(AbsoluteScore(), update=update).calibrate(archive).predict(archive)
    queries = len(archive.splits["test"])
    assert set(caster._pending) == set(range(queries - 3, queries))
    cal = archive.split("calibration")
    test = archive.split("test")
    for step in range(archive.horizon):
        for series in range(archive.num_series):
            expected = int(cal.observed[:, step, series].sum())
            expected += int(test.observed[: queries - step - 1, step, series].sum())
            assert len(caster._values[step * archive.num_series + series]) == expected

from dataclasses import replace

import torch

from tscp.methods import ConformalPredictor
from tscp.methods.engine import OnlineEngine, release_schedule, replay
from tscp.online import ACIUpdate
from tscp.quantile import WeightedQuantile
from tscp.scores import AbsoluteScore
from tscp.sources import ExpandingSource
from tscp.testing import assert_regions_equal, check_method, synthetic_archive


class AdaptiveQuantile(WeightedQuantile):
    adapts_online = True

    def reset_online(self):
        self.seen = []

    def forward(self, pool, weights, query):
        origin = query.origins[0]
        timeline = query.timeline
        assert len(query) == 1
        assert timeline.origins[-1] == origin
        assert (pool.rows.timeline.origins <= origin).all()
        unavailable = timeline.target_times > origin
        assert not timeline.residuals[unavailable].any()
        assert not timeline.observed[unavailable].any()
        assert not query.observed.any()
        assert not pool.scores[~pool.valid].any()
        self.seen.append(int(origin))
        return super().forward(pool, weights, query)


def test_adaptation_contracts_and_replay_reset():
    def factory():
        return ConformalPredictor(
            AbsoluteScore(),
            source=ExpandingSource(30),
            quantile=AdaptiveQuantile(),
            update=ACIUpdate(),
        )

    check_method(factory)
    archive = synthetic_archive(horizon=3, num_origins=120)
    method = factory().calibrate(archive)
    state = {k: v.clone() for k, v in method.state_dict().items()}
    first = method.predict(archive)
    assert len(method.quantile.seen) == len(archive.splits["test"])
    assert_regions_equal(first, method.predict(archive))
    assert all(torch.equal(v, method.state_dict()[k]) for k, v in state.items())


def test_adaptive_neutral_quantile_equals_static_and_hides_future():
    archive = synthetic_archive(horizon=3, num_origins=160, missing=0.1)
    adaptive = ConformalPredictor(
        AbsoluteScore(), source=ExpandingSource(30), quantile=AdaptiveQuantile()
    )
    static = ConformalPredictor(AbsoluteScore(), source=ExpandingSource(30))
    assert_regions_equal(
        adaptive.calibrate(archive).predict(archive), static.calibrate(archive).predict(archive)
    )
    test = archive.split("test")
    origin = test.origins[5]
    changed = replace(
        archive,
        residuals=torch.where(
            (archive.target_times > origin)[..., None]
            & (archive.observed if archive.observed is not None else True),
            torch.full_like(archive.residuals, 1e6),
            archive.residuals,
        ),
    )
    expected = adaptive.calibrate(archive).predict(archive)[:6]
    assert_regions_equal(expected, adaptive.calibrate(changed).predict(changed)[:6])


def test_release_scheduler_and_lazy_distribution_order():
    origins, offsets = torch.tensor([0, 2, 5, 6]), torch.tensor([1, 4])
    schedule = list(release_schedule(origins, offsets))
    assert schedule == [(0, []), (1, [(0, 0)]), (2, [(1, 0), (0, 1)]), (3, [(2, 0), (1, 1)])]

    class Spy(ACIUpdate):
        def reset(self, alpha, device):
            super().reset(alpha, device)
            self.released = []

        def observe(self, t, scores, thresholds, released, levels=None):
            self.released.append(t)
            super().observe(t, scores, thresholds, released, levels)

    update = Spy()
    engine = OnlineEngine(update, torch.full((1, 1), 0.2), True, 1)
    quantile = WeightedQuantile()
    archive = synthetic_archive(horizon=1, num_origins=80)
    pool_rows = archive.split("calibration")
    from tscp.quantile import Pool

    pool = Pool(pool_rows.residuals.abs(), pool_rows)

    def provide(t):
        assert update.released == list(range(t))
        return quantile(pool, None, archive.head(1))

    replay(
        engine,
        torch.arange(4),
        torch.tensor([1]),
        provide,
        torch.zeros(4, 1, 1),
        None,
        torch.empty(4, 1, 1),
    )

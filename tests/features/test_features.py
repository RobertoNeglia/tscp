from dataclasses import replace
from datetime import UTC, datetime

import pytest
import torch

from tscp.datasets import gaussian_control
from tscp.features import Calendar, Concat, ContextColumns, Phase, ResidualHistory, calendar_fields
from tscp.testing import check_feature_map, synthetic_archive, with_missing_targets

MAPS = {
    "history": lambda: ResidualHistory(3),
    "history_absolute": lambda: ResidualHistory(2, absolute=True),
    "phase": lambda: Phase(24),
    "phase_cyclic": lambda: Phase(7, offset=2, encoding="cyclic"),
    "calendar": lambda: Calendar(["hour", "dayofweek"]),
    "calendar_cyclic": lambda: Calendar(["month", "dayofyear", "minute"], encoding="cyclic"),
    "context": ContextColumns,
    "context_named": lambda: ContextColumns(["temperature"]),
    "concat": lambda: Concat([Phase(24), ResidualHistory(2), ContextColumns()]),
}


@pytest.mark.parametrize("name", sorted(MAPS))
def test_feature_maps_keep_the_feature_map_contract(name):
    check_feature_map(MAPS[name])


def test_residual_history_holds_the_last_known_residuals_of_each_step():
    archive = gaussian_control(horizon=2, num_origins=60, num_series=2, seed=3)
    values = ResidualHistory(lags=3)(archive)
    valid = ResidualHistory(lags=3).valid(archive)
    for t in range(archive.num_origins):
        for index, step in enumerate((1, 2)):
            for lag in (1, 2, 3):
                # At origin t, the newest known residual of step h is the one of origin t - h.
                source = t - step - lag + 1
                expected = archive.residuals[source, index] if source >= 0 else torch.zeros(2)
                assert torch.equal(values[t, index, :, lag - 1], expected)
            assert bool(valid[t, index].all()) == (t - step - 2 >= 0)


def test_residual_history_skips_unobserved_targets():
    archive = with_missing_targets(gaussian_control(horizon=1, num_origins=80), 0.3, seed=1)
    history = ResidualHistory(lags=1)
    values, valid = history(archive), history.valid(archive)
    previous = archive.observed[:-1, 0, 0]  # lag 1 at origin t is the target of origin t - 1
    assert torch.equal(valid[1:, 0, 0], previous)
    assert (values[1:, 0, 0, 0][~previous] == 0).all()


def test_calendar_fields_agree_with_datetime():
    generator = torch.Generator().manual_seed(0)
    seconds = torch.randint(-2_208_988_800, 4_102_444_800, (2000,), generator=generator)
    edges = [
        datetime(2000, 2, 29, 23, 59, tzinfo=UTC),
        datetime(1900, 3, 1, tzinfo=UTC),
        datetime(1969, 12, 31, 23, 59, 59, tzinfo=UTC),
        datetime(1970, 1, 1, tzinfo=UTC),
        datetime(2024, 12, 31, 12, tzinfo=UTC),
        datetime(2100, 2, 28, 6, tzinfo=UTC),
    ]
    seconds = torch.cat([seconds, torch.tensor([int(edge.timestamp()) for edge in edges])])
    fields = calendar_fields(seconds * 1_000_000_000)
    for index, second in enumerate(seconds.tolist()):
        moment = datetime.fromtimestamp(second, tz=UTC)
        expected = {
            "minute": moment.minute,
            "hour": moment.hour,
            "dayofweek": moment.weekday(),
            "day": moment.day,
            "month": moment.month,
            "dayofyear": moment.timetuple().tm_yday,
        }
        assert {name: int(values[index]) for name, values in fields.items()} == expected, moment


def test_calendar_and_phase_read_the_synthetic_hours():
    archive = synthetic_archive(horizon=1, num_origins=200)  # hourly from a Monday midnight
    hours = archive.origins % 24
    assert torch.equal(Calendar(["hour"])(archive)[:, 0], hours.float())
    assert torch.equal(Phase(24)(archive)[:, 0], hours.float())
    weekdays = Calendar(["dayofweek"])(archive)[:, 0]
    assert torch.equal(weekdays, ((archive.origins // 24) % 7).float())
    cyclic = Calendar(["hour", "month"], encoding="cyclic")(archive)
    assert cyclic.shape == (200, 4)
    assert torch.allclose(cyclic[:, :2].square() + cyclic[:, 2:].square(), torch.ones(200, 2))
    with pytest.raises(ValueError, match="timestamps"):
        Calendar()(replace(archive, timestamps=None))
    with pytest.raises(ValueError, match="fields"):
        Calendar(["week"])


def test_context_columns_select_by_name():
    archive = synthetic_archive(horizon=1, num_origins=100)
    assert torch.equal(ContextColumns()(archive), archive.context)
    assert torch.equal(ContextColumns(["temperature"])(archive), archive.context[:, 1:])
    with pytest.raises(ValueError, match="no columns"):
        ContextColumns(["wind"])(archive)
    with pytest.raises(ValueError, match="no context"):
        ContextColumns()(gaussian_control(horizon=1, num_origins=50))


def test_concat_repeats_coarser_maps_and_joins_validity():
    archive = gaussian_control(horizon=2, num_origins=50, num_series=3)
    phase, history = Phase(5), ResidualHistory(2)
    joined = Concat([phase, history])
    values = joined(archive)
    assert values.shape == (50, 2, 3, 3)
    assert torch.equal(values[..., 0], phase(archive)[:, None, None, 0].expand(50, 2, 3))
    assert torch.equal(values[..., 1:], history(archive))
    assert torch.equal(joined.valid(archive), history.valid(archive))
    assert Concat([phase, Phase(7)]).valid(archive) is None

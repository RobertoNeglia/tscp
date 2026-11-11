import pytest
import torch

from tscp.data import ResidualArchive, SplitPlan, chronological_split


def test_splits_skip_the_origins_that_would_share_targets():
    origins, steps = torch.arange(100), torch.arange(1, 4)
    plan = chronological_split(origins, steps, {"fit": 20, "calibration": 30, "test": None})
    fit, calibration, test = plan["fit"], plan["calibration"], plan["test"]
    assert fit.tolist() == list(range(20))
    # The last fit target is at 19 + 3 = 22; calibration starts at origin 22 (target 23).
    assert int(calibration[0]) == 22 and calibration.numel() == 30
    assert int(test[0]) == int(calibration[-1]) + 3
    assert int(test[-1]) == 99


def test_one_step_splits_are_contiguous():
    plan = chronological_split(torch.arange(50), torch.tensor([1]), {"calibration": 20, "test": 30})
    assert int(plan["calibration"][-1]) + 1 == int(plan["test"][0])


def test_fractions_are_of_all_rows():
    plan = chronological_split(
        torch.arange(1000), torch.tensor([1]), {"fit": 0.2, "calibration": 0.3, "test": None}
    )
    assert plan.sizes() == {"fit": 200, "calibration": 300, "test": 500}
    assert plan.names == ("fit", "calibration", "test")


def test_splits_follow_time_order_whatever_the_key_order():
    plan = chronological_split(torch.arange(30), torch.tensor([1]), {"test": None, "fit": 10})
    assert plan.names == ("fit", "test")


@pytest.mark.parametrize(
    ("sizes", "message"),
    [
        ({"calibration": 8, "test": 5}, "not enough origins"),
        ({"calibration": None, "test": 5}, "only the last split"),
        ({"calibration": 1.5, "test": None}, r"\(0, 1\)"),
        ({"calibration": True, "test": None}, "int, a float or None"),
        ({"train": 5}, "unknown split"),
        ({}, "names no split"),
    ],
)
def test_invalid_sizes_raise(sizes, message):
    with pytest.raises(ValueError, match=message):
        chronological_split(torch.arange(10), torch.tensor([1]), sizes)


def test_shared_target_times_are_rejected():
    origins, steps = torch.arange(20), torch.arange(1, 4)
    too_close = SplitPlan({"calibration": torch.arange(0, 10), "test": torch.arange(11, 20)})
    with pytest.raises(ValueError, match="share target times"):
        too_close.validate(origins, steps)
    # Last calibration target 9 + 3 = 12; the first test target 12 + 1 = 13 comes after it.
    SplitPlan({"calibration": torch.arange(0, 10), "test": torch.arange(12, 20)}).validate(
        origins, steps
    )


@pytest.mark.parametrize(
    "indices",
    [
        {"calibration": torch.arange(5, 10), "test": torch.arange(0, 5)},
        {"calibration": torch.arange(0, 10), "test": torch.arange(5, 15)},
    ],
)
def test_overlapping_or_reversed_splits_are_rejected(indices):
    with pytest.raises(ValueError, match="must start after"):
        SplitPlan(indices).validate(torch.arange(20), torch.tensor([1]))


@pytest.mark.parametrize(
    "rows", [torch.tensor([3, 2]), torch.tensor([-1, 0]), torch.tensor([]), torch.tensor([0.5])]
)
def test_split_rows_must_be_increasing_integers(rows):
    with pytest.raises(ValueError):
        SplitPlan({"test": rows})


def test_rows_past_the_archive_are_rejected():
    with pytest.raises(ValueError, match="past the archive"):
        SplitPlan({"test": torch.arange(5, 12)}).validate(torch.arange(10), torch.tensor([1]))


def test_the_archive_validates_its_plan():
    with pytest.raises(ValueError, match="share target times"):
        ResidualArchive(
            residuals=torch.zeros(20, 2, 1),
            origins=torch.arange(20),
            steps=torch.tensor([1, 2]),
            splits={"calibration": torch.arange(0, 10), "test": torch.arange(10, 20)},
        )

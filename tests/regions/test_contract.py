"""Checks every region type must pass."""

import pytest
import torch

from tscp.regions import Box, Interval, move_event_dims, restore_event_dims
from tscp.testing import check_region

Q, H, N = 6, 3, 2
SIZES = {"step": H, "series": N}
EVENT_DIMS = {"interval": (), "box_step": ("step",), "box_series": ("series",)}
EVENT_DIMS["box_step_series"] = ("step", "series")


def bounds_shape(event_dims) -> list[int]:
    kept = [size for name, size in SIZES.items() if name not in event_dims]
    event = [size for name, size in SIZES.items() if name in event_dims]
    return [Q, *kept, *event]


def make_region(name: str):
    """Random bounds, with batch element 0 unbounded, 1 empty and 2 of zero width."""
    event_dims = EVENT_DIMS[name]
    shape = bounds_shape(event_dims)
    generator = torch.Generator().manual_seed(len(name))
    lower = torch.randn(shape, generator=generator) - 1
    upper = lower + 0.2 + 2 * torch.rand(shape, generator=generator)
    first = (0,) * (len(shape) - 1)  # the first kept element and the first coordinate
    lower[(0, *first)], upper[(0, *first)] = -torch.inf, torch.inf
    lower[(1, *first)], upper[(1, *first)] = 1.0, -1.0
    lower[(2, *first)], upper[(2, *first)] = 0.5, 0.5
    if not event_dims:
        return Interval(lower, upper)
    return Box(lower, upper, event_dims)


def enclosing(region, margin=0.5):
    """Finite boxes around the regions, for the Monte Carlo volume check."""
    lower, upper = region.lower - margin, region.upper + margin
    finite = torch.isfinite(lower) & torch.isfinite(upper)
    return torch.where(finite, lower, 0.0), torch.where(finite, upper, 1.0)


@pytest.mark.parametrize("name", sorted(EVENT_DIMS))
def test_regions_keep_the_region_contract(name):
    check_region(lambda: make_region(name), enclosing=enclosing(make_region(name)))


@pytest.mark.parametrize("name", sorted(EVENT_DIMS))
def test_log_volume_conventions(name):
    region = make_region(name)
    log_volume = region.log_volume()
    special = (0,) * (len(region.batch_shape) - 1)
    assert log_volume[(0, *special)] == torch.inf  # unbounded
    assert log_volume[(1, *special)] == -torch.inf  # empty
    assert log_volume[(2, *special)] == -torch.inf  # zero width
    assert torch.isfinite(log_volume[3:]).all()


@pytest.mark.parametrize("name", sorted(EVENT_DIMS))
def test_joint_membership_is_every_coordinate_inside(name):
    """A region over some dimensions covers a point when all its coordinates are covered."""
    region = make_region(name)
    points = torch.randn(Q, H, N) * 2
    per_coordinate = Interval(
        restore_event_dims(region.lower, region.event_dims),
        restore_event_dims(region.upper, region.event_dims),
    ).contains(points)
    joint = [1 + i for i, dim in enumerate(SIZES) if dim in region.event_dims]
    expected = per_coordinate.all(dim=tuple(joint)) if joint else per_coordinate
    assert torch.equal(region.contains(points), expected)


def test_moving_event_dims_last_and_back():
    values = torch.arange(2 * Q * H * N).reshape(2, Q, H, N)
    assert move_event_dims(values, ("step",)).shape == (2, Q, N, H)
    assert torch.equal(move_event_dims(values, ("step",))[0, 0, 1], values[0, 0, :, 1])
    assert move_event_dims(values, ("series",)).shape == (2, Q, H, N)
    for event_dims in EVENT_DIMS.values():
        moved = move_event_dims(values, event_dims)
        assert torch.equal(restore_event_dims(moved, event_dims), values)


def test_box_checks_its_arguments():
    bounds = torch.zeros(Q, N, H)
    with pytest.raises(ValueError, match="at least one"):
        Box(bounds, bounds, event_dims=())
    with pytest.raises(ValueError, match="event_dims"):
        Box(bounds, bounds, event_dims=("time",))
    with pytest.raises(ValueError, match="same shape"):
        Box(bounds, bounds[:-1], event_dims=("step",))
    box = Box(torch.zeros(Q, H, N), torch.ones(Q, H, N), ("series", "step"))
    assert box.event_dims == ("step", "series")

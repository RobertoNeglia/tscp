"""Checks every region type must pass."""

import math

import pytest
import torch

from tscp.regions import Box, Interval, move_event_dims

Q, H, N = 6, 3, 2
SIZES = {"step": H, "series": N}
EVENT_DIMS = {"interval": (), "box_step": ("step",), "box_series": ("series",)}
EVENT_DIMS["box_step_series"] = ("step", "series")
DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    ),
]


def bounds_shape(event_dims) -> list[int]:
    kept = [size for name, size in SIZES.items() if name not in event_dims]
    event = [size for name, size in SIZES.items() if name in event_dims]
    return [Q, *kept, *event]


def make_region(name: str):
    """Random bounds, with batch element 0 unbounded, 1 empty and 2 of zero width."""
    event_dims = EVENT_DIMS[name]
    shape = bounds_shape(event_dims)
    lower = torch.randn(shape) - 1
    upper = lower + 0.2 + 2 * torch.rand(shape)
    first = (0,) * (len(shape) - 1)  # the first kept element and the first coordinate
    lower[(0, *first)], upper[(0, *first)] = -torch.inf, torch.inf
    lower[(1, *first)], upper[(1, *first)] = 1.0, -1.0
    lower[(2, *first)], upper[(2, *first)] = 0.5, 0.5
    if not event_dims:
        return Interval(lower, upper)
    return Box(lower, upper, event_dims)


def special(region, query: int) -> tuple[int, ...]:
    """Index of the special batch element of a query origin."""
    return (query,) + (0,) * (len(region.batch_shape) - 1)


def back_to_residuals(moved: torch.Tensor, event_dims) -> torch.Tensor:
    """Invert :func:`move_event_dims`: ``[..., Q, *kept, *event]`` back to ``[..., Q, H, N]``."""
    order = [name for name in SIZES if name not in event_dims]
    order += [name for name in SIZES if name in event_dims]
    lead = moved.ndim - 3
    permutation = list(range(lead + 1)) + [lead + 1 + order.index(name) for name in SIZES]
    return moved.permute(permutation)


@pytest.fixture(params=sorted(EVENT_DIMS))
def region(request):
    return make_region(request.param)


def test_answers_have_the_batch_shape(region):
    covered = region.contains(torch.randn(Q, H, N))
    assert covered.dtype == torch.bool and covered.shape == region.batch_shape
    assert region.log_volume().shape == region.batch_shape
    expected = math.prod(SIZES[name] for name in region.event_dims) if region.event_dims else 1
    assert region.dimension == expected


def test_log_volume_conventions(region):
    log_volume = region.log_volume()
    assert log_volume[special(region, 0)] == torch.inf  # unbounded
    assert log_volume[special(region, 1)] == -torch.inf  # empty
    assert log_volume[special(region, 2)] == -torch.inf  # zero width
    assert torch.isfinite(log_volume[3:]).all()


def test_joint_membership_is_every_coordinate_inside(region):
    """A region over some dimensions covers a point when all its coordinates are covered."""
    points = torch.randn(Q, H, N) * 2
    per_coordinate = Interval(
        back_to_residuals(region.lower, region.event_dims),
        back_to_residuals(region.upper, region.event_dims),
    ).contains(points)
    joint = [1 + i for i, name in enumerate(SIZES) if name in region.event_dims]
    expected = per_coordinate.all(dim=tuple(joint)) if joint else per_coordinate
    assert torch.equal(region.contains(points), expected)


def test_log_volume_matches_a_monte_carlo_estimate(region):
    """Sample uniformly in a slightly larger box and count the points inside."""
    samples, margin = 4000, 0.5
    enclosing_lower, enclosing_upper = region.lower - margin, region.upper + margin
    finite = torch.isfinite(enclosing_lower) & torch.isfinite(enclosing_upper)
    enclosing_lower = torch.where(finite, enclosing_lower, 0.0)
    enclosing_upper = torch.where(finite, enclosing_upper, 1.0)
    uniform = torch.rand(samples, *region.lower.shape)
    moved = enclosing_lower + uniform * (enclosing_upper - enclosing_lower)
    points = back_to_residuals(moved, region.event_dims)
    inside = torch.stack([region.contains(point) for point in points]).double().mean(dim=0)
    width = (enclosing_upper - enclosing_lower).reshape(*region.batch_shape, -1)
    enclosing_log_volume = torch.log(width.double()).sum(dim=-1)
    estimate = inside * enclosing_log_volume.exp()
    exact = region.log_volume().double().exp()
    error = (inside * (1 - inside) / samples).sqrt() * enclosing_log_volume.exp()
    proper = torch.isfinite(region.log_volume())
    assert proper.sum() > 0
    assert ((estimate - exact).abs() <= 5 * error + 1e-12)[proper].all()


def test_shifting_by_forecasts_preserves_membership_and_volume(region):
    residuals, forecasts = torch.randn(Q, H, N), 10 * torch.randn(Q, H, N)
    shifted = region.to_target(forecasts)
    assert type(shifted) is type(region) and shifted.event_dims == region.event_dims
    assert torch.equal(shifted.contains(residuals + forecasts), region.contains(residuals))
    finite = torch.isfinite(region.log_volume())
    assert torch.allclose(shifted.log_volume()[finite], region.log_volume()[finite], atol=1e-5)
    assert torch.equal(shifted.log_volume()[~finite], region.log_volume()[~finite])


def test_indexing_selects_along_batch_dimensions(region):
    tensors = region.tensors()
    for index in (1, slice(1, 4), torch.tensor([True, False] * (Q // 2))):
        selected = region[index]
        assert type(selected) is type(region)
        for name, value in selected.tensors().items():
            assert torch.equal(value, tensors[name][index])
    assert region[1].batch_shape == region.batch_shape[1:]
    with pytest.raises(IndexError, match="Ellipsis"):
        region[..., 0]
    with pytest.raises(IndexError, match="too many"):
        region[(0,) * (len(region.batch_shape) + 1)]


@pytest.mark.parametrize("device", DEVICES)
def test_regions_move_between_devices(region, device):
    moved = region.to(device)
    assert all(value.device.type == device for value in moved.tensors().values())
    back = moved.to("cpu")
    for name, value in back.tensors().items():
        assert torch.equal(value, region.tensors()[name])


def test_move_event_dims_puts_joint_dimensions_last():
    values = torch.arange(Q * H * N).reshape(Q, H, N)
    assert move_event_dims(values, ()).shape == (Q, H, N)
    assert move_event_dims(values, ("step",)).shape == (Q, N, H)
    assert torch.equal(move_event_dims(values, ("step",))[0, 1], values[0, :, 1])
    assert move_event_dims(values, ("series",)).shape == (Q, H, N)


def test_box_checks_its_arguments():
    bounds = torch.zeros(Q, N, H)
    with pytest.raises(ValueError, match="at least one"):
        Box(bounds, bounds, event_dims=())
    with pytest.raises(ValueError, match="event_dims"):
        Box(bounds, bounds, event_dims=("time",))
    with pytest.raises(ValueError, match="same shape"):
        Box(bounds, bounds[:-1], event_dims=("step",))
    assert Box(torch.zeros(Q, H, N), torch.ones(Q, H, N), ("series", "step")).event_dims == (
        "step",
        "series",
    )

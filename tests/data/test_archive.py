from dataclasses import replace

import pytest
import torch

from tscp.data import Graph, ResidualArchive, availability_mask, target_times


def archive(num_origins=10, horizon=2, num_series=3, **fields) -> ResidualArchive:
    return ResidualArchive(
        residuals=fields.pop("residuals", torch.randn(num_origins, horizon, num_series)),
        origins=fields.pop("origins", torch.arange(num_origins)),
        steps=fields.pop("steps", torch.arange(1, horizon + 1)),
        **fields,
    )


def test_shapes_and_target_times():
    a = archive()
    assert (a.num_origins, a.horizon, a.num_series, len(a)) == (10, 2, 3, 10)
    assert a.target_times[3].tolist() == [4, 5]
    assert "T=10" in repr(a) and "H=2" in repr(a)


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"residuals": torch.zeros(10, 2, 3, dtype=torch.int64)}, "floating"),
        ({"residuals": torch.zeros(10, 2)}, r"\[T, H, N\]"),
        ({"origins": torch.tensor([0, 1, 2, 3, 4, 4, 6, 7, 8, 9])}, "strictly increase"),
        ({"origins": torch.arange(10.0)}, "integers"),
        ({"steps": torch.tensor([0, 1])}, "positive"),
        ({"forecasts": torch.zeros(10, 2, 2)}, "forecasts"),
        ({"outputs": {"scale": torch.ones(9, 2, 3)}}, "output 'scale'"),
        ({"outputs": {"bad name": torch.ones(10, 2, 3)}}, "identifier"),
        ({"context": torch.zeros(9, 4)}, "context"),
        ({"context": torch.zeros(10, 4), "context_names": ("a",)}, "context_names"),
        ({"context_names": ("a",)}, "without context"),
        ({"series": ("a", "b")}, "series"),
        ({"graph": Graph(torch.tensor([[0], [1]]), num_nodes=2)}, "num_nodes"),
    ],
)
def test_invalid_fields_raise(fields, message):
    with pytest.raises(ValueError, match=message):
        archive(**fields)


def test_unobserved_residuals_must_be_zero_and_observed_ones_finite():
    residuals = torch.randn(10, 2, 3)
    observed = torch.ones(10, 2, 3, dtype=torch.bool)
    observed[0, 0, 0] = False
    with pytest.raises(ValueError, match="exactly 0"):
        archive(residuals=residuals, observed=observed)
    residuals[0, 0, 0] = 0.0
    archive(residuals=residuals, observed=observed)
    residuals[1, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        archive(residuals=residuals, observed=observed)


def test_select_and_split_slice_every_row_field():
    a = archive(
        forecasts=torch.randn(10, 2, 3),
        outputs={"scale": torch.rand(10, 2, 3)},
        context=torch.randn(10, 3, 4),
        observed=torch.ones(10, 2, 3, dtype=torch.bool),
        splits={"calibration": torch.arange(0, 4), "test": torch.arange(6, 10)},
    )
    calibration = a.split("calibration")
    assert calibration.splits is None
    assert calibration.origins.tolist() == [0, 1, 2, 3]
    for field in ("forecasts", "context", "observed"):
        assert torch.equal(getattr(calibration, field), getattr(a, field)[:4])
    assert torch.equal(calibration.outputs["scale"], a.outputs["scale"][:4])
    with pytest.raises(KeyError, match="no 'fit' split"):
        a.split("fit")
    with pytest.raises(KeyError, match="row subset"):
        calibration.split("calibration")


@pytest.mark.parametrize("rows", [torch.tensor([2, 1]), torch.tensor([0, 10]), torch.tensor([])])
def test_select_rejects_bad_rows(rows):
    with pytest.raises(ValueError):
        archive().select(rows)


def test_fingerprint_tracks_the_arrays():
    a = archive()
    assert a.fingerprint() == replace(a, residuals=a.residuals.clone()).fingerprint()
    assert a.fingerprint() != replace(a, residuals=a.residuals + 1).fingerprint()
    unhashed = replace(a, provenance={"note": "metadata is not hashed"})
    assert a.fingerprint() == unhashed.fingerprint()
    assert a.to("cpu").fingerprint() == a.fingerprint()


def test_target_times_and_availability():
    origins, steps = torch.tensor([0, 1, 2]), torch.tensor([1, 2])
    assert target_times(origins, steps).tolist() == [[1, 2], [2, 3], [3, 4]]
    mask = availability_mask(origins, torch.tensor([2, 3]), steps)
    assert mask.shape == (2, 3, 2)
    # At origin 2: (0, step 1) and (0, step 2) and (1, step 1) are known.
    assert mask[0].tolist() == [[True, True], [True, False], [False, False]]
    assert mask[1].tolist() == [[True, True], [True, True], [True, False]]

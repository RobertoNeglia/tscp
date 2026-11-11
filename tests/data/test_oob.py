import pytest
import torch

from tscp.data import load_archive, save_archive
from tscp.data.oob import from_bootstrap_predictions


def inputs():
    truth = torch.arange(10, dtype=torch.float64)
    members = torch.arange(30, dtype=torch.float64).reshape(10, 3) / 4
    in_bag = torch.tensor(
        [[True, False, False, True], [False, True, False, False], [False, False, True, False]]
    )
    splits = {"calibration": torch.arange(4), "test": torch.arange(4, 10)}
    return truth, members, in_bag, splits


def test_bootstrap_aggregation_equals_mean_over_oob_ensembles(tmp_path):
    truth, members, in_bag, splits = inputs()
    archive = from_bootstrap_predictions(truth, members, in_bag, splits=splits)
    oob_predictions = torch.stack([members[:, ~in_bag[:, i]].mean(-1) for i in range(4)])
    expected = oob_predictions.mean(0)
    expected[:4] = oob_predictions.diagonal()[:4]
    torch.testing.assert_close(archive.forecasts[:, 0, 0], expected, rtol=1e-15, atol=1e-15)
    torch.testing.assert_close(archive.residuals[:, 0, 0], truth - expected, rtol=1e-15, atol=1e-15)
    assert torch.equal(archive.outputs["oob_count"][:4, 0, 0], (~in_bag).sum(0))
    path = tmp_path / "oob.npz"
    save_archive(archive, path)
    loaded = load_archive(path)
    assert loaded.fingerprint() == archive.fingerprint()
    assert loaded.provenance == archive.provenance


def test_bootstrap_builder_records_missing_truth_without_standardizing():
    truth, members, in_bag, splits = inputs()
    truth[2] = torch.nan
    archive = from_bootstrap_predictions(truth, members, in_bag, splits=splits)
    assert not archive.observed[2, 0, 0] and archive.residuals[2, 0, 0] == 0
    assert torch.isnan(archive.targets()[2, 0, 0])


@pytest.mark.parametrize("failure", ["no_oob", "membership", "shape", "finite", "reserved"])
def test_bootstrap_builder_rejects_ambiguous_or_invalid_inputs(failure):
    truth, members, in_bag, splits = inputs()
    kwargs = {}
    if failure == "no_oob":
        in_bag[:, 0] = True
    elif failure == "membership":
        in_bag = in_bag.long()
    elif failure == "shape":
        members = members[:-1]
    elif failure == "finite":
        members[0, 0] = torch.nan
    else:
        kwargs["provenance"] = {"oob": {}}
    with pytest.raises(ValueError):
        from_bootstrap_predictions(truth, members, in_bag, splits=splits, **kwargs)

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from tscp.data import SplitPlan
from tscp.data.oob import from_bootstrap_predictions
from tscp.methods.enbpi import EnbPI
from tscp.testing import assert_regions_equal, check_method

REFERENCE = Path(__file__).parents[1] / "fixtures" / "reference" / "enbpi_2023.npz"


def reference_archive():
    metadata = json.loads(REFERENCE.with_suffix(".json").read_text())
    with np.load(REFERENCE, allow_pickle=False) as data:
        values = {key: torch.from_numpy(data[key]) for key in data.files}
    count = metadata["parameters"]["calibration_size"]
    splits = {"calibration": torch.arange(count), "test": torch.arange(count, len(values["truth"]))}
    archive = from_bootstrap_predictions(
        values["truth"], values["member_forecasts"], values["in_bag"], splits=splits
    )
    return archive, values, metadata


def test_enbpi_aggregation_matches_pinned_journal_reference():
    archive, expected, metadata = reference_archive()
    tolerance = metadata["tolerance"]
    torch.testing.assert_close(archive.forecasts[:, 0, 0], expected["forecasts"], **tolerance)
    torch.testing.assert_close(archive.residuals[:, 0, 0], expected["residuals"], **tolerance)


def test_enbpi_intervals_match_pinned_journal_reference():
    archive, expected, metadata = reference_archive()
    method = EnbPI(alpha=metadata["parameters"]["alpha"]).calibrate(archive)
    region = method.predict(archive).to_target(archive.split("test").forecasts)
    actual = torch.stack([region.lower[:, 0, 0], region.upper[:, 0, 0]], dim=-1)
    torch.testing.assert_close(actual, expected["target_bounds"], **metadata["tolerance"])


def test_enbpi_keeps_supported_method_contracts():
    archive, _, _ = reference_archive()
    check_method(lambda: EnbPI(window_size=8), archives=[archive])


def test_enbpi_missing_residuals_consume_window_slots_without_imputation():
    archive, _, _ = reference_archive()
    observed = torch.ones_like(archive.residuals, dtype=torch.bool)
    observed[archive.splits["calibration"][-4:]] = False
    archive = replace(archive, observed=observed, residuals=archive.residuals * observed)
    method = EnbPI(window_size=4).calibrate(archive)
    region = method.predict(archive)
    assert torch.isneginf(region.lower[0]).all() and torch.isposinf(region.upper[0]).all()
    # At the second query only the first test residual is observed in the four-row window.
    value = archive.split("test").residuals[0]
    assert torch.equal(region.lower[1], value) and torch.equal(region.upper[1], value)
    assert_regions_equal(method.predict(archive), region)


def test_enbpi_rejects_other_calibration_values_even_at_the_same_origins():
    archive, _, _ = reference_archive()
    method = EnbPI().calibrate(archive)
    changed = archive.residuals.clone()
    changed[0] += 1
    with pytest.raises(ValueError, match="calibration archive"):
        method.predict(replace(archive, residuals=changed))


def test_enbpi_requires_oob_provenance_and_supported_shape():
    archive, _, _ = reference_archive()
    with pytest.raises(ValueError, match="from_bootstrap_predictions"):
        EnbPI().calibrate(replace(archive, provenance={}))
    with pytest.raises(ValueError, match="one step"):
        EnbPI().calibrate(replace(archive, steps=torch.tensor([2]), splits=None))
    with pytest.raises(ValueError, match="window_size"):
        EnbPI(window_size=0)
    with pytest.raises(RuntimeError, match="calibrate"):
        EnbPI().predict(archive)


def test_enbpi_rejects_resplitting_an_oob_aggregation():
    archive, _, metadata = reference_archive()
    count = metadata["parameters"]["calibration_size"] - 1
    changed = replace(
        archive,
        splits=SplitPlan(
            {"calibration": torch.arange(count), "test": torch.arange(count, len(archive))}
        ),
    )
    with pytest.raises(ValueError, match="aggregation rows"):
        EnbPI().calibrate(changed)

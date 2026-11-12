import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from tscp.data import ResidualArchive, SplitPlan
from tscp.imports import has_module
from tscp.methods.spci import SPCI
from tscp.quantile.forest import QuantileForest
from tscp.testing import assert_regions_equal, check_method

REFERENCE = Path(__file__).parents[1] / "fixtures" / "reference" / "spci_2023.npz"
FOREST = pytest.mark.skipif(not has_module("sklearn_quantile"), reason="needs the forest extra")


def archive_from_residuals(residuals, calibration_size):
    residuals = torch.as_tensor(residuals, dtype=torch.float64)
    return ResidualArchive.from_forecasts(
        residuals,
        torch.zeros_like(residuals),
        splits={
            "calibration": torch.arange(calibration_size),
            "test": torch.arange(calibration_size, len(residuals)),
        },
    )


@FOREST
def test_spci_matches_pinned_author_reference():
    metadata = json.loads(REFERENCE.with_suffix(".json").read_text())
    with np.load(REFERENCE, allow_pickle=False) as data:
        archive = archive_from_residuals(data["residuals"], metadata["parameters"]["window_size"])
        expected = torch.from_numpy(data["bounds"])
    method = SPCI(**metadata["parameters"]).calibrate(archive)
    region = method.predict(archive)
    actual = torch.stack([region.lower[:, 0, 0], region.upper[:, 0, 0]], dim=-1)
    torch.testing.assert_close(actual, expected, **metadata["tolerance"])


@FOREST
def test_spci_keeps_supported_method_contracts():
    archive = archive_from_residuals(torch.sin(torch.arange(18) * 0.7), 12)
    check_method(lambda: SPCI(lags=2, window_size=8, n_estimators=3, seed=9), archives=[archive])


@FOREST
def test_spci_accepts_overlapping_lower_and_upper_probability_grids():
    archive = archive_from_residuals(torch.sin(torch.arange(18) * 0.7), 12)
    method = SPCI(lags=2, window_size=8, n_estimators=3, alpha=0.7).calibrate(archive)
    region = method.predict(archive)
    assert torch.isfinite(region.lower).all() and torch.isfinite(region.upper).all()
    assert (region.lower <= region.upper).all()


def spy_forest(monkeypatch):
    calls = []
    monkeypatch.setattr(QuantileForest, "check_dependency", lambda self: None)

    def predict(self, features, targets, query, probabilities):
        calls.append((features.clone(), targets.clone(), query.clone()))
        return torch.quantile(targets, probabilities)

    monkeypatch.setattr(QuantileForest, "forward", predict)
    return calls


def test_spci_refits_only_on_released_training_pairs_and_resets(monkeypatch):
    calls = spy_forest(monkeypatch)
    archive = archive_from_residuals(torch.arange(18), 12)
    method = SPCI(lags=2, window_size=6).calibrate(archive)
    before = {key: value.clone() for key, value in method.state_dict().items()}
    region = method.predict(archive)
    assert len(calls) == 6
    for u, (features, targets, query) in enumerate(calls):
        expected = torch.arange(6 + u, 12 + u, dtype=torch.float64)
        assert torch.equal(features, expected.unfold(0, 3, 1)[:, :2])
        assert torch.equal(targets, expected[2:])
        assert torch.equal(query, expected[-2:])
    assert before.keys() == method.state_dict().keys()
    for key, value in before.items():
        assert torch.equal(value, method.state_dict()[key])
    assert_regions_equal(method.predict(archive), region)
    changed = archive.residuals.clone()
    changed[14:] += 10000
    other = method.predict(replace(archive, residuals=changed))
    assert_regions_equal(other[:3], region[:3])


def test_spci_skips_invalid_training_pairs_and_cannot_bridge_missing_query_lags(monkeypatch):
    calls = spy_forest(monkeypatch)
    archive = archive_from_residuals(torch.arange(18), 12)
    observed = torch.ones_like(archive.residuals, dtype=torch.bool)
    observed[9] = False
    archive = replace(archive, observed=observed, residuals=archive.residuals * observed)
    region = SPCI(lags=2, window_size=8).calibrate(archive).predict(archive)
    features, targets, _ = calls[0]
    assert not (features == 9).any() and not (targets == 9).any()
    assert torch.isfinite(region.lower[0]).all()
    observed[11] = False
    missing_last = replace(archive, observed=observed, residuals=archive.residuals * observed)
    region = SPCI(lags=2, window_size=8).calibrate(missing_last).predict(missing_last)
    assert torch.isneginf(region.lower[:2]).all() and torch.isposinf(region.upper[:2]).all()


def grid_archive(origins, calibration, test):
    origins = torch.as_tensor(origins)
    return ResidualArchive(
        residuals=origins.double()[:, None, None],  # each residual equals its origin
        origins=origins,
        steps=torch.tensor([1]),
        splits=SplitPlan({"calibration": torch.as_tensor(calibration), "test": test}),
    )


@pytest.mark.parametrize(
    "origins,calibration,test,cadence,absent",
    [
        # A row between the splits belongs to neither.
        (torch.arange(60), torch.arange(30), torch.arange(31, 60), 1, 30),
        # An origin is missing from the archive.
        (torch.cat([torch.arange(40), torch.arange(41, 61)]), torch.arange(30), None, 1, 40),
        # Origins on a grid of two base steps, with one missing.
        (
            torch.cat([torch.arange(0, 80, 2), torch.arange(82, 122, 2)]),
            torch.arange(30),
            None,
            2,
            80,
        ),
    ],
)
def test_spci_lags_stay_adjacent_on_the_origin_grid(
    monkeypatch, origins, calibration, test, cadence, absent
):
    calls = spy_forest(monkeypatch)
    test = torch.arange(len(calibration), len(origins)) if test is None else test
    archive = grid_archive(origins, calibration, test)
    lags = 3
    region = SPCI(lags=lags, window_size=12).calibrate(archive).predict(archive)
    query_origins = archive.split("test").origins.tolist()
    fitted = [u for u, o in enumerate(query_origins) if not o - lags * cadence <= absent < o]
    unbounded = [u for u in range(len(query_origins)) if u not in fitted]
    assert unbounded and len(calls) == len(fitted)
    assert torch.isneginf(region.lower[unbounded]).all()
    assert torch.isposinf(region.upper[unbounded]).all()
    for u, (features, targets, query) in zip(fitted, calls, strict=True):
        origin = query_origins[u]
        expected = torch.arange(origin - lags * cadence, origin, cadence, dtype=torch.float64)
        assert torch.equal(query, expected)
        pairs = torch.cat((features, targets[:, None]), dim=1)
        assert (pairs.diff(dim=1) == cadence).all()
        assert not (pairs == absent).any()


def test_spci_warmup_and_parameter_validation(monkeypatch):
    calls = spy_forest(monkeypatch)
    archive = archive_from_residuals(torch.arange(10), 3)
    region = SPCI(lags=3, window_size=8).calibrate(archive).predict(archive)
    assert torch.isposinf(region.upper[:2]).all()
    assert len(calls) == 5
    with pytest.raises(ValueError, match="lags"):
        SPCI(lags=0, window_size=8)
    with pytest.raises(ValueError, match="training pairs"):
        SPCI(lags=7, window_size=8)


def test_spci_missing_dependency_has_installation_hint(monkeypatch):
    import tscp.imports as imports

    real_import = imports.importlib.import_module

    def unavailable(name):
        if name == "sklearn_quantile":
            raise ImportError("unavailable")
        return real_import(name)

    monkeypatch.setattr(imports.importlib, "import_module", unavailable)
    archive = archive_from_residuals(torch.arange(12), 8)
    with pytest.raises(ImportError, match=r"tscp\[forest\]"):
        SPCI(lags=2, window_size=8).calibrate(archive).predict(archive)

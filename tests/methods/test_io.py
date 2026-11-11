import json

import numpy as np
import pytest
import torch

from tscp.datasets import variance_shift
from tscp.methods import SplitCP, get_method, load_method, save_method
from tscp.regions import Interval
from tscp.scores import AbsoluteScore, NormalizedScore
from tscp.testing import assert_regions_equal


def fitted(score=None):
    archive = variance_shift(horizon=2, num_origins=300, num_series=2)
    method = SplitCP(score=score if score is not None else NormalizedScore()).fit(archive)
    return method, archive


def test_a_loaded_method_predicts_what_the_saved_one_did(tmp_path):
    method, archive = fitted()
    expected = method.calibrate(archive).predict(archive)
    save_method(method, tmp_path / "method.npz")
    loaded = load_method(tmp_path / "method.npz")
    assert torch.equal(loaded.score.fitted_scale, method.score.fitted_scale)
    assert loaded.fit_fingerprint == archive.fingerprint() == method.fit_fingerprint
    assert_regions_equal(loaded.calibrate(archive).predict(archive), expected)


def test_files_are_never_overwritten(tmp_path):
    method, _ = fitted()
    save_method(method, tmp_path / "method.npz")
    with pytest.raises(FileExistsError):
        save_method(method, tmp_path / "method.npz")


def _rewrite(path, target, **changes):
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    arrays.update(changes)
    np.savez_compressed(target, **arrays)


def test_tampered_state_and_foreign_files_are_refused(tmp_path):
    method, _ = fitted()
    path = tmp_path / "method.npz"
    save_method(method, path)
    with np.load(path, allow_pickle=False) as data:
        doubled = data["state.score.fitted_scale"] * 2
        header = json.loads(str(data["header"].item()))
    _rewrite(path, tmp_path / "tampered.npz", **{"state.score.fitted_scale": doubled})
    with pytest.raises(ValueError, match="fingerprint"):
        load_method(tmp_path / "tampered.npz")
    header["schema_version"] = 2
    _rewrite(path, tmp_path / "v2.npz", header=np.asarray(json.dumps(header)))
    with pytest.raises(ValueError, match="schema version 2"):
        load_method(tmp_path / "v2.npz")
    np.savez(tmp_path / "other.npz", x=np.zeros(3))
    with pytest.raises(ValueError, match="no header"):
        load_method(tmp_path / "other.npz")


class RememberedScore(AbsoluteScore):
    """A score whose fitted state is a Python number, outside torch: it refits on load."""

    refit_on_load = True

    def __init__(self):
        super().__init__()
        self.width = None

    def fit(self, train, validation=None):
        self.width = float(train.residuals.abs().mean())
        return self

    def region(self, threshold, rows):
        return Interval(-threshold - self.width, threshold + self.width)


def test_components_with_state_outside_torch_refit_on_the_same_archive(tmp_path):
    method, archive = fitted(RememberedScore())
    expected = method.calibrate(archive).predict(archive)
    save_method(method, tmp_path / "method.npz")
    with pytest.raises(ValueError, match="pass the archive"):
        load_method(tmp_path / "method.npz")
    other = variance_shift(horizon=2, num_origins=300, num_series=2, seed=9)
    with pytest.raises(ValueError, match="not the one"):
        load_method(tmp_path / "method.npz", other)
    loaded = load_method(tmp_path / "method.npz", archive)
    assert_regions_equal(loaded.calibrate(archive).predict(archive), expected)


def test_untrained_methods_save_and_load_too(tmp_path):
    archive = variance_shift(horizon=1, num_origins=200)
    method = get_method("aci", gamma=0.02)
    save_method(method, tmp_path / "aci.npz")
    loaded = load_method(tmp_path / "aci.npz")
    expected = method.calibrate(archive).predict(archive)
    assert_regions_equal(loaded.calibrate(archive).predict(archive), expected)

import json

import numpy as np
import pytest
import torch

from tscp.data import Graph, ResidualArchive, load_archive, save_archive


def full_archive() -> ResidualArchive:
    residuals = torch.randn(12, 2, 3)
    observed = torch.ones(12, 2, 3, dtype=torch.bool)
    observed[4, 1, 2] = False
    residuals[4, 1, 2] = 0.0
    return ResidualArchive(
        residuals=residuals,
        origins=torch.arange(12) * 2,
        steps=torch.tensor([1, 2]),
        splits={"fit": torch.arange(0, 4), "calibration": torch.arange(5, 8), "test": [9, 10, 11]},
        forecasts=torch.randn(12, 2, 3),
        outputs={"scale": torch.rand(12, 2, 3)},
        context=torch.randn(12, 3, 2),
        context_names=("hour", "load"),
        observed=observed,
        graph=Graph(torch.tensor([[0, 1], [1, 2]]), num_nodes=3, edge_weight=torch.rand(2)),
        series=("a", "b", "c"),
        provenance={"dataset": "unit", "horizon": 2},
    )


def test_round_trip_keeps_everything(tmp_path):
    original = full_archive()
    path = tmp_path / "archive.npz"
    save_archive(original, path)
    loaded = load_archive(path)
    assert loaded.fingerprint() == original.fingerprint()
    for name, value in original.arrays().items():
        assert torch.equal(loaded.arrays()[name], value), name
    assert loaded.context_names == original.context_names
    assert loaded.series == original.series
    assert loaded.graph.num_nodes == 3
    assert loaded.provenance == original.provenance
    assert loaded.splits.sizes() == original.splits.sizes()


def test_existing_files_are_never_overwritten(tmp_path):
    path = tmp_path / "archive.npz"
    save_archive(full_archive(), path)
    with pytest.raises(FileExistsError):
        save_archive(full_archive(), path)


def _rewrite(path, target, **changes):
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    arrays.update(changes)
    np.savez_compressed(target, **arrays)


def test_tampered_arrays_fail_the_fingerprint_check(tmp_path):
    path = tmp_path / "archive.npz"
    save_archive(full_archive(), path)
    with np.load(path, allow_pickle=False) as data:
        doubled = data["residuals"] * 2  # keeps unobserved residuals at 0
    _rewrite(path, tmp_path / "tampered.npz", residuals=doubled)
    with pytest.raises(ValueError, match="fingerprint"):
        load_archive(tmp_path / "tampered.npz")


def test_other_schema_versions_are_refused(tmp_path):
    path = tmp_path / "archive.npz"
    save_archive(full_archive(), path)
    with np.load(path, allow_pickle=False) as data:
        header = json.loads(str(data["header"].item()))
    header["schema_version"] = 2
    _rewrite(path, tmp_path / "v2.npz", header=np.asarray(json.dumps(header)))
    with pytest.raises(ValueError, match="schema version 2"):
        load_archive(tmp_path / "v2.npz")


def test_foreign_files_are_refused(tmp_path):
    np.savez(tmp_path / "other.npz", residuals=np.zeros(3))
    with pytest.raises(ValueError, match="no header"):
        load_archive(tmp_path / "other.npz")
    path = tmp_path / "archive.npz"
    save_archive(full_archive(), path)
    _rewrite(path, tmp_path / "extra.npz", extra=np.zeros(3))
    with pytest.raises(ValueError, match="unexpected arrays"):
        load_archive(tmp_path / "extra.npz")


@pytest.mark.parametrize("provenance", [{"bad": object()}, {"bad": float("nan")}])
def test_provenance_must_be_json(tmp_path, provenance):
    original = full_archive()
    bad = ResidualArchive(
        residuals=original.residuals,
        origins=original.origins,
        steps=original.steps,
        observed=original.observed,
        provenance=provenance,
    )
    with pytest.raises(ValueError, match="JSON"):
        save_archive(bad, tmp_path / "bad.npz")

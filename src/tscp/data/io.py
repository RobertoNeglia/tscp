"""Self-contained ``.npz`` residual archives.

A file holds every array of a :class:`~tscp.data.ResidualArchive` plus a JSON header with
the format name, schema version, names, provenance and the archive fingerprint. Files are
written without pickle, are never overwritten, and are checked against their fingerprint
when loaded.
"""

import json
from pathlib import Path

import numpy as np
import torch

from tscp import __version__
from tscp.data.archive import Graph, ResidualArchive
from tscp.data.splits import SplitPlan

FORMAT = "tscp.residual_archive"
SCHEMA_VERSION = 1

_PLAIN = ("residuals", "origins", "steps", "forecasts", "context", "observed")
_PREFIXES = ("split.", "output.", "graph.")


def save_archive(archive: ResidualArchive, path: str | Path) -> None:
    """Write ``archive`` to a new ``.npz`` file.

    Raises:
        FileExistsError: If ``path`` exists; archives are never overwritten.
        ValueError: If the provenance is not JSON-serializable without NaN.
    """
    arrays = {name: value.detach().cpu().numpy() for name, value in archive.arrays().items()}
    header = {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "tscp_version": __version__,
        "fingerprint": archive.fingerprint(),
        "series": list(archive.series),
        "context_names": list(archive.context_names),
        "graph_num_nodes": None if archive.graph is None else archive.graph.num_nodes,
        "provenance": dict(archive.provenance),
    }
    try:
        text = json.dumps(header, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"provenance must be JSON-serializable without NaN: {error}") from error
    arrays["header"] = np.asarray(text)
    with Path(path).open("xb") as handle:
        np.savez_compressed(handle, **arrays)


def load_archive(path: str | Path) -> ResidualArchive:
    """Read an archive written by :func:`save_archive`, checking its fingerprint.

    Raises:
        ValueError: If the file is not a tscp archive, has another schema version, holds
            unexpected arrays, or its arrays do not match the fingerprint in its header.
    """
    with np.load(path, allow_pickle=False) as data:
        if "header" not in data.files:
            raise ValueError(f"{path} is not a tscp residual archive: it has no header")
        header = json.loads(str(data["header"].item()))
        if header.get("format") != FORMAT:
            raise ValueError(f"{path} is not a tscp residual archive")
        if header.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"{path} has schema version {header.get('schema_version')}; "
                f"this tscp reads version {SCHEMA_VERSION}"
            )
        names = [name for name in data.files if name != "header"]
        unexpected = [n for n in names if n not in _PLAIN and not n.startswith(_PREFIXES)]
        if unexpected:
            raise ValueError(f"{path} holds unexpected arrays: {unexpected}")
        arrays = {name: torch.from_numpy(data[name]) for name in names}

    splits = {n.removeprefix("split."): v for n, v in arrays.items() if n.startswith("split.")}
    outputs = {n.removeprefix("output."): v for n, v in arrays.items() if n.startswith("output.")}
    graph = None
    if "graph.edge_index" in arrays:
        graph = Graph(
            edge_index=arrays["graph.edge_index"],
            num_nodes=header["graph_num_nodes"],
            edge_weight=arrays.get("graph.edge_weight"),
        )
    archive = ResidualArchive(
        residuals=arrays["residuals"],
        origins=arrays["origins"],
        steps=arrays["steps"],
        splits=SplitPlan(splits) if splits else None,
        forecasts=arrays.get("forecasts"),
        outputs=outputs,
        context=arrays.get("context"),
        context_names=tuple(header["context_names"]),
        observed=arrays.get("observed"),
        graph=graph,
        series=tuple(header["series"]),
        provenance=header["provenance"],
    )
    if archive.fingerprint() != header["fingerprint"]:
        raise ValueError(f"{path}: the arrays do not match the fingerprint in the header")
    return archive


__all__ = ["FORMAT", "SCHEMA_VERSION", "load_archive", "save_archive"]

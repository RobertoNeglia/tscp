"""Fitted methods in self-contained ``.npz`` files.

A file holds a method's config and its fitted state, ``state_dict()``, with a JSON header
that records the format, the schema version, the tscp version, the config and the
fingerprints of the state and of the archive the method was fitted on. Files are written
without pickle and never overwritten; loading checks the state against its fingerprint.

Calibration state is not saved, because calibration is cheap: load, then ``calibrate``.
"""

import json
from pathlib import Path

import numpy as np
import torch

from tscp import __version__
from tscp.data import ResidualArchive
from tscp.utils import Component, array_fingerprint, from_config

FORMAT = "tscp.method"
SCHEMA_VERSION = 1
_PREFIX = "state."


def save_method(method: Component, path: str | Path) -> None:
    """Write a fitted method, or any fitted component, to a new ``.npz`` file.

    Raises:
        FileExistsError: If ``path`` exists; files are never overwritten.
        ValueError: If the config cannot be written as JSON.
    """
    state = {name: value.detach().cpu() for name, value in method.state_dict().items()}
    arrays = {_PREFIX + name: value.numpy() for name, value in state.items()}
    header = {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "tscp_version": __version__,
        "config": method.to_config(),
        "state_fingerprint": array_fingerprint(state),
        "fit_archive": getattr(method, "fit_fingerprint", None),
    }
    try:
        text = json.dumps(header, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"the method's config cannot be written as JSON: {error}") from error
    arrays["header"] = np.asarray(text)
    with Path(path).open("xb") as handle:
        np.savez_compressed(handle, **arrays)


def load_method(path: str | Path, archive: ResidualArchive | None = None) -> Component:
    """Read a method written by :func:`save_method`, fitted and ready to calibrate.

    The method is rebuilt from its config and its fitted state is loaded. A component
    whose fitted state lives outside torch sets ``refit_on_load``; it is fitted again on
    ``archive.split("fit")``, so ``archive`` must then be the archive the method was
    fitted on, which its fingerprint checks.

    Raises:
        ValueError: If the file is not a tscp method, has another schema version, holds
            state that does not match its fingerprint, or needs an archive to refit that is
            missing or is not the one the method was fitted on.
    """
    with np.load(path, allow_pickle=False) as data:
        if "header" not in data.files:
            raise ValueError(f"{path} is not a tscp method: it has no header")
        header = json.loads(str(data["header"].item()))
        if header.get("format") != FORMAT:
            raise ValueError(f"{path} is not a tscp method")
        if header.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"{path} has schema version {header.get('schema_version')}; "
                f"this tscp reads version {SCHEMA_VERSION}"
            )
        unexpected = [
            name for name in data.files if name != "header" and not name.startswith(_PREFIX)
        ]
        if unexpected:
            raise ValueError(f"{path} holds unexpected arrays: {unexpected}")
        state = {
            name.removeprefix(_PREFIX): torch.from_numpy(data[name])
            for name in data.files
            if name.startswith(_PREFIX)
        }
    if array_fingerprint(state) != header["state_fingerprint"]:
        raise ValueError(f"{path}: the fitted state does not match the fingerprint in the header")
    method = from_config(header["config"])
    method.load_state_dict(state)
    refit = [m for m in method.modules() if isinstance(m, Component) and m.refit_on_load]
    if refit:
        if archive is None:
            raise ValueError(
                "this method refits components on load: pass the archive it was fitted on"
            )
        if archive.fingerprint() != header["fit_archive"]:
            raise ValueError("the archive is not the one the method was fitted on")
        plan = archive.splits
        validation = archive.split("validation") if "validation" in plan else None
        for component in refit:
            component.fit(archive.split("fit"), validation)
    method.fit_fingerprint = header["fit_archive"]
    return method


__all__ = ["FORMAT", "SCHEMA_VERSION", "load_method", "save_method"]

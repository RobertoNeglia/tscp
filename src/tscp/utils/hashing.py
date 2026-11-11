"""Fingerprints of arrays and configs, for file checks and artifact IDs."""

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch


def _as_numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(value)


def array_fingerprint(arrays: Mapping[str, np.ndarray | torch.Tensor]) -> str:
    """SHA-256 over array names, dtypes, shapes and bytes, in sorted name order."""
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = _as_numpy(arrays[name])
        header = json.dumps([name, array.dtype.str, list(array.shape)])
        digest.update(header.encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def config_hash(config: Mapping[str, Any]) -> str:
    """SHA-256 of a config's canonical JSON (sorted keys, no NaN)."""
    text = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


__all__ = ["array_fingerprint", "config_hash"]

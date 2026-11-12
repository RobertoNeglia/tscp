"""Generate tiny scalar equation oracles without importing tscp, torch or NumPy.

Run from the repository root with ``python tests/reference/generate_basic.py``.
The fixture tests calibration and sequential updates conditional on fixed forecasts;
it does not reproduce forecaster training from the cited experiments.
"""

import json
import math
import platform
from pathlib import Path


def quantile(values, alpha, weights=None):
    if alpha <= 0:
        return math.inf
    if alpha >= 1:
        return -math.inf
    weights = [1.0] * len(values) if weights is None else weights
    mass, total = 0.0, sum(weights) + 1.0
    for value, weight in sorted(zip(values, weights, strict=True)):
        mass += weight
        if mass / total >= 1 - alpha - 1e-12:
            return value
    return math.inf


def encode(value):
    if isinstance(value, float) and math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if isinstance(value, list):
        return [encode(v) for v in value]
    return value


def main():
    calibration = [-6.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0]
    outcomes = [1.0, -9.0, 0.0, 10.0, -1.0]
    origins = list(range(len(calibration))) + list(range(10, 15))
    scale = [1.0, 0.5, 2.0, 1.0, 0.5, 1.5, 2.0, 1.0, 4.0] + [0.5, 1.0, 2.0, 1.5, 3.0]
    lower = [-2.0 - i % 3 for i in range(14)]
    upper = [2.0 + i % 4 for i in range(14)]
    alpha, decay, gamma = 0.25, 0.93, 0.2
    expected = {}
    q = quantile([abs(r) for r in calibration], alpha)
    expected["split"] = [[-q, q] for _ in outcomes]
    left = quantile([-r for r in calibration], alpha / 2)
    right = quantile(calibration, alpha / 2)
    expected["signed"] = [[-left, right] for _ in outcomes]
    q = quantile([abs(r) / s for r, s in zip(calibration, scale, strict=False)], alpha)
    expected["normalized"] = [[-q * s, q * s] for s in scale[9:]]
    fitted = math.sqrt(sum(r * r for r in calibration) / len(calibration))
    q = quantile([abs(r) / fitted for r in calibration], alpha)
    expected["normalized_fitted"] = [[-q * fitted, q * fitted] for _ in outcomes]
    q = quantile(
        [max(lo - r, r - u) for r, lo, u in zip(calibration, lower, upper, strict=False)], alpha
    )
    expected["cqr"] = [[lo - q, u + q] for lo, u in zip(lower[9:], upper[9:], strict=True)]
    left = quantile([lo - r for lo, r in zip(lower, calibration, strict=False)], alpha / 2)
    right = quantile([r - u for r, u in zip(calibration, upper, strict=False)], alpha / 2)
    expected["cqr_asymmetric"] = [
        [lo - left, u + right] for lo, u in zip(lower[9:], upper[9:], strict=True)
    ]
    expected["nexcp"], expected["aci"] = [], []
    pool = list(calibration)
    level = alpha
    for t, outcome in enumerate(outcomes):
        age_weights = [decay ** (origins[9 + t] - origin) for origin in origins[: 9 + t]]
        q = quantile([abs(r) for r in pool], alpha, age_weights)
        expected["nexcp"].append([-q, q])
        q = quantile([abs(r) for r in pool], level)
        expected["aci"].append([-q, q])
        level += gamma * (alpha - int(abs(outcome) > q))
        pool.append(outcome)
    window, period, missing = 6, 3, 1
    expected["windowed"], expected["windowed_missing"], expected["mondrian"] = [], [], []
    for t in range(len(outcomes)):
        rows = list(calibration) + outcomes[:t]
        q = quantile([abs(r) for r in rows[-window:]], alpha)
        expected["windowed"].append([-q, q])
        # A missing outcome keeps its slot in the window and leaves the quantile.
        known = [True] * len(calibration) + [i != missing for i in range(t)]
        recent = zip(rows[-window:], known[-window:], strict=True)
        q = quantile([abs(r) for r, ok in recent if ok], alpha)
        expected["windowed_missing"].append([-q, q])
        origin = origins[len(calibration) + t]
        same = zip(calibration, origins, strict=False)
        q = quantile([abs(r) for r, o in same if o % period == origin % period], alpha)
        expected["mondrian"].append([-q, q])
    fixture = {
        "provenance": {
            "kind": "independent scalar evaluation of published equations",
            "generator": "tests/reference/generate_basic.py",
            "dependencies": "Python standard library only; no random draws",
            "versions": {"python": platform.python_version()},
            "sources": {
                "split_signed_normalized": "https://arxiv.org/abs/1604.04173",
                "nexcp": "https://arxiv.org/abs/2202.13415v5",
                "aci": "https://arxiv.org/abs/2106.00170v3",
                "cqr": "https://arxiv.org/abs/1905.03222v1",
                "windowed": "https://arxiv.org/abs/2202.13415v5",
                "mondrian": "https://proceedings.mlr.press/v25/vovk12.html",
            },
            "conventions": "test-point mass 1 at infinity; left inverse weighted CDF; "
            "truth minus forecast; outcomes released before the next origin; "
            "alpha<=0 unbounded and alpha>=1 empty; windows hold the latest released rows, "
            "a missing row keeps its slot; Mondrian groups are origin mod period",
            "deviations": "fixed external forecasts; expanding calibration for ACI/NexCP; "
            "two-sided CQR is a union-bound extension",
            "tolerance": {"atol": 1e-12, "rtol": 1e-12},
        },
        "inputs": {
            "calibration": calibration,
            "test": outcomes,
            "origins": origins,
            "scales": scale,
            "lower": lower,
            "upper": upper,
        },
        "parameters": {
            "alpha": alpha,
            "decay": decay,
            "gamma": gamma,
            "window": window,
            "period": period,
            "missing_test_outcome": missing,
        },
        "expected": {name: encode(value) for name, value in expected.items()},
    }
    path = Path(__file__).parents[1] / "fixtures" / "references" / "basic.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fixture, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

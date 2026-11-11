"""Regenerate small online-controller references into a new JSON file.

PID outputs execute the pinned authors' controller with supplied scorecasts. Only
the three reviewed function definitions are loaded; dataset and forecasting imports
are not executed. DtACI outputs independently evaluate Algorithm 2's scalar equations.

Usage: python tests/fixtures/generate_online_reference.py NEW_PATH.json
"""

import ast
import hashlib
import json
import math
import sys
import urllib.request
from pathlib import Path

import numpy as np

PID_REVISION = "b729c3f5ff633bfc43f0f7ca08199b549c2573ac"
PID_URL = (
    "https://raw.githubusercontent.com/aangelopoulos/conformal-time-series/"
    f"{PID_REVISION}/core/methods.py"
)


class SuppliedScorecasts:
    columns = ("scorecasts",)

    def __init__(self, values):
        self.values = values

    def __getitem__(self, key):
        assert key == "scorecasts"
        return [(value, value) for value in self.values]


def pid_reference():
    source = urllib.request.urlopen(PID_URL, timeout=30).read()
    tree = ast.parse(source)
    names = {"mytan", "saturation_fn_log", "quantile_integrator_log_scorecaster"}
    functions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {"np": np, "tqdm": lambda values: values}
    # The pinned source uses the alias removed in NumPy 2.
    if not hasattr(np, "infty"):
        np.infty = np.inf
    exec(compile(ast.Module(body=functions, type_ignores=[]), PID_URL, "exec"), namespace)
    scores = [0.4, 0.1, 1.3, -0.2, 0.6, 2.1, 0.3, 0.8, -0.4, 0.0, 1.7, 0.2]
    casts = [0.0, 0.2, 0.3, 0.4, 0.1, 0.0, 0.7, 0.5, 0.2, 0.1, 0.4, 0.2]
    cases = []
    for proportional in (False, True):
        for scorecast in (False, True):
            settings = {"alpha": 0.2, "lr": 0.3, "Csat": 2.0, "KI": 0.7, "T_burnin": 4}
            result = namespace["quantile_integrator_log_scorecaster"](
                scores=np.array(scores),
                data=SuppliedScorecasts(casts),
                upper=True,
                ahead=1,
                proportional_lr=proportional,
                scorecast=scorecast,
                **settings,
            )
            cases.append(
                {
                    "settings": settings,
                    "proportional_lr": proportional,
                    "scorecast": scorecast,
                    "thresholds": result["q"].tolist(),
                }
            )
    return {
        "provenance": {
            "url": PID_URL,
            "revision": PID_REVISION,
            "sha256": hashlib.sha256(source).hexdigest(),
            "kind": "pinned upstream execution with supplied scorecasts",
            "generator": "tests/fixtures/generate_online_reference.py",
            "versions": {"python": sys.version.split()[0], "numpy": np.__version__},
            "tolerance": {"atol": 1e-12, "rtol": 0},
            "conventions": "zero-initialized tracker and lagged integrator; supplied scorecasts",
            "deviations": [
                "cached scorecasts replace the score-forecasting stage",
                "only the three controller function definitions are executed",
                "np.infty is restored as an alias for np.inf when running under NumPy 2",
            ],
        },
        "scores": scores,
        "scorecasts": casts,
        "cases": cases,
    }


def dtaci_reference():
    betas = [0.7, 0.03, 0.5, 0.01, 0.4, 0.08, 0.9, 0.02, 0.6, 0.04, 0.2, 0.001]
    alpha, eta, sigma, gammas = 0.1, 2.72, 0.001, [0.01, 0.05, 0.2]
    levels = [alpha] * len(gammas)
    probabilities = [1 / len(gammas)] * len(gammas)
    combined, trajectory, weight_trajectory = [], [], []
    for beta in betas:
        combined.append(sum(p * level for p, level in zip(probabilities, levels, strict=True)))
        trajectory.append(levels[:])
        weight_trajectory.append(probabilities[:])
        loss = [alpha * (beta - level) - min(beta - level, 0) for level in levels]
        weights = [p * math.exp(-eta * value) for p, value in zip(probabilities, loss, strict=True)]
        total = sum(weights)
        probabilities = [(1 - sigma) * w / total + sigma / len(gammas) for w in weights]
        levels = [
            level + gamma * (alpha - (level > beta))
            for level, gamma in zip(levels, gammas, strict=True)
        ]
    return {
        "provenance": {
            "url": "https://jmlr.org/papers/v25/22-1218.html",
            "algorithm": 2,
            "kind": "independent scalar equations; continuous level convention",
            "generator": "tests/fixtures/generate_online_reference.py",
            "versions": {"python": sys.version.split()[0]},
            "dependencies": "Python standard-library math; no random draws",
            "tolerance": {
                "expert_state": {"atol": 1e-14, "rtol": 1e-13},
                "combined_level": {"atol": 1e-13, "rtol": 0},
            },
            "conventions": "deterministic Algorithm 2; threshold(alpha) = 1 - alpha",
            "deviations": [
                "independent scalar evaluation rather than upstream R execution",
                "the continuous test distribution has no empirical quantile ties",
            ],
        },
        "betas": betas,
        "alpha": alpha,
        "gammas": gammas,
        "eta": eta,
        "sigma": sigma,
        "combined_levels": combined,
        "expert_levels": trajectory,
        "probabilities": weight_trajectory,
    }


if __name__ == "__main__":
    payload = {"pid": pid_reference(), "dtaci": dtaci_reference()}
    with Path(sys.argv[1]).open("x") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")

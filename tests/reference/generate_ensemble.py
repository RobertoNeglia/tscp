"""Regenerate small fixtures by running pinned author routines on cached predictions.

Run explicitly with ``--method enbpi`` or ``--method spci`` and a new output path.
This script needs network access; ordinary tests read the committed fixtures only.
SPCI also needs the optional forest dependency. No forecaster is trained.
"""

import argparse
import ast
import contextlib
import hashlib
import importlib.metadata
import io
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.request import urlopen

import numpy as np

REFERENCES = {
    "enbpi": ("EnbPI", "5df45367b0b13c841c44f173e5a791e070e0c511"),
    "spci": ("SPCI-code", "d598dd32099402bc1b6d636f471a155001c0e54e"),
}


def source(method, filename, metadata):
    repo, revision = REFERENCES[method]
    url = f"https://raw.githubusercontent.com/hamrel-cxu/{repo}/{revision}/{filename}"
    with urlopen(url, timeout=30) as response:
        raw = response.read()
    metadata["sources"].append({"url": url, "sha256": hashlib.sha256(raw).hexdigest()})
    return ast.parse(raw.decode())


def functions(tree, names, namespace):
    selected = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in selected} != set(names):
        raise ValueError("the pinned source does not contain the expected routines")
    exec(compile(ast.Module(body=selected, type_ignores=[]), "<reference>", "exec"), namespace)


def reference_class(tree, name, names, namespace):
    definition = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name
    )
    functions(definition, names, namespace)
    return type("Reference", (), {key: namespace[key] for key in names})


def metadata_for(method):
    repo, revision = REFERENCES[method]
    return {
        "reference": f"https://github.com/hamrel-cxu/{repo}/tree/{revision}",
        "revision": revision,
        "sources": [],
        "generated_by": "pinned author routines with cached inputs",
        "versions": {"numpy": np.__version__},
    }


def generate_enbpi():
    metadata = metadata_for("enbpi")
    namespace = {
        "np": np,
        "math": math,
        "time": time,
        "pd": SimpleNamespace(DataFrame=lambda values, **_: values),
    }
    functions(source("enbpi", "utils_EnbPI.py", metadata), ["binning", "strided_app"], namespace)
    generator = np.random.default_rng(41)
    calibration_size, test_size, members = 20, 10, 6
    bootstrap = np.stack(
        [
            generator.choice(
                np.arange(calibration_size)[np.arange(calibration_size) % members != b],
                size=calibration_size,
            )
            for b in range(members)
        ]
    )
    in_bag = np.zeros((members, calibration_size), dtype=bool)
    for b in range(members):
        in_bag[b, bootstrap[b]] = True
    time_index = np.arange(calibration_size + test_size, dtype=float)
    center = 0.2 * time_index + np.sin(time_index / 4)
    truth = center + generator.exponential(size=len(center)) - 0.7
    predictions = np.stack(
        [center + 0.15 * b + 0.2 * np.cos(time_index / (b + 2)) for b in range(members)]
    )
    namespace["util"] = SimpleNamespace(
        binning=namespace["binning"],
        strided_app=namespace["strided_app"],
        generate_bootstrap_samples=lambda *_: bootstrap,
    )
    reference = reference_class(
        source("enbpi", "PI_class_EnbPI.py", metadata),
        "prediction_interval",
        ["__init__", "fit_bootstrap_models_online", "compute_PIs_Ensemble_online"],
        namespace,
    )

    class CachedRegressor:
        def __init__(self):
            self.member = -1

        def fit(self, *_):
            self.member += 1
            return self

        def predict(self, _):
            return predictions[self.member]

    model = reference(
        CachedRegressor(),
        time_index[:calibration_size, None],
        time_index[calibration_size:, None],
        truth[:calibration_size],
        truth[calibration_size:],
    )
    alpha = 0.13
    with contextlib.redirect_stdout(io.StringIO()):
        model.fit_bootstrap_models_online(members, [])
        model.compute_PIs_Ensemble_online(alpha, stride=1, smallT=False)
    metadata.update(
        {
            "parameters": {"alpha": alpha, "calibration_size": calibration_size, "bins": 5},
            "conventions": (
                "journal OOB mean, signed residuals, ceil-percent linear quantiles, first tie"
            ),
            "tolerance": {"rtol": 1e-12, "atol": 1e-12},
            "deviations": ["cached member predictions replace fitting the forecasters"],
        }
    )
    return {
        "truth": truth,
        "member_forecasts": predictions.T,
        "in_bag": in_bag,
        "forecasts": np.concatenate(
            [model.Ensemble_train_interval_centers, model.Ensemble_pred_interval_centers]
        ),
        "residuals": model.Ensemble_online_resid,
        "target_bounds": model.Ensemble_pred_interval_ends,
        "beta": np.asarray(model.beta_hat_bins),
    }, metadata


def generate_spci():
    from sklearn_quantile import RandomForestQuantileRegressor

    class ScalarQueryForest(RandomForestQuantileRegressor):
        def predict(self, features):
            values = super().predict(features)
            # Recent NumPy rejects the old implicit conversion of a length-one array
            # to a scalar when the reference stores each interval endpoint.
            return values.reshape(-1) if len(features) == 1 else values

    metadata = metadata_for("spci")
    for package in ("sklearn-quantile", "scikit-learn", "scipy"):
        metadata["versions"][package] = importlib.metadata.version(package)
    seed, calibration_size, test_size, lags, alpha = 7, 24, 8, 4, 0.13
    namespace = {
        "np": np,
        "math": math,
        "time": time,
        "pd": SimpleNamespace(DataFrame=lambda values, **_: values),
        "sliding_window_view": np.lib.stride_tricks.sliding_window_view,
        "RandomForestQuantileRegressor": lambda **kwargs: ScalarQueryForest(
            **{**kwargs, "random_state": seed, "n_jobs": 1}
        ),
    }
    functions(
        source("spci", "utils_SPCI.py", metadata),
        ["binning_use_RF_quantile_regr", "strided_app"],
        namespace,
    )
    namespace["utils"] = SimpleNamespace(
        **{key: namespace[key] for key in ("binning_use_RF_quantile_regr", "strided_app")}
    )
    reference = reference_class(
        source("spci", "SPCI_class.py", metadata),
        "SPCI_and_EnbPI",
        ["compute_PIs_Ensemble_online", "multi_step_QRF", "train_QRF"],
        namespace,
    )
    generator = np.random.default_rng(43)
    residuals = generator.exponential(size=calibration_size + test_size) - 0.8
    for i in range(1, len(residuals)):
        residuals[i] += 0.6 * residuals[i - 1]
    model = reference()
    model.X_train = np.zeros((calibration_size, 1))
    model.Ensemble_online_resid = residuals
    model.Ensemble_pred_interval_centers = np.zeros(test_size)
    model.Ensemble_pred_interval_sigma = np.ones(test_size)
    model.n_estimators, model.max_d = 10, 2
    model.criterion, model.bins = "squared_error", 5
    model.T1, model.weigh_residuals = None, False
    with contextlib.redirect_stdout(io.StringIO()):
        model.compute_PIs_Ensemble_online(
            alpha, stride=1, smallT=False, past_window=lags, use_SPCI=True
        )
    metadata.update(
        {
            "parameters": {
                "alpha": alpha,
                "lags": lags,
                "window_size": calibration_size,
                "n_estimators": 10,
                "max_depth": 2,
                "seed": seed,
                "n_jobs": 1,
            },
            "conventions": "rolling one-step QRF refit, five unrounded beta candidates, first tie",
            "tolerance": {"rtol": 1e-6, "atol": 1e-7},
            "deviations": [
                "forecasts are fixed at zero to isolate the interval algorithm",
                "fixed random_state and n_jobs are injected into the reference forest constructor",
                "criterion squared_error is the current spelling of the reference mse alias",
                "single-query forest output is squeezed for NumPy scalar-assignment compatibility",
            ],
        }
    )
    return {"residuals": residuals, "bounds": model.PIs_Ensemble}, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=REFERENCES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arrays, metadata = generate_enbpi() if args.method == "enbpi" else generate_spci()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")

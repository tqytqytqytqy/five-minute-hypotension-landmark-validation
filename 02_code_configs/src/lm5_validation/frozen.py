"""Immutable model-bundle creation, loading and prediction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .statistics import RidgeLogisticResult


def fit_imputation(frame: pd.DataFrame, features: Sequence[str]) -> dict[str, float]:
    medians: dict[str, float] = {}
    for feature in features:
        values = pd.to_numeric(frame[feature], errors="coerce")
        if values.notna().sum() == 0:
            raise ValueError(f"structurally missing training feature: {feature}")
        medians[feature] = float(values.median())
    return medians


def apply_imputation(
    frame: pd.DataFrame,
    features: Sequence[str],
    medians: Mapping[str, float],
    *,
    enforce_structural_gate: bool = True,
) -> np.ndarray:
    missing_columns = [feature for feature in features if feature not in frame]
    if missing_columns:
        raise ValueError(f"missing feature columns: {missing_columns}")
    if enforce_structural_gate:
        structural = [
            feature
            for feature in features
            if pd.to_numeric(frame[feature], errors="coerce").notna().sum() == 0
        ]
        if structural:
            raise ValueError(
                "structurally missing target-dataset features cannot be imputed: "
                + ", ".join(structural)
            )
    columns = []
    for feature in features:
        if feature not in medians:
            raise ValueError(f"frozen median missing for feature: {feature}")
        values = pd.to_numeric(frame[feature], errors="coerce").fillna(float(medians[feature]))
        columns.append(values.to_numpy(dtype=float))
    X = np.column_stack(columns)
    if not np.isfinite(X).all():
        raise ValueError("imputed design contains non-finite values")
    return X


def model_to_dict(model: RidgeLogisticResult, feature_names: Sequence[str]) -> dict:
    return {
        "family": "binomial",
        "link": "logit",
        "penalty": "ridge",
        "penalize_intercept": False,
        "l2": float(model.l2),
        "intercept_raw_scale": float(model.intercept_),
        "coefficients_raw_scale": {
            name: float(value) for name, value in zip(feature_names, model.coef_)
        },
        "standardized_intercept": float(model.standardized_intercept_),
        "standardized_coefficients": {
            name: float(value)
            for name, value in zip(feature_names, model.standardized_coef_)
        },
        "feature_mean": {
            name: float(value) for name, value in zip(feature_names, model.feature_mean_)
        },
        "feature_scale": {
            name: float(value) for name, value in zip(feature_names, model.feature_scale_)
        },
        "converged": bool(model.converged_),
        "iterations": int(model.n_iter_),
        "objective": float(model.objective_),
        "gradient_norm": float(model.gradient_norm_),
        "solver": model.solver_,
        "message": model.message_,
        "feature_order": list(feature_names),
    }


def predict_model_dict(model: Mapping, X: np.ndarray) -> np.ndarray:
    feature_order = list(model["feature_order"])
    coefficients = np.array(
        [model["coefficients_raw_scale"][name] for name in feature_order], dtype=float
    )
    if X.ndim != 2 or X.shape[1] != len(feature_order):
        raise ValueError("design width does not match frozen feature order")
    linear = float(model["intercept_raw_scale"]) + X @ coefficients
    probability = np.empty_like(linear, dtype=float)
    nonnegative = linear >= 0
    probability[nonnegative] = 1.0 / (1.0 + np.exp(-linear[nonnegative]))
    exp_linear = np.exp(linear[~nonnegative])
    probability[~nonnegative] = exp_linear / (1.0 + exp_linear)
    return probability


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(root: Path, exclude_name: str = "SHA256SUMS.csv") -> pd.DataFrame:
    rows = []
    excluded = root / exclude_name
    for path in sorted(
        p for p in root.rglob("*") if p.is_file() and p.resolve() != excluded.resolve()
    ):
        rows.append(
            {
                "relative_path": str(path.relative_to(root)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return pd.DataFrame(rows)


def load_bundle(bundle_dir: Path) -> tuple[dict, dict, dict]:
    model = json.loads((bundle_dir / "model.json").read_text(encoding="utf-8"))
    preprocess = json.loads((bundle_dir / "preprocess.json").read_text(encoding="utf-8"))
    thresholds = json.loads((bundle_dir / "thresholds.json").read_text(encoding="utf-8"))
    return model, preprocess, thresholds


def validate_bundle_contract(
    models: Mapping,
    preprocess: Mapping,
    thresholds: Mapping,
    *,
    expected_study_id: str | None = None,
) -> None:
    study_ids = {
        models.get("study_id"),
        preprocess.get("study_id"),
        thresholds.get("study_id"),
    }
    if None in study_ids or len(study_ids) != 1:
        raise ValueError(f"bundle study_id values are absent or inconsistent: {study_ids}")
    if expected_study_id is not None and study_ids != {expected_study_id}:
        raise ValueError("bundle study_id differs from the locked analysis study_id")
    primary = models.get("primary_model")
    model_map = models.get("models")
    preprocess_map = preprocess.get("models")
    if not isinstance(model_map, Mapping) or not model_map or primary not in model_map:
        raise ValueError("bundle models or primary_model are invalid")
    if not isinstance(preprocess_map, Mapping) or set(preprocess_map) != set(model_map):
        raise ValueError("preprocess model names differ from model.json")
    for name, model in model_map.items():
        order = list(model.get("feature_order", []))
        if not order or len(order) != len(set(order)):
            raise ValueError(f"{name} has an empty or duplicate feature order")
        coefficients = model.get("coefficients_raw_scale", {})
        if set(coefficients) != set(order):
            raise ValueError(f"{name} coefficient names differ from feature order")
        numeric_model_values = [model.get("intercept_raw_scale")]
        numeric_model_values += [coefficients[feature] for feature in order]
        if not np.isfinite(np.asarray(numeric_model_values, dtype=float)).all():
            raise ValueError(f"{name} has non-finite intercept or coefficients")
        pp = preprocess_map[name]
        if list(pp.get("feature_order", [])) != order:
            raise ValueError(f"{name} preprocess feature order differs from model")
        for key in ["imputation_medians", "standardization_means", "standardization_sds"]:
            values = pp.get(key, {})
            if set(values) != set(order):
                raise ValueError(f"{name} {key} names differ from feature order")
            array = np.asarray([values[feature] for feature in order], dtype=float)
            if not np.isfinite(array).all():
                raise ValueError(f"{name} {key} contains non-finite values")
            if key == "standardization_sds" and np.any(array <= 0):
                raise ValueError(f"{name} standardization SD is not positive")


def verify_test_vectors(bundle_dir: Path, tolerance: float = 1e-12) -> dict:
    models, preprocess, _ = load_bundle(bundle_dir)
    vectors = pd.read_csv(bundle_dir / "test_vectors.csv")
    if vectors.empty:
        raise AssertionError("test_vectors.csv is empty")
    results = {}
    for model_name, model in models["models"].items():
        features = model["feature_order"]
        medians = preprocess["models"][model_name]["imputation_medians"]
        expected_column = f"expected_probability_{model_name}"
        missing = [column for column in features + [expected_column] if column not in vectors]
        if missing:
            raise AssertionError(f"test vectors missing columns for {model_name}: {missing}")
        X = apply_imputation(vectors, features, medians, enforce_structural_gate=False)
        predicted = predict_model_dict(model, X)
        expected = pd.to_numeric(
            vectors[expected_column], errors="raise"
        ).to_numpy(float)
        if not np.isfinite(predicted).all() or not np.isfinite(expected).all():
            raise AssertionError(f"test vectors contain non-finite probabilities for {model_name}")
        if np.any((predicted < 0) | (predicted > 1) | (expected < 0) | (expected > 1)):
            raise AssertionError(f"test-vector probabilities are outside [0,1] for {model_name}")
        maximum = float(np.max(np.abs(predicted - expected)))
        results[model_name] = maximum
        if not np.isfinite(maximum) or maximum > tolerance:
            raise AssertionError(
                f"test-vector mismatch for {model_name}: {maximum} > {tolerance}"
            )
    return {"tolerance": tolerance, "maximum_absolute_error_by_model": results}


__all__ = [
    "apply_imputation",
    "fit_imputation",
    "load_bundle",
    "model_to_dict",
    "predict_model_dict",
    "sha256_file",
    "sha256_tree",
    "verify_test_vectors",
    "validate_bundle_contract",
]

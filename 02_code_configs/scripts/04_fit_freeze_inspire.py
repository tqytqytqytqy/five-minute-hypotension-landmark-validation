#!/usr/bin/env python3
"""Develop, internally evaluate, refit and freeze LM5-common-18 in INSPIRE."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_code_configs/src"))

from lm5_validation.frozen import (  # noqa: E402
    apply_imputation,
    fit_imputation,
    model_to_dict,
    predict_model_dict,
    sha256_file,
    sha256_tree,
    verify_test_vectors,
)
from lm5_validation.observation import FEATURE_COLUMNS  # noqa: E402
from lm5_validation.statistics import (  # noqa: E402
    auprc,
    auroc,
    binary_log_loss,
    brier_score,
    calibration_in_the_large,
    calibration_intercept_slope,
    evaluate_binary_predictions,
    fit_ridge_logistic,
    fit_ridge_logistic_cv,
    ici_equal_frequency,
    oe_ratio_log_ci,
    paired_patient_bootstrap,
    scaled_brier_score,
)


CONFIG = json.loads((ROOT / "02_code_configs/configs/analysis.json").read_text(encoding="utf-8"))
SEED = int(CONFIG["random_seed"])
MODEL_DIR = ROOT / "04_frozen_INSPIRE_LM5_model"
INSPIRE_DIR = ROOT / "03_derived_cohorts/INSPIRE"
TABLE_DIR = ROOT / "06_tables"
QA_DIR = ROOT / "09_QA_reproducibility/reports"


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def json_dump(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def stable_hash(value: object, kind: str) -> str:
    return hashlib.sha256(
        f"{CONFIG['study_id']}|{kind}|{value}".encode("utf-8")
    ).hexdigest()


def patient_stratified_split(frame: pd.DataFrame, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    if frame["patient_id"].duplicated().any():
        raise ValueError("primary INSPIRE main cohort must have one row per patient")
    rng = np.random.default_rng(SEED)
    development: list[int] = []
    test: list[int] = []
    y_numeric = pd.to_numeric(frame["primary_outcome"], errors="raise")
    if y_numeric.isna().any() or not set(y_numeric.unique()).issubset({0.0, 1.0}):
        raise ValueError("INSPIRE primary_outcome must be strictly binary before splitting")
    y = y_numeric.astype(int)
    for value in [0, 1]:
        indices = np.flatnonzero(y.to_numpy() == value)
        rng.shuffle(indices)
        cut = int(math.floor(len(indices) * fraction))
        development.extend(indices[:cut].tolist())
        test.extend(indices[cut:].tolist())
    return np.array(sorted(development)), np.array(sorted(test))


def performance_row(name: str, y: np.ndarray, p: np.ndarray, scope: str) -> tuple[dict, pd.DataFrame]:
    base = evaluate_binary_predictions(y, p).to_dict()
    citl = calibration_in_the_large(y, p)
    slope = calibration_intercept_slope(y, p)
    oe = oe_ratio_log_ci(y, p)
    ici = ici_equal_frequency(y, p, n_bins=10)
    row = {
        "scope": scope,
        "model": name,
        "n": len(y),
        "events": int(y.sum()),
        "event_rate": float(y.mean()),
        **base,
        "calibration_in_the_large": citl.estimate,
        "calibration_slope": slope.slope,
        "calibration_intercept_joint": slope.intercept,
        "observed_expected_ratio": oe.ratio,
        "ici_equal_frequency": ici.ici,
    }
    bins = ici.bins.copy()
    bins.insert(0, "model", name)
    bins.insert(0, "scope", scope)
    return row, bins


def bootstrap_metric_table(
    y: np.ndarray,
    predictions: dict[str, np.ndarray],
    patient_ids: np.ndarray,
    n_boot: int = 2000,
) -> pd.DataFrame:
    metrics = {
        "auroc": auroc,
        "auprc": auprc,
        "brier": brier_score,
        "scaled_brier": scaled_brier_score,
        "log_loss": binary_log_loss,
        "calibration_slope": lambda yy, pp: calibration_intercept_slope(yy, pp).slope,
        "calibration_in_the_large": lambda yy, pp: calibration_in_the_large(yy, pp).estimate,
        "observed_expected_ratio": lambda yy, pp: oe_ratio_log_ci(yy, pp).ratio,
    }
    comparisons = [("LM5_common18", name) for name in predictions if name != "LM5_common18"]
    parts = []
    for metric_name, metric in metrics.items():
        result = paired_patient_bootstrap(
            y,
            predictions,
            patient_ids,
            metric,
            n_boot=n_boot,
            random_state=SEED,
            comparisons=comparisons,
            return_replicates=False,
            min_valid_fraction=0.75,
        )
        table = result.summary.copy()
        table.insert(0, "metric", metric_name)
        parts.append(table)
    return pd.concat(parts, ignore_index=True)


def solve_intercept_shift(lp: np.ndarray, target_rate: float) -> float:
    lo, hi = -30.0, 30.0
    for _ in range(200):
        mid = (lo + hi) / 2
        mean = float(np.mean(1.0 / (1.0 + np.exp(-(lp + mid)))))
        if mean > target_rate:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def precision_plan(probability: np.ndarray, assumed_rate: float = 0.12) -> pd.DataFrame:
    p = np.clip(probability, 1e-12, 1 - 1e-12)
    lp = np.log(p / (1 - p))
    shift = solve_intercept_shift(lp, assumed_rate)
    p_ext = 1.0 / (1.0 + np.exp(-(lp + shift)))
    design = np.column_stack([np.ones(len(lp)), lp])
    info_per_subject = (design.T @ (design * (p_ext * (1 - p_ext))[:, None])) / len(lp)
    inverse = np.linalg.pinv(info_per_subject)
    slope_n = math.ceil((2 * 1.96 * math.sqrt(inverse[1, 1]) / 0.20) ** 2)

    events_needed = math.ceil(
        max((1.96 / math.log(1.10)) ** 2, (1.96 / abs(math.log(0.90))) ** 2)
    )
    oe_n = math.ceil(events_needed / assumed_rate)

    # AUC=0.70 was prespecified as a conservative planning value; no MOVER
    # model predictions or performance enter this precision calculation.
    auc_plan = 0.70
    auc_n = None
    for n in range(200, 100_001):
        n1 = max(2, int(round(n * assumed_rate)))
        n0 = n - n1
        q1 = auc_plan / (2 - auc_plan)
        q2 = 2 * auc_plan**2 / (1 + auc_plan)
        variance = (
            auc_plan * (1 - auc_plan)
            + (n1 - 1) * (q1 - auc_plan**2)
            + (n0 - 1) * (q2 - auc_plan**2)
        ) / (n1 * n0)
        if 2 * 1.96 * math.sqrt(variance) <= 0.05:
            auc_n = n
            break
    required = max(slope_n, oe_n, int(auc_n or 100_000))
    alert = p_ext >= float(CONFIG["thresholds"]["clinical_action_primary"])
    odds = float(CONFIG["thresholds"]["clinical_action_primary"]) / (
        1 - float(CONFIG["thresholds"]["clinical_action_primary"])
    )
    expected_contribution = alert * (p_ext - (1 - p_ext) * odds)
    conditional_variance = alert * (1 + odds) ** 2 * p_ext * (1 - p_ext)
    total_variance = float(np.mean(conditional_variance) + np.var(expected_contribution, ddof=1))
    nb_width = 2 * 1.96 * math.sqrt(total_variance / required)
    return pd.DataFrame(
        [
            {"precision_target": "calibration_slope_CI_total_width_le_0.20", "required_n": slope_n},
            {"precision_target": "O_over_E_CI_approximately_0.90_to_1.10", "required_n": oe_n},
            {"precision_target": "AUROC_CI_total_width_le_0.05_at_AUC_0.70", "required_n": auc_n},
            {"precision_target": "maximum_required_n", "required_n": required},
            {
                "precision_target": "expected_primary_threshold_net_benefit_CI_width_at_maximum_n",
                "required_n": required,
                "estimated_width": nb_width,
            },
        ]
    ).assign(assumed_external_event_rate=assumed_rate)


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    QA_DIR.mkdir(parents=True, exist_ok=True)
    data_path = INSPIRE_DIR / "inspire_main_stage2.csv.gz"
    if not data_path.exists():
        raise FileNotFoundError("Run 03_build_cohort.py --dataset INSPIRE first")
    data = pd.read_csv(data_path, low_memory=False)
    features = list(FEATURE_COLUMNS)
    if features != CONFIG["features_in_order"]:
        raise AssertionError("feature order differs between code and locked analysis.json")
    if data[["patient_id", "case_id"]].isna().any().any():
        raise AssertionError("primary model cohort has missing patient_id or case_id")
    if data["patient_id"].duplicated().any() or data["case_id"].duplicated().any():
        raise AssertionError("primary model cohort is not one unique operation per patient")
    y_numeric = pd.to_numeric(data["primary_outcome"], errors="raise")
    if y_numeric.isna().any() or not set(y_numeric.unique()).issubset({0.0, 1.0}):
        raise AssertionError("INSPIRE primary_outcome is not strictly binary")
    y = y_numeric.astype(int).to_numpy()
    development, test = patient_stratified_split(data, CONFIG["model"]["development_fraction"])
    log(f"INSPIRE model cohort n={len(data):,}; development={len(development):,}; test={len(test):,}")

    primary_medians = fit_imputation(data.iloc[development], features)
    X_dev = apply_imputation(data.iloc[development], features, primary_medians)
    X_test = apply_imputation(data.iloc[test], features, primary_medians)
    cv = fit_ridge_logistic_cv(
        X_dev,
        y[development],
        data.iloc[development]["patient_id"].to_numpy(),
        CONFIG["model"]["lambda_grid"],
        n_splits=CONFIG["model"]["cv_folds"],
        scoring="log_loss",
        max_iter=200,
        tol=1e-9,
        random_state=SEED,
    )
    if not cv.model.converged_:
        raise RuntimeError(f"primary development model did not converge: {cv.model.message_}")
    log(f"Selected ridge lambda={cv.best_l2:g}")

    model_specs = {
        "LM5_common18": features,
        "simple_recovered_by_5min": ["recovered_by_5min"],
        "simple_early_mean_map": ["early_mean_map_0_5"],
        "simple_t0_map": ["t0_map"],
    }
    development_models = {"LM5_common18": cv.model}
    development_medians = {"LM5_common18": primary_medians}
    test_predictions = {
        "LM5_common18": cv.model.predict_proba(X_test),
    }
    for name, cols in model_specs.items():
        if name == "LM5_common18":
            continue
        medians = fit_imputation(data.iloc[development], cols)
        X = apply_imputation(data.iloc[development], cols, medians)
        model = fit_ridge_logistic(X, y[development], l2=0.001, max_iter=200, tol=1e-9)
        if not model.converged_:
            raise RuntimeError(f"{name} development model did not converge")
        development_models[name] = model
        development_medians[name] = medians
        test_predictions[name] = model.predict_proba(
            apply_imputation(data.iloc[test], cols, medians)
        )

    internal_rows = []
    calibration_bins = []
    for name, probability in test_predictions.items():
        row, bins = performance_row(name, y[test], probability, "INSPIRE_patient_holdout")
        internal_rows.append(row)
        calibration_bins.append(bins)
    pd.DataFrame(internal_rows).to_csv(
        TABLE_DIR / "INSPIRE_internal_holdout_performance.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(calibration_bins, ignore_index=True).to_csv(
        TABLE_DIR / "INSPIRE_internal_holdout_calibration_bins.csv", index=False, encoding="utf-8-sig"
    )
    bootstrap = bootstrap_metric_table(
        y[test],
        test_predictions,
        data.iloc[test]["patient_id"].to_numpy(),
        n_boot=2000,
    )
    bootstrap.to_csv(
        TABLE_DIR / "INSPIRE_internal_holdout_bootstrap_2000.csv", index=False, encoding="utf-8-sig"
    )
    cv.cv_results.to_csv(MODEL_DIR / "ridge_cv_summary.csv", index=False, encoding="utf-8-sig")
    cv.fold_results.to_csv(MODEL_DIR / "ridge_cv_folds.csv", index=False, encoding="utf-8-sig")

    # Capacity thresholds are determined only from INSPIRE holdout prediction distribution.
    thresholds = {}
    for fraction in CONFIG["thresholds"]["capacity_top_fractions_from_inspire_patient_holdout"]:
        value = float(np.quantile(test_predictions["LM5_common18"], 1 - fraction, method="higher"))
        thresholds[f"capacity_top_{int(round(fraction * 100))}_percent"] = {
            "risk_threshold": value,
            "source": "INSPIRE_patient_level_holdout_LM5_common18_prediction_distribution",
            "clinical_action_threshold": False,
        }

    # Final refit uses all INSPIRE stage-2 cases and the selected development lambda.
    final_models = {}
    final_preprocess = {}
    final_probabilities = {}
    for name, cols in model_specs.items():
        medians = fit_imputation(data, cols)
        X_all = apply_imputation(data, cols, medians)
        l2 = cv.best_l2 if name == "LM5_common18" else 0.001
        model = fit_ridge_logistic(X_all, y, l2=l2, max_iter=300, tol=1e-10)
        if not model.converged_:
            raise RuntimeError(f"final {name} model did not converge: {model.message_}")
        final_models[name] = model_to_dict(model, cols)
        final_preprocess[name] = {
            "feature_order": cols,
            "imputation_medians": medians,
            "standardization_means": {
                col: float(value) for col, value in zip(cols, model.feature_mean_)
            },
            "standardization_sds": {
                col: float(value) for col, value in zip(cols, model.feature_scale_)
            },
            "parameters_estimated_from": "all_INSPIRE_final_stage2_refit",
            "target_database_reestimation_forbidden": True,
        }
        final_probabilities[name] = model.predict_proba(X_all)

    model_payload = {
        "study_id": CONFIG["study_id"],
        "bundle_version": "1.1.0",
        "development_database": "INSPIRE_1.4.2",
        "external_validation_database": "MOVER_EPIC",
        "primary_model": "LM5_common18",
        "selected_ridge_lambda_from_development_CV": cv.best_l2,
        "models": final_models,
    }
    preprocess_payload = {
        "study_id": CONFIG["study_id"],
        "models": final_preprocess,
        "individual_missingness": "frozen_INSPIRE_median_imputation",
        "structural_missingness": "No-Go; whole columns may not be imputed",
    }
    thresholds_payload = {
        "study_id": CONFIG["study_id"],
        "capacity_thresholds": thresholds,
        "clinical_action_primary": CONFIG["thresholds"]["clinical_action_primary"],
        "clinical_action_sensitivity": CONFIG["thresholds"]["clinical_action_sensitivity"],
        "clinical_action": CONFIG["thresholds"]["clinical_action"],
        "legacy_30_feature_thresholds_forbidden": CONFIG["legacy_30_feature_thresholds_forbidden"],
        "mover_threshold_optimization_forbidden": True,
    }
    json_dump(MODEL_DIR / "model.json", model_payload)
    json_dump(MODEL_DIR / "preprocess.json", preprocess_payload)
    json_dump(MODEL_DIR / "thresholds.json", thresholds_payload)

    feature_contract = {
        "study_id": CONFIG["study_id"],
        "feature_order": features,
        "prediction_time": "t0+5 min",
        "definitions": {
            "age_years": "age available before anaesthesia",
            "male": "1 male, 0 female",
            "bmi": "weight_kg / height_m^2; MOVER unlabelled weight is ounces",
            "asa": "numeric ASA physical status",
            "t0_map": "selected H5 MAP at t0",
            "t0_map_squared": "t0_map squared",
            "t0_arterial_source": "1 if selected t0 H5 source is ART",
            "anesthesia_start_to_t0_min": "H5 t0 grid time from anaesthesia start",
            "pre10_map_record_count": "actual selected H5 points in [t0-10,t0)",
            "pre10_last_measurement_gap_min": "t0 minus last actual H5 point in [t0-10,t0)",
            "pre10_last_map": "last selected H5 MAP in [t0-10,t0)",
            "pre10_mean_map": "arithmetic mean of actual H5 MAP in [t0-10,t0)",
            "pre10_map_ols_slope_per_min": "OLS slope using actual H5 points in [t0-10,t0); missing if fewer than 2 times",
            "recovered_by_5min": "any actual selected H5 MAP >=65 in (t0,t0+5]",
            "early_auc65_0_5_mmhg_min": "trapezoidal MAP deficit below 65 over [t0,t0+5]",
            "early_min_map_0_5": "minimum selected H5 MAP in [t0,t0+5]",
            "early_mean_map_0_5": "mean selected H5 MAP in [t0,t0+5]",
            "early_map_record_count_0_5": "number of actual selected H5 points in [t0,t0+5]",
        },
        "unknown_value_rule": "leave individual missing values missing, then use frozen INSPIRE median",
        "structural_missingness_rule": "any wholly absent or semantically non-equivalent feature is No-Go",
    }
    json_dump(MODEL_DIR / "feature_contract.json", feature_contract)
    json_dump(
        MODEL_DIR / "cohort_endpoint.json",
        {
            "study_id": CONFIG["study_id"],
            "common_population": CONFIG["common_population"],
            "h5": CONFIG["h5"],
            "r1": CONFIG["r1"],
            "landmark": CONFIG["landmark"],
            "primary_endpoint": CONFIG["primary_endpoint"],
            "model_risk_set": "first fully evaluable operation per patient and not stage1 direct alert",
        },
    )

    split = data[["patient_id", "case_id", "primary_outcome"]].copy()
    split["split"] = ""
    split.iloc[development, split.columns.get_loc("split")] = "development"
    split.iloc[test, split.columns.get_loc("split")] = "patient_holdout"
    split["patient_hash"] = split["patient_id"].map(lambda value: stable_hash(value, "patient"))
    split["case_hash"] = split["case_id"].map(lambda value: stable_hash(value, "case"))
    split[["patient_hash", "case_hash", "primary_outcome", "split"]].to_csv(
        MODEL_DIR / "split_manifest.csv", index=False, encoding="utf-8-sig"
    )

    order = np.argsort(final_probabilities["LM5_common18"])
    positions = np.unique(np.linspace(0, len(order) - 1, 120, dtype=int))
    vectors = data.iloc[order[positions]][features].copy().reset_index(drop=True)
    vectors.insert(0, "vector_id", [f"TV{i+1:03d}" for i in range(len(vectors))])
    for name, model in final_models.items():
        X = apply_imputation(vectors, model["feature_order"], final_preprocess[name]["imputation_medians"], enforce_structural_gate=False)
        vectors[f"expected_probability_{name}"] = predict_model_dict(model, X)
    vectors.to_csv(MODEL_DIR / "test_vectors.csv", index=False, encoding="utf-8-sig")
    verification = verify_test_vectors(MODEL_DIR, tolerance=1e-12)
    json_dump(MODEL_DIR / "test_vector_verification.json", verification)

    precision = precision_plan(final_probabilities["LM5_common18"], assumed_rate=0.12)
    precision.to_csv(MODEL_DIR / "formal_external_validation_precision_plan.csv", index=False, encoding="utf-8-sig")
    json_dump(
        MODEL_DIR / "environment.lock",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "random_seed": SEED,
            "solver": "custom_damped_Newton_IRLS_numpy",
        },
    )

    # Record source and cohort fingerprints inside the immutable bundle.
    source_manifest = ROOT / "01_source_audit_lineage/source_manifest.json"
    json_dump(
        MODEL_DIR / "freeze_provenance.json",
        {
            "analysis_config_sha256": sha256_file(ROOT / "02_code_configs/configs/analysis.json"),
            "SAP_sha256": sha256_file(ROOT / "00_protocol_SAP/SAP_v1.0.md"),
            "deviation_log_sha256": sha256_file(ROOT / "00_protocol_SAP/deviation_log.csv"),
            "source_manifest_sha256": sha256_file(source_manifest),
            "inspire_model_cohort_sha256": sha256_file(data_path),
            "MOVER_patient_level_predictions_seen_before_freeze": False,
            "MOVER_performance_metrics_seen_before_freeze": False,
            "known_prior_MOVER_proxy_event_summary": True,
        },
    )
    hashes = sha256_tree(MODEL_DIR)
    hashes.to_csv(MODEL_DIR / "SHA256SUMS.csv", index=False, encoding="utf-8-sig")
    log(f"Freeze complete; {len(vectors)} test vectors; max error <=1e-12")


if __name__ == "__main__":
    main()

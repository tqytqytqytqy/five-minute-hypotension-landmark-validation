"""External-validation reporting and paired patient bootstrap."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .statistics import (
    auprc,
    auroc,
    binary_log_loss,
    brier_score,
    calibration_in_the_large,
    calibration_intercept_slope,
    decision_curve,
    evaluate_binary_predictions,
    fit_ridge_logistic,
    fixed_threshold_workload,
    ici_equal_frequency,
    oe_ratio_log_ci,
    scaled_brier_score,
)


def _strict_binary_outcome(y: np.ndarray, label: str) -> np.ndarray:
    try:
        values = np.asarray(y, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric binary 0 and 1") from error
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError(f"{label} must be a finite one-dimensional array")
    if not np.isin(values, [0.0, 1.0]).all():
        raise ValueError(f"{label} must contain exactly binary 0 and 1 values")
    return values.astype(np.int8)


def point_performance(y: np.ndarray, probability: np.ndarray) -> tuple[dict, pd.DataFrame]:
    base = evaluate_binary_predictions(y, probability).to_dict()
    citl = calibration_in_the_large(y, probability)
    slope = calibration_intercept_slope(y, probability)
    oe = oe_ratio_log_ci(y, probability)
    ici = ici_equal_frequency(y, probability, n_bins=10)
    result = {
        "n": int(len(y)),
        "events": int(np.sum(y)),
        "event_rate": float(np.mean(y)),
        **base,
        "auprc_over_event_rate": float(base["auprc"] / np.mean(y)) if np.mean(y) > 0 else np.nan,
        "calibration_in_the_large": float(citl.estimate),
        "calibration_in_the_large_se": float(citl.standard_error),
        "calibration_in_the_large_ci_lower": float(citl.ci_lower),
        "calibration_in_the_large_ci_upper": float(citl.ci_upper),
        "calibration_intercept_joint": float(slope.intercept),
        "calibration_slope": float(slope.slope),
        "calibration_slope_se": float(slope.slope_se),
        "calibration_slope_ci_lower": float(slope.slope_ci_lower),
        "calibration_slope_ci_upper": float(slope.slope_ci_upper),
        "observed_expected_ratio": float(oe.ratio),
        "observed_expected_ci_lower": float(oe.ci_lower),
        "observed_expected_ci_upper": float(oe.ci_upper),
        "ici_equal_frequency": float(ici.ici),
    }
    return result, ici.bins


def _rcs_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Restricted cubic spline basis: linear term plus K-2 nonlinear terms."""

    values = np.asarray(x, dtype=float)
    k = np.asarray(knots, dtype=float)
    if k.size != 4 or np.any(np.diff(k) <= 0):
        raise ValueError("four distinct increasing knots are required")

    def cube(z: np.ndarray) -> np.ndarray:
        return np.maximum(z, 0.0) ** 3

    span = (k[-1] - k[0]) ** 2
    columns = [values]
    for knot in k[:-2]:
        term = (
            cube(values - knot)
            - (k[-1] - knot) / (k[-1] - k[-2]) * cube(values - k[-2])
            + (k[-2] - knot) / (k[-1] - k[-2]) * cube(values - k[-1])
        ) / span
        columns.append(term)
    return np.column_stack(columns)


def rcs_calibration_curve(
    y: np.ndarray,
    probability: np.ndarray,
    *,
    knots: np.ndarray | None = None,
    grid_probability: np.ndarray | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    p = np.clip(np.asarray(probability, dtype=float), 1e-12, 1 - 1e-12)
    y = np.asarray(y, dtype=float)
    if np.unique(y).size != 2:
        raise ValueError("RCS calibration requires both outcome classes")
    lp = np.log(p / (1 - p))
    if not np.isfinite(lp).all() or float(np.var(lp)) <= 1e-12:
        raise ValueError("RCS calibration requires non-constant finite prediction logits")
    if knots is None:
        knots = np.unique(np.quantile(lp, [0.05, 0.35, 0.65, 0.95]))
        if len(knots) != 4:
            # Deterministic infinitesimal spreading for a near-discrete score.
            base = np.quantile(lp, [0.05, 0.35, 0.65, 0.95])
            knots = np.maximum.accumulate(base + np.arange(4) * 1e-8)
    if grid_probability is None:
        lower, upper = np.quantile(p, [0.01, 0.99])
        grid_probability = np.linspace(lower, upper, 101)
    basis = _rcs_basis(lp, knots)
    if np.linalg.matrix_rank(np.column_stack([np.ones(len(basis)), basis])) < basis.shape[1] + 1:
        raise ValueError("RCS calibration design matrix is rank deficient")
    model = fit_ridge_logistic(
        basis,
        y,
        l2=0.0,
        standardize=True,
        max_iter=250,
        tol=1e-9,
    )
    if not model.converged_:
        raise RuntimeError(f"RCS calibration model did not converge: {model.message_}")
    grid_p = np.clip(np.asarray(grid_probability, dtype=float), 1e-12, 1 - 1e-12)
    grid_lp = np.log(grid_p / (1 - grid_p))
    observed = model.predict_proba(_rcs_basis(grid_lp, knots))
    if not np.isfinite(observed).all():
        raise RuntimeError("RCS calibration produced non-finite fitted probabilities")
    return pd.DataFrame(
        {
            "predicted_probability": grid_p,
            "rcs_calibrated_observed_probability": observed,
        }
    ), np.asarray(knots, dtype=float)


def threshold_point_tables(
    y: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    patient_ids: np.ndarray,
    thresholds: Sequence[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    workload_parts = []
    dca_parts = []
    for name, probability in predictions.items():
        workload = fixed_threshold_workload(
            y, probability, thresholds, patient_ids=patient_ids
        )
        workload.insert(0, "model", name)
        workload["alerts_per_1000"] = workload["alert_rate"] * 1000
        workload["true_positive_alerts_per_1000"] = (
            workload["true_positive_weight"] / workload["landmark_weight"] * 1000
        )
        workload["false_alerts_per_1000"] = (
            workload["false_positive_weight"] / workload["landmark_weight"] * 1000
        )
        workload_parts.append(workload)
        dca = decision_curve(y, probability, thresholds)
        dca.insert(0, "model", name)
        dca_parts.append(dca)
    return pd.concat(workload_parts, ignore_index=True), pd.concat(dca_parts, ignore_index=True)


def two_stage_strategy_bootstrap(
    y: np.ndarray,
    stage1_alert: np.ndarray,
    stage2_probability: np.ndarray,
    patient_ids: np.ndarray,
    thresholds: Sequence[float],
    *,
    net_benefit_thresholds: Sequence[float] | None = None,
    n_boot: int = 2000,
    random_state: int = 20260712,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate stage-1 fixed alerts plus stage-2 thresholded probability.

    ``stage2_probability`` is NaN for stage-1 patients by design.  No continuous
    DCA is constructed for the mixed strategy; only a fixed binary strategy at
    each prespecified threshold is evaluated.
    """

    y = _strict_binary_outcome(y, "two-stage outcome")
    stage1_numeric = _strict_binary_outcome(stage1_alert, "stage-1 alert")
    stage1 = stage1_numeric.astype(bool)
    p = np.asarray(stage2_probability, dtype=float)
    ids = np.asarray(patient_ids)
    if p.ndim != 1 or ids.ndim != 1 or not (len(y) == len(stage1) == len(p) == len(ids)):
        raise ValueError("two-stage arrays must have equal length")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("two-stage primary cohort must be one row per patient")
    if not np.isfinite(p[~stage1]).all():
        raise ValueError("stage-2 patients are missing/non-finite frozen probabilities")
    if np.any((p[~stage1] < 0) | (p[~stage1] > 1)):
        raise ValueError("stage-2 probabilities are outside [0,1]")

    thresholds = sorted(set(float(value) for value in thresholds))
    nb_thresholds = set(
        thresholds
        if net_benefit_thresholds is None
        else [float(value) for value in net_benefit_thresholds]
    )
    if not nb_thresholds.issubset(set(thresholds)):
        raise ValueError("net-benefit thresholds must be a subset of workload thresholds")

    def metrics(index: np.ndarray, threshold: float) -> dict[str, float]:
        yy = y[index]
        s1 = stage1[index]
        pp = p[index]
        alert = s1 | ((~s1) & (pp >= threshold))
        tp = float(np.sum(alert & (yy == 1)))
        fp = float(np.sum(alert & (yy == 0)))
        fn = float(np.sum((~alert) & (yy == 1)))
        tn = float(np.sum((~alert) & (yy == 0)))
        odds = threshold / (1 - threshold)
        result = {
            "alert_rate": float(np.mean(alert)),
            "alerts_per_1000": float(np.mean(alert) * 1000),
            "true_positive_alerts_per_1000": tp / len(index) * 1000,
            "false_alerts_per_1000": fp / len(index) * 1000,
            "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
            "specificity": tn / (tn + fp) if tn + fp else np.nan,
            "positive_predictive_value": tp / (tp + fp) if tp + fp else np.nan,
            "alerts_per_true_positive": (tp + fp) / tp if tp else np.nan,
            "false_alerts_per_true_positive": fp / tp if tp else np.nan,
        }
        if threshold in nb_thresholds:
            strategy_nb = tp / len(index) - fp / len(index) * odds
            all_nb = float(np.mean(yy)) - (1 - float(np.mean(yy))) * odds
            result.update(
                {
                    "fixed_binary_strategy_net_benefit": strategy_nb,
                    "net_benefit_all": all_nb,
                    "net_benefit_none": 0.0,
                    "net_interventions_avoided_per_100_vs_all": (
                        (strategy_nb - all_nb) / odds * 100
                    ),
                }
            )
        return result

    point_rows = []
    all_index = np.arange(len(y))
    for threshold in thresholds:
        point_rows.append({"threshold": threshold, **metrics(all_index, threshold)})
    point = pd.DataFrame(point_rows)

    rng = np.random.default_rng(random_state)
    replicate_rows = []
    for replicate in range(n_boot):
        index = rng.integers(0, len(y), size=len(y))
        for threshold in thresholds:
            replicate_rows.append(
                {
                    "bootstrap_replicate": replicate + 1,
                    "threshold": threshold,
                    **metrics(index, threshold),
                }
            )
    replicates = pd.DataFrame(replicate_rows)
    summaries = []
    workload_metric_names = [
        "alert_rate",
        "alerts_per_1000",
        "true_positive_alerts_per_1000",
        "false_alerts_per_1000",
        "sensitivity",
        "specificity",
        "positive_predictive_value",
        "alerts_per_true_positive",
        "false_alerts_per_true_positive",
    ]
    net_benefit_metric_names = [
        "fixed_binary_strategy_net_benefit",
        "net_benefit_all",
        "net_benefit_none",
        "net_interventions_avoided_per_100_vs_all",
    ]
    for row in point.itertuples(index=False):
        subset = replicates[replicates["threshold"].eq(row.threshold)]
        metric_names = workload_metric_names + (
            net_benefit_metric_names if row.threshold in nb_thresholds else []
        )
        for metric_name in metric_names:
            values = pd.to_numeric(subset[metric_name], errors="coerce").dropna().to_numpy(float)
            summaries.append(
                {
                    "threshold": row.threshold,
                    "metric": metric_name,
                    "estimate": float(getattr(row, metric_name)),
                    "bootstrap_se": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                    "ci95_lower": float(np.quantile(values, 0.025)) if len(values) else np.nan,
                    "ci95_upper": float(np.quantile(values, 0.975)) if len(values) else np.nan,
                    "valid_replicates": len(values),
                    "requested_replicates": n_boot,
                }
            )
    return pd.DataFrame(summaries), replicates


def _safe_metric(metric, y: np.ndarray, p: np.ndarray) -> float:
    try:
        value = metric(y, p)
        scalar = float(value) if np.size(value) == 1 else np.nan
        return scalar if np.isfinite(scalar) else np.nan
    except (ValueError, RuntimeError, OverflowError, np.linalg.LinAlgError):
        return np.nan


def paired_bootstrap_validation(
    y: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    patient_ids: np.ndarray,
    thresholds: Sequence[float],
    *,
    primary_model: str,
    net_benefit_thresholds: Sequence[float] | None = None,
    n_boot: int = 2000,
    random_state: int = 20260712,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """One non-stratified patient draw per replicate for all models/metrics."""

    y = _strict_binary_outcome(y, "primary validation outcome")
    ids = np.asarray(patient_ids)
    if ids.ndim != 1 or len(y) != len(ids) or len(np.unique(ids)) != len(ids):
        raise ValueError("primary first-operation validation requires one row per patient")
    arrays = {name: np.asarray(p, dtype=float) for name, p in predictions.items()}
    if any(p.ndim != 1 or len(p) != len(y) for p in arrays.values()):
        raise ValueError("prediction lengths differ")
    if any(
        not np.isfinite(p).all() or np.any((p < 0) | (p > 1))
        for p in arrays.values()
    ):
        raise ValueError("predictions must be finite one-dimensional probabilities in [0,1]")
    if primary_model not in arrays:
        raise ValueError("primary_model absent from predictions")
    threshold_values = sorted(set(float(value) for value in thresholds))
    nb_threshold_values = sorted(
        set(
            threshold_values
            if net_benefit_thresholds is None
            else [float(value) for value in net_benefit_thresholds]
        )
    )
    if not set(nb_threshold_values).issubset(set(threshold_values)):
        raise ValueError("net-benefit thresholds must be a subset of workload thresholds")
    rng = np.random.default_rng(random_state)

    metric_functions = {
        "auroc": auroc,
        "auprc": auprc,
        "brier": brier_score,
        "scaled_brier": scaled_brier_score,
        "log_loss": binary_log_loss,
        "calibration_slope": lambda yy, pp: calibration_intercept_slope(yy, pp).slope,
        "calibration_in_the_large": lambda yy, pp: calibration_in_the_large(yy, pp).estimate,
        "observed_expected_ratio": lambda yy, pp: oe_ratio_log_ci(yy, pp).ratio,
        "ici_equal_frequency": lambda yy, pp: ici_equal_frequency(yy, pp, n_bins=10).ici,
    }

    point: dict[str, float] = {}
    for name, p in arrays.items():
        for metric_name, metric in metric_functions.items():
            point[f"{name}__{metric_name}"] = _safe_metric(metric, y, p)
        for threshold in threshold_values:
            alert = p >= threshold
            tp = float(np.sum(alert & (y == 1)))
            fp = float(np.sum(alert & (y == 0)))
            fn = float(np.sum((~alert) & (y == 1)))
            prefix = f"{name}__pt_{threshold:.6f}"
            point[f"{prefix}__alert_rate"] = float(np.mean(alert))
            point[f"{prefix}__sensitivity"] = tp / (tp + fn) if tp + fn else np.nan
            point[f"{prefix}__ppv"] = tp / (tp + fp) if tp + fp else np.nan
            point[f"{prefix}__alerts_per_true_positive"] = (tp + fp) / tp if tp else np.nan
            point[f"{prefix}__false_alerts_per_true_positive"] = fp / tp if tp else np.nan
            if threshold in nb_threshold_values:
                odds = threshold / (1 - threshold)
                point[f"{prefix}__net_benefit"] = tp / len(y) - fp / len(y) * odds

    # RCS curve uses fixed full-MOVER knots and grid, as prespecified.
    rcs_point, knots = rcs_calibration_curve(y, arrays[primary_model])
    grid = rcs_point["predicted_probability"].to_numpy(float)
    rcs_replicates = np.full((n_boot, len(grid)), np.nan)
    rows: list[dict[str, float]] = []
    for replicate in range(n_boot):
        index = rng.integers(0, len(y), size=len(y))
        yy = y[index]
        row: dict[str, float] = {"bootstrap_replicate": replicate + 1}
        for name, full_p in arrays.items():
            p = full_p[index]
            for metric_name, metric in metric_functions.items():
                row[f"{name}__{metric_name}"] = _safe_metric(metric, yy, p)
            for threshold in threshold_values:
                alert = p >= threshold
                tp = float(np.sum(alert & (yy == 1)))
                fp = float(np.sum(alert & (yy == 0)))
                fn = float(np.sum((~alert) & (yy == 1)))
                prefix = f"{name}__pt_{threshold:.6f}"
                row[f"{prefix}__alert_rate"] = float(np.mean(alert))
                row[f"{prefix}__sensitivity"] = tp / (tp + fn) if tp + fn else np.nan
                row[f"{prefix}__ppv"] = tp / (tp + fp) if tp + fp else np.nan
                row[f"{prefix}__alerts_per_true_positive"] = (
                    (tp + fp) / tp if tp else np.nan
                )
                row[f"{prefix}__false_alerts_per_true_positive"] = (
                    fp / tp if tp else np.nan
                )
                if threshold in nb_threshold_values:
                    odds = threshold / (1 - threshold)
                    row[f"{prefix}__net_benefit"] = tp / len(yy) - fp / len(yy) * odds
        rows.append(row)
        try:
            curve, _ = rcs_calibration_curve(
                yy, arrays[primary_model][index], knots=knots, grid_probability=grid
            )
            rcs_replicates[replicate] = curve[
                "rcs_calibrated_observed_probability"
            ].to_numpy(float)
        except (ValueError, RuntimeError, OverflowError, np.linalg.LinAlgError):
            pass

    replicates = pd.DataFrame(rows)
    # Add paired primary-minus-comparator differences on the identical draw.
    core_metrics = list(metric_functions)
    core_metrics += [f"pt_{threshold:.6f}__net_benefit" for threshold in nb_threshold_values]
    for comparator in arrays:
        if comparator == primary_model:
            continue
        for metric_name in core_metrics:
            primary_col = f"{primary_model}__{metric_name}"
            comparator_col = f"{comparator}__{metric_name}"
            contrast = f"delta__{primary_model}_minus_{comparator}__{metric_name}"
            replicates[contrast] = replicates[primary_col] - replicates[comparator_col]
            point[contrast] = point[primary_col] - point[comparator_col]

    summaries = []
    for column, estimate in point.items():
        candidate = pd.to_numeric(replicates[column], errors="coerce").to_numpy(float)
        values = candidate[np.isfinite(candidate)]
        summaries.append(
            {
                "estimand": column,
                "estimate": estimate,
                "bootstrap_se": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                "ci95_lower": float(np.quantile(values, 0.025)) if len(values) else np.nan,
                "ci95_upper": float(np.quantile(values, 0.975)) if len(values) else np.nan,
                "valid_replicates": len(values),
                "requested_replicates": n_boot,
                "sampling": "non-stratified patient-level paired bootstrap",
            }
        )
    summary = pd.DataFrame(summaries)
    primary_core = summary[
        summary["estimand"].isin(
            [f"{primary_model}__{metric}" for metric in metric_functions]
        )
    ]
    if primary_core.empty or (primary_core["valid_replicates"] < 0.95 * n_boot).any():
        raise RuntimeError("fewer than 95% valid bootstrap replicates for a primary metric")
    valid_curve = np.isfinite(rcs_replicates).all(axis=1)
    if int(valid_curve.sum()) < math.ceil(0.95 * n_boot):
        raise RuntimeError("fewer than 95% valid RCS bootstrap calibration curves")
    curve_summary = rcs_point.copy()
    curve_summary["ci95_lower"] = np.nanquantile(rcs_replicates, 0.025, axis=0)
    curve_summary["ci95_upper"] = np.nanquantile(rcs_replicates, 0.975, axis=0)
    curve_summary["valid_bootstrap_curves"] = int(valid_curve.sum())
    for index, value in enumerate(knots):
        curve_summary[f"lp_knot_{index + 1}"] = value
    return summary, replicates, curve_summary


__all__ = [
    "paired_bootstrap_validation",
    "point_performance",
    "rcs_calibration_curve",
    "threshold_point_tables",
    "two_stage_strategy_bootstrap",
]

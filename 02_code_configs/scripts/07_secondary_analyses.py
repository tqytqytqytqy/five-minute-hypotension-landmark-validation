#!/usr/bin/env python3
"""Prespecified post-U0 observation, drift, subgroup and update analyses."""

from __future__ import annotations

import gc
import hashlib
import itertools
import json
import math
import re
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_code_configs/src"))

from lm5_validation.cohort import canonical_cases, canonical_source_map  # noqa: E402
from lm5_validation.frozen import (  # noqa: E402
    apply_imputation,
    fit_imputation,
    load_bundle,
    predict_model_dict,
)
from lm5_validation.observation import (  # noqa: E402
    FEATURE_COLUMNS,
    build_h5,
    build_lm5_common18,
    classify_future_outcome,
    classify_stage1,
    find_h5_t0,
)
from lm5_validation.statistics import (  # noqa: E402
    auprc,
    auroc,
    binary_log_loss,
    brier_score,
    calibration_in_the_large,
    calibration_intercept_slope,
    evaluate_binary_predictions,
    fit_ridge_logistic,
    ici_equal_frequency,
    oe_ratio_log_ci,
    paired_patient_bootstrap,
)
from lm5_validation.validation import point_performance  # noqa: E402


CONFIG = json.loads((ROOT / "02_code_configs/configs/analysis.json").read_text(encoding="utf-8"))
SEED = int(CONFIG["random_seed"])
MOVER = ROOT / "03_derived_cohorts/MOVER"
INSPIRE = ROOT / "03_derived_cohorts/INSPIRE"
U0 = ROOT / "05_MOVER_validation/H5_primary"
R1_OUT = ROOT / "05_MOVER_validation/R1_sensitivity"
OBS_OUT = ROOT / "05_MOVER_validation/observation_2x2"
DRIFT_OUT = ROOT / "05_MOVER_validation/subgroups_drift"
PATHS = json.loads((ROOT / "02_code_configs/configs/paths.json").read_text(encoding="utf-8"))
SECONDARY_PRESPEC = json.loads(
    (ROOT / "00_protocol_SAP/secondary_analysis_prespec.json").read_text(encoding="utf-8")
)
N_BOOT = int(SECONDARY_PRESPEC["observation_2x2"]["patient_bootstrap_repetitions"])
if N_BOOT < 2000:
    raise ValueError("secondary analyses require at least 2000 patient bootstrap repetitions")

VASOPRESSOR_PATTERN = re.compile(
    r"phenylephrine|neo[- ]?synephrine|ephedrine|norepinephrine|levophed|"
    r"\bepinephrine\b|adrenaline|vasopressin|\bdopamine\b|\bdobutamine\b",
    re.I,
)
VASOPRESSOR_EXCLUDE = re.compile(
    r"lidocaine|bupivacaine|ropivacaine|tumescent|nasal|topical|infiltration|"
    r"subcutaneous|ophthalmic|nebul|local",
    re.I,
)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def analysis_seed(label: str) -> int:
    """Return a stable analysis-specific seed without relying on Python's hash salt."""

    digest = hashlib.sha256(f"{CONFIG['study_id']}|{label}|{SEED}".encode()).hexdigest()
    return int(digest[:8], 16)


def core_metrics() -> dict[str, object]:
    return {
        "auroc": auroc,
        "auprc": auprc,
        "brier": brier_score,
        "log_loss": binary_log_loss,
        "calibration_slope": lambda y, p: calibration_intercept_slope(y, p).slope,
        "calibration_in_the_large": lambda y, p: calibration_in_the_large(y, p).estimate,
    }


def performance_with_patient_ci(
    y: np.ndarray,
    probability: np.ndarray,
    patient_ids: np.ndarray,
    *,
    label: str,
) -> tuple[dict, pd.DataFrame]:
    """Point performance plus non-stratified patient-cluster bootstrap CIs."""

    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    patient_ids = np.asarray(patient_ids)
    if not (len(y) == len(probability) == len(patient_ids)):
        raise ValueError(f"{label}: outcome, probability and patient ID lengths differ")
    if len(y) == 0 or np.unique(y).size != 2:
        raise ValueError(f"{label}: both outcome classes are required")
    if pd.isna(patient_ids).any() or len(pd.unique(patient_ids)) < 2:
        raise ValueError(f"{label}: at least two nonmissing patient clusters are required")
    point, _ = point_performance(y, probability)
    summaries = []
    for metric_name, metric in core_metrics().items():
        result = paired_patient_bootstrap(
            y,
            {"estimate": probability},
            patient_ids,
            metric,
            n_boot=N_BOOT,
            random_state=analysis_seed(f"{label}|{metric_name}"),
            comparisons=[],
            return_replicates=False,
            min_valid_fraction=0.95,
        )
        row = result.summary.iloc[0].to_dict()
        row.update(
            {
                "analysis_label": label,
                "metric": metric_name,
                "n_patients": result.n_patients,
                "n_boot": result.n_boot,
            }
        )
        summaries.append(row)
        point[f"{metric_name}_patient_bootstrap_ci_lower"] = row["ci_lower"]
        point[f"{metric_name}_patient_bootstrap_ci_upper"] = row["ci_upper"]
        point[f"{metric_name}_patient_bootstrap_valid_replicates"] = row["n_valid"]
    point["patient_clusters"] = int(len(pd.unique(patient_ids)))
    point["bootstrap_repetitions"] = N_BOOT
    point["estimability_status"] = "estimable"
    point["nonestimable_reason"] = ""
    return point, pd.DataFrame(summaries)


def frozen_primary_probability(frame: pd.DataFrame, model: dict, preprocess: dict) -> np.ndarray:
    features = model["feature_order"]
    X = apply_imputation(
        frame,
        features,
        preprocess["imputation_medians"],
        enforce_structural_gate=True,
    )
    return predict_model_dict(model, X)


def assemble_process_cohort(
    cases: pd.DataFrame,
    series: pd.DataFrame,
    *,
    outcome_process: str,
    t0_search_start: float = 3.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    t0 = find_h5_t0(series, search_start_exclusive=t0_search_start)
    t0 = t0.merge(
        cases[["case_id", "anesthesia_duration_min"]], on="case_id", how="inner"
    )
    t0 = t0[t0["anesthesia_duration_min"].ge(t0["t0_min"] + 30)].copy()
    features, feature_audit = build_lm5_common18(cases, series, t0)
    stage1 = classify_stage1(series, t0)
    outcome = classify_future_outcome(series, t0, process=outcome_process)
    meta = cases.drop(columns=["age_years", "male", "bmi", "asa"])
    joined = (
        meta.merge(t0[["case_id", "t0_min", "t0_source"]], on="case_id", how="inner")
        .merge(feature_audit[["case_id", "feature_evaluable"]], on="case_id", how="left")
        .merge(features, on="case_id", how="left")
        .merge(
            stage1[
                ["case_id", "stage1_evaluable", "stage1_high_alert", "stage2_eligible"]
            ],
            on="case_id",
            how="left",
        )
        .merge(outcome.drop(columns=["t0_min"], errors="ignore"), on="case_id", how="left")
    )
    joined["fully_evaluable"] = (
        joined["feature_evaluable"].eq(1)
        & joined["stage1_evaluable"].eq(1)
        & joined["outcome_evaluable"].eq(1)
    )
    eligible = joined[joined["fully_evaluable"]].copy()
    order = (
        pd.to_datetime(eligible["anesthesia_start"], errors="coerce")
        if eligible["dataset"].astype(str).str.contains("MOVER").all()
        else pd.to_numeric(eligible["anesthesia_start"], errors="coerce")
    )
    eligible["_order"] = order
    first = (
        eligible.sort_values(["patient_id", "_order", "case_id"], kind="mergesort")
        .groupby("patient_id", sort=False)
        .head(1)
    )
    main = first[first["stage2_eligible"].eq(1)].drop(columns="_order").copy()
    flow = pd.DataFrame(
        [
            ["t0_after_search_gate", len(t0)],
            ["fully_evaluable_operations", len(eligible)],
            ["first_fully_evaluable_per_patient", len(first)],
            ["stage1_direct_alert", int(first["stage1_high_alert"].eq(1).sum())],
            ["stage2_model_cohort", len(main)],
            ["stage2_events", int(pd.to_numeric(main["primary_outcome"]).sum())],
        ],
        columns=["step", "n"],
    )
    return main, flow


def build_h5_variant(
    raw: pd.DataFrame,
    *,
    interval: str = "right_closed",
    statistic: str = "median",
    source_priority: tuple[str, str] = ("ART", "NIBP"),
) -> pd.DataFrame:
    """Prespecified H5 implementation variants used only after locked U0.

    ``nearest_right_boundary`` means the actual observation within each source
    whose timestamp is nearest to the interval's right boundary.  In a
    right-closed interval this is exactly the last timestamp within that
    interval.  Ties at an identical timestamp are reduced by their median;
    there is no cross-interval nearest-neighbour search, interpolation or LOCF.
    Source priority is applied only after the within-source observation is
    selected.
    """

    data = raw[["case_id", "minute_from_anesthesia_start", "map", "source"]].copy()
    data = data[
        data["source"].isin(["ART", "NIBP"])
        & data["map"].between(20, 200, inclusive="both")
    ].copy()
    time_values = pd.to_numeric(data["minute_from_anesthesia_start"], errors="coerce")
    if interval == "right_closed":
        data = data[time_values.gt(0)].copy()
        t = pd.to_numeric(data["minute_from_anesthesia_start"], errors="coerce")
        data["time_min"] = np.ceil(t / 5 - 1e-12) * 5
    elif interval == "left_closed":
        data = data[time_values.ge(0)].copy()
        t = pd.to_numeric(data["minute_from_anesthesia_start"], errors="coerce")
        data["time_min"] = (np.floor(t / 5 + 1e-12) + 1) * 5
    else:
        raise ValueError("interval must be right_closed or left_closed")
    if statistic == "median":
        within = (
            data.groupby(["case_id", "time_min", "source"], as_index=False)
            .agg(map=("map", "median"), n_raw_records=("map", "size"))
        )
    elif statistic == "nearest_right_boundary":
        at_time = (
            data.groupby(
                [
                    "case_id",
                    "time_min",
                    "source",
                    "minute_from_anesthesia_start",
                ],
                as_index=False,
            )
            .agg(map=("map", "median"), n_raw_records=("map", "size"))
        )
        within = (
            at_time.sort_values(
                ["case_id", "time_min", "source", "minute_from_anesthesia_start"]
            )
            .drop_duplicates(["case_id", "time_min", "source"], keep="last")
            [["case_id", "time_min", "source", "map", "n_raw_records"]]
        )
    else:
        raise ValueError("statistic must be median or nearest_right_boundary")
    priority = {source_priority[0]: 0, source_priority[1]: 1}
    within["_priority"] = within["source"].map(priority)
    total = (
        data.groupby(["case_id", "time_min"], as_index=False)
        .size()
        .rename(columns={"size": "n_raw_records_all_sources"})
    )
    selected = (
        within.sort_values(["case_id", "time_min", "_priority"])
        .drop_duplicates(["case_id", "time_min"], keep="first")
        .merge(total, on=["case_id", "time_min"], how="left")
    )
    return selected[
        [
            "case_id",
            "time_min",
            "map",
            "source",
            "n_raw_records",
            "n_raw_records_all_sources",
        ]
    ].sort_values(["case_id", "time_min"]).reset_index(drop=True)


def observation_2x2(
    primary_model: dict,
    primary_preprocess: dict,
    main_case_patient: pd.DataFrame,
) -> None:
    expected_cells = [
        "H5_features__H5_outcome",
        "H5_features__R1_outcome",
        "R1_features__H5_outcome",
        "R1_features__R1_outcome",
    ]
    main_case_patient = main_case_patient.copy()
    main_case_patient["case_id"] = main_case_patient["case_id"].astype("string")
    main_case_patient["patient_id"] = main_case_patient["patient_id"].astype("string")
    if main_case_patient["case_id"].duplicated().any() or main_case_patient["patient_id"].isna().any():
        raise ValueError("fixed H5 base must have unique cases and nonmissing patient IDs")
    cells = pd.read_csv(
        MOVER / "mover_observation_2x2_cells.csv.gz",
        dtype={"case_id": "string"},
        low_memory=False,
    )
    observed_cells = set(cells["cell"].dropna().astype(str))
    if observed_cells != set(expected_cells):
        raise ValueError(
            f"observation 2x2 requires exactly the four prespecified cells; got {sorted(observed_cells)}"
        )
    cells = cells.merge(
        main_case_patient[["case_id", "patient_id"]],
        on="case_id",
        how="left",
        validate="many_to_one",
    )
    if cells["patient_id"].isna().any():
        raise ValueError("observation 2x2 contains cases outside the fixed H5 base")
    fixed_base_n = int(main_case_patient["case_id"].nunique())
    point_rows = []
    predicted_parts = []
    for cell in expected_cells:
        frame = cells[cells["cell"].eq(cell)].copy()
        if frame["case_id"].duplicated().any():
            raise ValueError(f"{cell}: duplicate case rows")
        if frame["case_id"].nunique() != fixed_base_n:
            raise ValueError(f"{cell}: does not retain every case in the fixed H5 base")
        evaluable = frame[frame["cell_evaluable"].eq(1)].copy()
        base = {
            "cell": cell,
            "feature_process": cell.split("_features__")[0],
            "outcome_process": cell.split("__")[1].split("_outcome")[0],
            "fixed_H5_base_n": fixed_base_n,
            "cell_evaluable_n": len(evaluable),
            "cell_evaluable_percent": 100 * len(evaluable) / fixed_base_n if fixed_base_n else np.nan,
        }
        y_numeric = pd.to_numeric(evaluable["primary_outcome"], errors="raise")
        if y_numeric.isna().any() or not y_numeric.isin([0, 1]).all():
            raise ValueError(f"{cell}: evaluable outcomes must be strictly binary")
        y = y_numeric.astype(int).to_numpy()
        if len(y) == 0 or np.unique(y).size != 2:
            point_rows.append(
                {
                    **base,
                    "events": int(y.sum()) if len(y) else 0,
                    "estimability_status": "nonestimable",
                    "nonestimable_reason": "empty cell or only one outcome class",
                }
            )
            continue
        probability = frozen_primary_probability(evaluable, primary_model, primary_preprocess)
        row, _ = point_performance(y, probability)
        row.update(
            {
                **base,
                "estimability_status": "estimable",
                "nonestimable_reason": "",
            }
        )
        point_rows.append(row)
        predicted_parts.append(
            pd.DataFrame(
                {
                    "case_id": evaluable["case_id"].astype(str),
                    "patient_id": evaluable["patient_id"].astype(str),
                    "cell": cell,
                    "primary_outcome": y,
                    "probability": probability,
                }
            )
        )
    point = pd.DataFrame(point_rows)
    point.to_csv(OBS_OUT / "observation_2x2_point_performance.csv", index=False, encoding="utf-8-sig")
    if len(predicted_parts) != 4:
        raise RuntimeError("all four 2x2 cells must be estimable for factorial contrasts")
    predicted = pd.concat(predicted_parts, ignore_index=True)

    # Complete four-cell intersection for the full 2x2 factorial estimands.
    counts = predicted.groupby("case_id")["cell"].nunique()
    complete_ids = set(counts[counts.eq(4)].index)
    complete = predicted[predicted["case_id"].isin(complete_ids)].copy()
    if not complete_ids:
        raise RuntimeError("four-cell complete intersection is empty")
    cell_names = expected_cells
    matrices = {}
    for cell in cell_names:
        frame = complete[complete["cell"].eq(cell)].sort_values("case_id")
        matrices[cell] = {
            "case_id": frame["case_id"].to_numpy(),
            "patient_id": frame["patient_id"].to_numpy(),
            "y": frame["primary_outcome"].to_numpy(int),
            "p": frame["probability"].to_numpy(float),
        }
    reference_ids = matrices[cell_names[0]]["case_id"]
    if any(not np.array_equal(reference_ids, matrices[name]["case_id"]) for name in cell_names):
        raise AssertionError("2x2 complete-cell case ordering differs")
    reference_patients = matrices[cell_names[0]]["patient_id"]
    if any(
        not np.array_equal(reference_patients, matrices[name]["patient_id"])
        for name in cell_names
    ):
        raise AssertionError("2x2 complete-cell patient ordering differs")
    metrics = core_metrics()
    point_complete = {}
    for cell in cell_names:
        for metric_name, metric in metrics.items():
            point_complete[f"{cell}__{metric_name}"] = float(
                metric(matrices[cell]["y"], matrices[cell]["p"])
            )
    patient_codes, unique_patients = pd.factorize(reference_patients, sort=False)
    cluster_rows = [np.flatnonzero(patient_codes == code) for code in range(len(unique_patients))]
    if len(cluster_rows) < 2:
        raise RuntimeError("factorial bootstrap requires at least two patient clusters")
    rng = np.random.default_rng(analysis_seed("observation_2x2_complete_factorial"))
    boot_rows = []
    for replicate in range(N_BOOT):
        selected_clusters = rng.integers(0, len(cluster_rows), size=len(cluster_rows))
        index = np.concatenate([cluster_rows[code] for code in selected_clusters])
        row = {"bootstrap_replicate": replicate + 1}
        for cell in cell_names:
            for metric_name, metric in metrics.items():
                try:
                    row[f"{cell}__{metric_name}"] = float(
                        metric(matrices[cell]["y"][index], matrices[cell]["p"][index])
                    )
                except Exception:
                    row[f"{cell}__{metric_name}"] = np.nan
        boot_rows.append(row)
    boot = pd.DataFrame(boot_rows)
    summaries = []

    def add_summary(estimand: str, estimate: float, values: pd.Series, kind: str) -> None:
        valid = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
        if len(valid) < math.ceil(0.95 * N_BOOT):
            raise RuntimeError(f"{estimand} has fewer than 95% valid bootstrap replicates")
        summaries.append(
            {
                "estimand": estimand,
                "estimand_kind": kind,
                "estimate": estimate,
                "ci95_lower": np.quantile(valid, 0.025),
                "ci95_upper": np.quantile(valid, 0.975),
                "valid_replicates": len(valid),
                "bootstrap_repetitions": N_BOOT,
                "complete_intersection_n": len(reference_ids),
                "complete_intersection_patient_clusters": len(unique_patients),
            }
        )

    for column, estimate in point_complete.items():
        add_summary(column, estimate, boot[column], "cell")

    # All six paired cell contrasts, followed by the complete factorial effects.
    for first_cell, second_cell in itertools.combinations(cell_names, 2):
        for metric_name in metrics:
            first = f"{first_cell}__{metric_name}"
            second = f"{second_cell}__{metric_name}"
            name = f"delta__{first_cell}_minus_{second_cell}__{metric_name}"
            boot[name] = boot[first] - boot[second]
            add_summary(
                name,
                point_complete[first] - point_complete[second],
                boot[name],
                "paired_cell_contrast",
            )

    a = "H5_features__H5_outcome"
    b = "H5_features__R1_outcome"
    c = "R1_features__H5_outcome"
    d = "R1_features__R1_outcome"
    for metric_name in metrics:
        cols = {name: f"{name}__{metric_name}" for name in [a, b, c, d]}
        estimands = {
            "feature_R1_minus_H5_at_H5_outcome": ({c: 1, a: -1}),
            "feature_R1_minus_H5_at_R1_outcome": ({d: 1, b: -1}),
            "outcome_R1_minus_H5_at_H5_features": ({b: 1, a: -1}),
            "outcome_R1_minus_H5_at_R1_features": ({d: 1, c: -1}),
            "marginal_feature_R1_minus_H5": ({c: 0.5, a: -0.5, d: 0.5, b: -0.5}),
            "marginal_outcome_R1_minus_H5": ({b: 0.5, a: -0.5, d: 0.5, c: -0.5}),
            "feature_by_outcome_interaction": ({d: 1, c: -1, b: -1, a: 1}),
        }
        for effect_name, weights in estimands.items():
            name = f"factorial__{effect_name}__{metric_name}"
            estimate = sum(weight * point_complete[cols[cell]] for cell, weight in weights.items())
            boot[name] = sum(weight * boot[cols[cell]] for cell, weight in weights.items())
            add_summary(name, estimate, boot[name], "factorial_effect")
    pd.DataFrame(summaries).to_csv(
        OBS_OUT / f"observation_2x2_complete_factorial_bootstrap_{N_BOOT}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    boot.to_csv(
        OBS_OUT / "observation_2x2_complete_factorial_replicates.csv.gz",
        index=False,
        compression={"method": "gzip", "compresslevel": 5},
    )

    # Label agreement uses its pairwise-complete set, not the four-cell intersection.
    label = predicted[
        predicted["cell"].isin(["H5_features__H5_outcome", "H5_features__R1_outcome"])
    ].pivot(index="case_id", columns="cell", values="primary_outcome").dropna()
    if label.empty:
        raise RuntimeError("H5/R1 outcome pairwise-complete set is empty")
    h = label["H5_features__H5_outcome"].astype(int)
    r = label["H5_features__R1_outcome"].astype(int)
    a = int(((h == 1) & (r == 1)).sum())
    b = int(((h == 1) & (r == 0)).sum())
    c = int(((h == 0) & (r == 1)).sum())
    d = int(((h == 0) & (r == 0)).sum())
    observed_agreement = (a + d) / len(label)
    expected_agreement = (
        ((a + b) / len(label)) * ((a + c) / len(label))
        + ((c + d) / len(label)) * ((b + d) / len(label))
    )
    kappa_denominator = 1 - expected_agreement
    agreement = pd.DataFrame(
        [
            {
                "n": len(label),
                "both_positive": a,
                "H5_positive_R1_negative": b,
                "H5_negative_R1_positive": c,
                "both_negative": d,
                "positive_agreement": 2 * a / (2 * a + b + c) if 2 * a + b + c else np.nan,
                "negative_agreement": 2 * d / (2 * d + b + c) if 2 * d + b + c else np.nan,
                "overall_agreement": observed_agreement,
                "cohen_kappa": (
                    (observed_agreement - expected_agreement) / kappa_denominator
                    if kappa_denominator > 0
                    else np.nan
                ),
                "estimability_status": "estimable" if kappa_denominator > 0 else "kappa_nonestimable",
            }
        ]
    )
    agreement.to_csv(OBS_OUT / "H5_R1_outcome_agreement.csv", index=False, encoding="utf-8-sig")

    # Prediction-process shift likewise uses its pairwise-complete set.
    prediction = predicted[
        predicted["cell"].isin(["H5_features__H5_outcome", "R1_features__H5_outcome"])
    ].pivot(index="case_id", columns="cell", values="probability").dropna()
    if prediction.empty:
        raise RuntimeError("H5/R1 feature-process pairwise-complete set is empty")
    h5p = prediction["H5_features__H5_outcome"]
    r1p = prediction["R1_features__H5_outcome"]
    shift_rows = [
        {
            "n": len(prediction),
            "mean_H5_probability": h5p.mean(),
            "mean_R1_probability": r1p.mean(),
            "mean_difference_R1_minus_H5": (r1p - h5p).mean(),
            "mean_absolute_difference": (r1p - h5p).abs().mean(),
            "pearson_correlation": h5p.corr(r1p),
            "spearman_correlation": h5p.rank().corr(r1p.rank()),
        }
    ]
    pd.DataFrame(shift_rows).to_csv(
        OBS_OUT / "H5_R1_prediction_process_shift.csv", index=False, encoding="utf-8-sig"
    )


def sensitivity_cohorts(
    cases: pd.DataFrame,
    raw_formal: pd.DataFrame,
    h5_primary: pd.DataFrame,
    r1: pd.DataFrame,
    primary_model: dict,
    primary_preprocess: dict,
) -> None:
    rows = []
    flows = []
    bootstrap_parts = []

    def add_performance(
        cohort: pd.DataFrame,
        probability: np.ndarray,
        *,
        analysis: str,
        setting: object,
    ) -> None:
        y = pd.to_numeric(cohort["primary_outcome"], errors="raise").astype(int).to_numpy()
        result, bootstrap = performance_with_patient_ci(
            y,
            probability,
            cohort["patient_id"].to_numpy(),
            label=f"{analysis}|{setting}",
        )
        result.update(
            {
                "analysis": analysis,
                "setting": setting,
                "population_handling": "full cohort re-selection under each observation variant",
            }
        )
        rows.append(result)
        bootstrap.insert(0, "setting", setting)
        bootstrap.insert(0, "analysis", analysis)
        bootstrap_parts.append(bootstrap)

    # Phase 0 is the locked primary input; phases 1-4 are sensitivity cohorts.
    for phase in range(5):
        h5 = h5_primary if phase == 0 else build_h5(raw_formal, phase_min=float(phase))
        cohort, flow = assemble_process_cohort(cases, h5, outcome_process="H5")
        probability = frozen_primary_probability(cohort, primary_model, primary_preprocess)
        add_performance(cohort, probability, analysis="H5_phase", setting=phase)
        flow.insert(0, "setting", phase)
        flow.insert(0, "analysis", "H5_phase")
        flows.append(flow)
        del h5, cohort
        gc.collect()

    for buffer in [5.0, 10.0]:
        cohort, flow = assemble_process_cohort(
            cases, h5_primary, outcome_process="H5", t0_search_start=buffer
        )
        probability = frozen_primary_probability(cohort, primary_model, primary_preprocess)
        add_performance(
            cohort,
            probability,
            analysis="induction_buffer_exclusive_min",
            setting=buffer,
        )
        flow.insert(0, "setting", buffer)
        flow.insert(0, "analysis", "induction_buffer_exclusive_min")
        flows.append(flow)

    variants = [
        ("right_closed_nearest_right_boundary_ART_priority", {"interval": "right_closed", "statistic": "nearest_right_boundary", "source_priority": ("ART", "NIBP")}),
        ("right_closed_median_NIBP_priority", {"interval": "right_closed", "statistic": "median", "source_priority": ("NIBP", "ART")}),
        ("left_closed_median_ART_priority", {"interval": "left_closed", "statistic": "median", "source_priority": ("ART", "NIBP")}),
    ]
    for name, kwargs in variants:
        variant_h5 = build_h5_variant(raw_formal, **kwargs)
        cohort, flow = assemble_process_cohort(cases, variant_h5, outcome_process="H5")
        probability = frozen_primary_probability(cohort, primary_model, primary_preprocess)
        add_performance(cohort, probability, analysis="H5_operator_variant", setting=name)
        flow.insert(0, "setting", name)
        flow.insert(0, "analysis", "H5_operator_variant")
        flows.append(flow)
        del variant_h5, cohort
        gc.collect()

    r1_cohort, r1_flow = assemble_process_cohort(cases, r1, outcome_process="R1")
    r1_probability = frozen_primary_probability(r1_cohort, primary_model, primary_preprocess)
    add_performance(r1_cohort, r1_probability, analysis="R1_redetected_t0", setting=0)
    r1_flow.insert(0, "setting", 0)
    r1_flow.insert(0, "analysis", "R1_redetected_t0")
    flows.append(r1_flow)
    r1_cohort.to_csv(
        R1_OUT / "R1_redetected_t0_main_stage2.csv.gz",
        index=False,
        compression={"method": "gzip", "compresslevel": 5},
    )
    pd.DataFrame(rows).to_csv(
        R1_OUT / "phase_buffer_R1_redetect_performance.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(flows, ignore_index=True).to_csv(
        R1_OUT / "phase_buffer_R1_redetect_cohort_flows.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(bootstrap_parts, ignore_index=True).to_csv(
        R1_OUT / f"phase_buffer_operator_R1_patient_bootstrap_{N_BOOT}.csv",
        index=False,
        encoding="utf-8-sig",
    )


def all_surgeries_patient_cluster_sensitivity(
    first_operation_main: pd.DataFrame,
    primary_model: dict,
    primary_preprocess: dict,
) -> None:
    """Compare the locked first-operation cohort with all eligible operations.

    The all-operation analysis changes only the operation-selection rule.  Its
    uncertainty is therefore resampled by patient cluster so that repeated
    operations from the same patient are never treated as independent.
    """

    all_t0 = pd.read_csv(
        MOVER / "mover_all_t0_operations.csv.gz",
        dtype={"case_id": "string", "patient_id": "string"},
        low_memory=False,
    )
    all_operations = all_t0[
        all_t0["fully_evaluable_t0_operation"].eq(1)
        & all_t0["stage2_eligible"].eq(1)
    ].copy()
    if all_operations["case_id"].duplicated().any():
        raise ValueError("all-operation sensitivity contains duplicate case IDs")
    cohorts = {
        "first_fully_evaluable_operation_primary_rule": first_operation_main,
        "all_fully_evaluable_stage2_operations": all_operations,
    }
    point_rows = []
    bootstrap_parts = []
    for selection_rule, cohort in cohorts.items():
        probability = frozen_primary_probability(cohort, primary_model, primary_preprocess)
        y = pd.to_numeric(cohort["primary_outcome"], errors="raise").astype(int).to_numpy()
        point, bootstrap = performance_with_patient_ci(
            y,
            probability,
            cohort["patient_id"].to_numpy(),
            label=f"operation_selection|{selection_rule}",
        )
        point.update(
            {
                "operation_selection_rule": selection_rule,
                "operations": len(cohort),
                "patients": int(cohort["patient_id"].nunique()),
                "patients_with_multiple_included_operations": int(
                    cohort.groupby("patient_id").size().gt(1).sum()
                ),
                "uncertainty_sampling_unit": "patient_cluster",
            }
        )
        point_rows.append(point)
        bootstrap.insert(0, "operation_selection_rule", selection_rule)
        bootstrap_parts.append(bootstrap)
    pd.DataFrame(point_rows).to_csv(
        R1_OUT / "first_vs_all_operations_patient_cluster_sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(bootstrap_parts, ignore_index=True).to_csv(
        R1_OUT / f"first_vs_all_operations_patient_bootstrap_{N_BOOT}.csv",
        index=False,
        encoding="utf-8-sig",
    )


def endpoint_sensitivities(
    main: pd.DataFrame,
    h5: pd.DataFrame,
    primary_model: dict,
    primary_preprocess: dict,
) -> None:
    probability = frozen_primary_probability(main, primary_model, primary_preprocess)
    t0 = main[["case_id", "t0_min"]]
    settings = [
        ("persistent_15_or_auc75", {"persistent_recovery_after_min": 15.0, "auc_event_threshold": 75.0}),
        ("persistent_10_or_auc50", {"persistent_recovery_after_min": 10.0, "auc_event_threshold": 50.0}),
        ("MAP60_persistent10_or_auc75", {"threshold": 60.0, "persistent_recovery_after_min": 10.0, "auc_event_threshold": 75.0}),
    ]
    rows = []
    bootstrap_parts = []
    component_rows = []
    reclassification_rows = []
    reference = pd.to_numeric(main["primary_outcome"], errors="raise").astype(int).to_numpy()

    def component_and_reclassification(
        name: str,
        persistent: np.ndarray,
        high_burden: np.ndarray,
        sensitivity_y: np.ndarray,
        reference_y: np.ndarray,
    ) -> None:
        persistent = np.asarray(persistent, dtype=bool)
        high_burden = np.asarray(high_burden, dtype=bool)
        sensitivity_y = np.asarray(sensitivity_y, dtype=int)
        reference_y = np.asarray(reference_y, dtype=int)
        n = len(sensitivity_y)
        component_rows.append(
            {
                "endpoint_sensitivity": name,
                "pairwise_evaluable_n": n,
                "persistent_component_n": int(persistent.sum()),
                "persistent_component_percent": 100 * persistent.mean() if n else np.nan,
                "high_burden_component_n": int(high_burden.sum()),
                "high_burden_component_percent": 100 * high_burden.mean() if n else np.nan,
                "both_components_n": int((persistent & high_burden).sum()),
                "persistent_only_n": int((persistent & ~high_burden).sum()),
                "high_burden_only_n": int((~persistent & high_burden).sum()),
                "neither_component_n": int((~persistent & ~high_burden).sum()),
                "composite_event_n": int(sensitivity_y.sum()),
                "composite_event_percent": 100 * sensitivity_y.mean() if n else np.nan,
            }
        )
        reclassification_rows.append(
            {
                "endpoint_sensitivity": name,
                "pairwise_evaluable_n": n,
                "primary0_sensitivity0": int(((reference_y == 0) & (sensitivity_y == 0)).sum()),
                "primary0_sensitivity1": int(((reference_y == 0) & (sensitivity_y == 1)).sum()),
                "primary1_sensitivity0": int(((reference_y == 1) & (sensitivity_y == 0)).sum()),
                "primary1_sensitivity1": int(((reference_y == 1) & (sensitivity_y == 1)).sum()),
                "absolute_reclassified_n": int((reference_y != sensitivity_y).sum()),
                "absolute_reclassified_percent": 100 * (reference_y != sensitivity_y).mean() if n else np.nan,
            }
        )

    for name, kwargs in settings:
        outcome = classify_future_outcome(h5, t0, process="H5", **kwargs)
        merged = main[["case_id", "patient_id", "primary_outcome"]].rename(
            columns={"primary_outcome": "reference_primary_outcome"}
        ).merge(
            outcome[
                [
                    "case_id",
                    "outcome_evaluable",
                    "future_persistent_recovery_gt10",
                    "future_high_burden_auc_ge75",
                    "primary_outcome",
                ]
            ],
            on="case_id",
            how="left",
            validate="one_to_one",
        )
        keep = merged["outcome_evaluable"].eq(1)
        y = pd.to_numeric(merged.loc[keep, "primary_outcome"]).astype(int).to_numpy()
        result, bootstrap = performance_with_patient_ci(
            y,
            probability[keep.to_numpy()],
            merged.loc[keep, "patient_id"].to_numpy(),
            label=f"endpoint|{name}",
        )
        result.update(
            {
                "endpoint_sensitivity": name,
                "base_main_n": len(main),
                "sensitivity_evaluable_n": int(keep.sum()),
                "sensitivity_evaluable_percent": 100 * keep.mean(),
            }
        )
        rows.append(result)
        bootstrap.insert(0, "endpoint_sensitivity", name)
        bootstrap_parts.append(bootstrap)
        component_and_reclassification(
            name,
            pd.to_numeric(
                merged.loc[keep, "future_persistent_recovery_gt10"], errors="raise"
            ).astype(int).to_numpy(),
            pd.to_numeric(
                merged.loc[keep, "future_high_burden_auc_ge75"], errors="raise"
            ).astype(int).to_numpy(),
            y,
            pd.to_numeric(
                merged.loc[keep, "reference_primary_outcome"], errors="raise"
            ).astype(int).to_numpy(),
        )
    # Complete-six-point coverage is the >=90% / max-gap<=5 H5 sensitivity.
    complete = main["n_outcome_grid_points"].eq(6) & main["max_adjacent_gap_min"].le(5)
    complete_y = pd.to_numeric(main.loc[complete, "primary_outcome"]).astype(int).to_numpy()
    result, bootstrap = performance_with_patient_ci(
        complete_y,
        probability[complete.to_numpy()],
        main.loc[complete, "patient_id"].to_numpy(),
        label="endpoint|H5_all_6_points_max_gap_5",
    )
    result.update(
        {
            "endpoint_sensitivity": "H5_all_6_points_max_gap_5",
            "base_main_n": len(main),
            "sensitivity_evaluable_n": int(complete.sum()),
            "sensitivity_evaluable_percent": 100 * complete.mean(),
        }
    )
    rows.append(result)
    bootstrap.insert(0, "endpoint_sensitivity", "H5_all_6_points_max_gap_5")
    bootstrap_parts.append(bootstrap)
    component_and_reclassification(
        "H5_all_6_points_max_gap_5",
        pd.to_numeric(
            main.loc[complete, "future_persistent_recovery_gt10"], errors="raise"
        ).astype(int).to_numpy(),
        pd.to_numeric(
            main.loc[complete, "future_high_burden_auc_ge75"], errors="raise"
        ).astype(int).to_numpy(),
        complete_y,
        reference[complete.to_numpy()],
    )
    pd.DataFrame(rows).to_csv(
        R1_OUT / "endpoint_and_coverage_sensitivity_performance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(bootstrap_parts, ignore_index=True).to_csv(
        R1_OUT / f"endpoint_and_coverage_patient_bootstrap_{N_BOOT}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(component_rows).to_csv(
        R1_OUT / "endpoint_component_prevalence.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(reclassification_rows).to_csv(
        R1_OUT / "endpoint_reclassification_vs_primary.csv",
        index=False,
        encoding="utf-8-sig",
    )


def drift_and_subgroups(
    inspire: pd.DataFrame,
    mover: pd.DataFrame,
    probability: np.ndarray,
) -> None:
    drift_rows = []
    for feature in FEATURE_COLUMNS:
        a = pd.to_numeric(inspire[feature], errors="coerce")
        b = pd.to_numeric(mover[feature], errors="coerce")
        mean_a, mean_b = a.mean(), b.mean()
        sd_a, sd_b = a.std(ddof=1), b.std(ddof=1)
        pooled = math.sqrt((sd_a**2 + sd_b**2) / 2) if np.isfinite(sd_a + sd_b) else np.nan
        q1, q99 = a.quantile([0.01, 0.99])
        b_nonmissing = b.notna()
        outside = b_nonmissing & ((b < q1) | (b > q99))
        drift_rows.append(
            {
                "feature": feature,
                "INSPIRE_n": int(a.notna().sum()),
                "MOVER_n": int(b.notna().sum()),
                "INSPIRE_mean": mean_a,
                "MOVER_mean": mean_b,
                "INSPIRE_sd": sd_a,
                "MOVER_sd": sd_b,
                "standardized_mean_difference_MOVER_minus_INSPIRE": (mean_b - mean_a) / pooled if pooled else np.nan,
                "INSPIRE_missing_percent": 100 * a.isna().mean(),
                "MOVER_missing_percent": 100 * b.isna().mean(),
                "INSPIRE_p01": q1,
                "INSPIRE_p99": q99,
                "MOVER_outside_INSPIRE_p01_p99_n": int(outside.sum()),
                "MOVER_outside_INSPIRE_p01_p99_denominator_nonmissing_n": int(b_nonmissing.sum()),
                "MOVER_outside_INSPIRE_p01_p99_percent": (
                    100 * outside.sum() / b_nonmissing.sum() if b_nonmissing.sum() else np.nan
                ),
            }
        )
    pd.DataFrame(drift_rows).to_csv(
        DRIFT_OUT / "feature_drift_INSPIRE_vs_MOVER.csv", index=False, encoding="utf-8-sig"
    )

    groups = {
        "age_lt65": mover["age_years"] < 65,
        "age_ge65": mover["age_years"] >= 65,
        "female": mover["male"] == 0,
        "male": mover["male"] == 1,
        "ASA_1_2": mover["asa"] <= 2,
        "ASA_ge3": mover["asa"] >= 3,
        "BMI_lt25": mover["bmi"] < 25,
        "BMI_25_30": mover["bmi"].between(25, 30, inclusive="left"),
        "BMI_ge30": mover["bmi"] >= 30,
        "t0_ART": mover["t0_arterial_source"] == 1,
        "t0_NIBP": mover["t0_arterial_source"] == 0,
        "t0_early_le10": mover["t0_min"] <= 10,
        "t0_late_gt10": mover["t0_min"] > 10,
    }
    rows = []
    bootstrap_parts = []
    for name, mask in groups.items():
        observed = mask.fillna(False).to_numpy()
        y = pd.to_numeric(mover.loc[observed, "primary_outcome"]).astype(int).to_numpy()
        events = int(y.sum())
        non_events = int(len(y) - events)
        base = {
            "subgroup": name,
            "subgroup_n": len(y),
            "subgroup_events": events,
            "subgroup_non_events": non_events,
            "main_cohort_n": len(mover),
            "excluded_or_not_in_category_n": int(len(mover) - len(y)),
            "estimability_gate": "n>=100 and events>=20 and non_events>=20",
        }
        if len(y) < 100 or events < 20 or non_events < 20:
            result = {
                **base,
                "estimability_status": "nonestimable",
                "nonestimable_reason": (
                    f"n={len(y)}, events={events}, non_events={non_events}; "
                    "prespecified minimum-information gate not met"
                ),
            }
            rows.append(result)
            continue
        result, bootstrap = performance_with_patient_ci(
            y,
            probability[observed],
            mover.loc[observed, "patient_id"].to_numpy(),
            label=f"subgroup|{name}",
        )
        result.update(base)
        rows.append(result)
        bootstrap.insert(0, "subgroup", name)
        bootstrap_parts.append(bootstrap)
    pd.DataFrame(rows).to_csv(
        DRIFT_OUT / "prespecified_subgroup_performance.csv", index=False, encoding="utf-8-sig"
    )
    if bootstrap_parts:
        pd.concat(bootstrap_parts, ignore_index=True).to_csv(
            DRIFT_OUT / f"prespecified_subgroup_patient_bootstrap_{N_BOOT}.csv",
            index=False,
            encoding="utf-8-sig",
        )


def update_analysis(
    mover: pd.DataFrame,
    u0_probability: np.ndarray,
) -> None:
    y = pd.to_numeric(mover["primary_outcome"]).astype(int).to_numpy()
    patient_hash = mover["patient_id"].astype(str).map(
        lambda value: hashlib.sha256(f"{CONFIG['study_id']}|update_split|{value}".encode()).hexdigest()
    )
    update_mask = patient_hash.str.slice(0, 8).map(lambda value: int(value, 16) % 2 == 0).to_numpy()
    evaluation_mask = ~update_mask
    if np.unique(y[update_mask]).size != 2 or np.unique(y[evaluation_mask]).size != 2:
        raise RuntimeError("U1/U2 update and evaluation halves must each contain both outcome classes")
    lp = np.log(np.clip(u0_probability, 1e-12, 1 - 1e-12) / np.clip(1 - u0_probability, 1e-12, 1))
    if np.var(lp[update_mask]) <= 1e-12 or np.var(lp[evaluation_mask]) <= 1e-12:
        raise RuntimeError("U1/U2 requires nonconstant prediction logits in both halves")
    u1 = calibration_in_the_large(y[update_mask], u0_probability[update_mask])
    if not u1.converged:
        raise RuntimeError("U1 intercept-only update did not converge")
    u1_alpha = u1.estimate
    u2 = calibration_intercept_slope(y[update_mask], u0_probability[update_mask])
    if not u2.converged:
        raise RuntimeError(f"U2 intercept/slope update did not converge: {u2.message}")
    p_eval_u0 = u0_probability[evaluation_mask]
    p_eval_u1 = 1 / (1 + np.exp(-(lp[evaluation_mask] + u1_alpha)))
    p_eval_u2 = 1 / (1 + np.exp(-(u2.intercept + u2.slope * lp[evaluation_mask])))
    predictions = {"U0": p_eval_u0, "U1_intercept": p_eval_u1, "U2_intercept_slope": p_eval_u2}
    point_rows = []
    for name, p in predictions.items():
        result, _ = point_performance(y[evaluation_mask], p)
        result.update({"model_state": name})
        point_rows.append(result)
    point = pd.DataFrame(point_rows)
    point["update_n"] = int(update_mask.sum())
    point["evaluation_n"] = int(evaluation_mask.sum())
    point["U1_alpha_estimated_in_update_half"] = u1_alpha
    point["U2_alpha_estimated_in_update_half"] = u2.intercept
    point["U2_beta_estimated_in_update_half"] = u2.slope
    point["update_events"] = int(y[update_mask].sum())
    point["evaluation_events"] = int(y[evaluation_mask].sum())
    point["U1_update_fit_converged"] = bool(u1.converged)
    point["U2_update_fit_converged"] = bool(u2.converged)
    point["U2_update_fit_message"] = u2.message
    point["validity_status"] = "estimable"
    point.to_csv(
        DRIFT_OUT / "U1_U2_independent_update_evaluation.csv", index=False, encoding="utf-8-sig"
    )
    metrics = {
        "auroc": auroc,
        "auprc": auprc,
        "brier": brier_score,
        "log_loss": binary_log_loss,
        "calibration_slope": lambda yy, pp: calibration_intercept_slope(yy, pp).slope,
        "calibration_in_the_large": lambda yy, pp: calibration_in_the_large(yy, pp).estimate,
    }
    parts = []
    ids = mover.loc[evaluation_mask, "patient_id"].to_numpy()
    for metric_name, metric in metrics.items():
        result = paired_patient_bootstrap(
            y[evaluation_mask],
            predictions,
            ids,
            metric,
            n_boot=N_BOOT,
            random_state=SEED,
            comparisons=[("U1_intercept", "U0"), ("U2_intercept_slope", "U0")],
            return_replicates=False,
            min_valid_fraction=0.95,
        )
        table = result.summary.copy()
        table.insert(0, "metric", metric_name)
        parts.append(table)
    pd.concat(parts, ignore_index=True).to_csv(
        DRIFT_OUT / f"U1_U2_paired_bootstrap_{N_BOOT}.csv", index=False, encoding="utf-8-sig"
    )


def outcome_observability_ipw(
    primary_model: dict,
    primary_preprocess: dict,
    u0_main: pd.DataFrame,
) -> None:
    """Landmark-preinformation IPW with explicit target/overlap diagnostics.

    The IPW target is the first *pre-outcome-eligible* stage-2 operation per
    patient, which intentionally differs from U0's first fully evaluable
    operation rule.  That target difference and any later-operation
    replacement in U0 are quantified rather than hidden.
    """

    all_t0 = pd.read_csv(
        MOVER / "mover_all_t0_operations.csv.gz",
        dtype={"case_id": "string", "patient_id": "string"},
        low_memory=False,
    )
    denominator = all_t0[
        all_t0["feature_evaluable"].eq(1)
        & all_t0["stage1_evaluable"].eq(1)
        & all_t0["stage2_eligible"].eq(1)
    ].copy()
    denominator["_order"] = pd.to_datetime(denominator["anesthesia_start"], errors="coerce")
    denominator = (
        denominator.sort_values(["patient_id", "_order", "case_id"], kind="mergesort")
        .groupby("patient_id", sort=False)
        .head(1)
        .drop(columns="_order")
    )
    if denominator["patient_id"].duplicated().any() or denominator["case_id"].duplicated().any():
        raise ValueError("IPW target must contain one unique operation per patient")
    observed = denominator["outcome_evaluable"].eq(1).astype(int).to_numpy()
    if np.unique(observed).size != 2:
        raise RuntimeError("IPW selection model requires observed and unobserved outcomes")
    exclusion_percent = 100 * (1 - observed.mean())
    nuisance_medians = fit_imputation(denominator, FEATURE_COLUMNS)
    X = apply_imputation(
        denominator, FEATURE_COLUMNS, nuisance_medians, enforce_structural_gate=True
    )
    selection_model = fit_ridge_logistic(
        X, observed, l2=0.1, max_iter=300, tol=1e-9
    )
    if not selection_model.converged_:
        raise RuntimeError(f"IPW selection model did not converge: {selection_model.message_}")
    raw_propensity = selection_model.predict_proba(X)
    propensity = np.clip(raw_propensity, 0.01, 0.99)
    raw_weight = 1 / propensity[observed == 1]
    lower, upper = np.quantile(raw_weight, [0.01, 0.99])
    weight = np.clip(raw_weight, lower, upper)
    included = denominator.iloc[np.flatnonzero(observed == 1)].copy()
    y = pd.to_numeric(included["primary_outcome"], errors="raise").astype(int).to_numpy()
    p = frozen_primary_probability(included, primary_model, primary_preprocess)

    # Make the target-population change from U0 auditable at patient/case level.
    target_map = denominator.set_index("patient_id")["case_id"].astype(str)
    observed_map = included.set_index("patient_id")["case_id"].astype(str)
    u0_map = (
        u0_main.assign(
            patient_id=u0_main["patient_id"].astype(str),
            case_id=u0_main["case_id"].astype(str),
        )
        .set_index("patient_id")["case_id"]
    )
    if u0_map.index.duplicated().any():
        raise ValueError("U0 main cohort must contain at most one operation per patient")
    common_target_u0 = target_map.index.intersection(u0_map.index)
    same_case = target_map.loc[common_target_u0].eq(u0_map.loc[common_target_u0])
    target_unobserved_patients = set(denominator.loc[observed == 0, "patient_id"].astype(str))
    later_replacement_patients = {
        patient
        for patient in common_target_u0.astype(str)
        if patient in target_unobserved_patients and target_map.loc[patient] != u0_map.loc[patient]
    }
    pd.DataFrame(
        [
            {
                "IPW_target_definition": "first pre-outcome-eligible stage2 operation per patient",
                "U0_target_definition": "first fully evaluable operation per patient, then stage2",
                "IPW_target_patients": len(target_map),
                "IPW_target_outcome_observed_patients": len(observed_map),
                "U0_patients": len(u0_map),
                "IPW_target_and_U0_shared_patients": len(common_target_u0),
                "shared_patients_same_case": int(same_case.sum()),
                "shared_patients_different_case": int((~same_case).sum()),
                "unobserved_target_first_case_replaced_by_later_U0_case_patients": len(
                    later_replacement_patients
                ),
                "interpretation": "IPW is a sensitivity estimand and never replaces U0",
            }
        ]
    ).to_csv(
        DRIFT_OUT / "outcome_observability_target_overlap.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Covariate balance before/after weighting against the full IPW target.
    balance_rows = []
    observed_rows = observed == 1
    for index, feature in enumerate(FEATURE_COLUMNS):
        target_values = X[:, index]
        included_values = X[observed_rows, index]
        target_mean = float(np.mean(target_values))
        target_sd = float(np.std(target_values, ddof=1))
        included_mean = float(np.mean(included_values))
        included_sd = float(np.std(included_values, ddof=1))
        weighted_mean = float(np.average(included_values, weights=weight))
        weighted_variance = float(
            np.average((included_values - weighted_mean) ** 2, weights=weight)
        )
        weighted_sd = math.sqrt(max(weighted_variance, 0.0))
        pooled_unweighted = math.sqrt((target_sd**2 + included_sd**2) / 2)
        pooled_weighted = math.sqrt((target_sd**2 + weighted_sd**2) / 2)
        raw_feature = pd.to_numeric(denominator[feature], errors="coerce")
        balance_rows.append(
            {
                "feature": feature,
                "target_n": len(target_values),
                "observed_n": len(included_values),
                "target_raw_missing_percent": 100 * raw_feature.isna().mean(),
                "observed_raw_missing_percent": 100 * raw_feature[observed_rows].isna().mean(),
                "target_imputed_mean": target_mean,
                "observed_unweighted_imputed_mean": included_mean,
                "observed_IPW_imputed_mean": weighted_mean,
                "unweighted_SMD_observed_minus_target": (
                    (included_mean - target_mean) / pooled_unweighted
                    if pooled_unweighted > 0
                    else np.nan
                ),
                "IPW_SMD_observed_minus_target": (
                    (weighted_mean - target_mean) / pooled_weighted
                    if pooled_weighted > 0
                    else np.nan
                ),
            }
        )
    pd.DataFrame(balance_rows).to_csv(
        DRIFT_OUT / "outcome_observability_covariate_balance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    rows = []
    for analysis, sample_weight in [("unweighted_complete_outcome", None), ("IPW_1st_99th_truncated", weight)]:
        base = evaluate_binary_predictions(y, p, sample_weight=sample_weight).to_dict()
        citl = calibration_in_the_large(y, p, sample_weight=sample_weight)
        slope = calibration_intercept_slope(y, p, sample_weight=sample_weight)
        oe = oe_ratio_log_ci(y, p, sample_weight=sample_weight)
        ici = ici_equal_frequency(y, p, sample_weight=sample_weight)
        rows.append(
            {
                "analysis": analysis,
                "selection_denominator_n": len(denominator),
                "outcome_observed_n": int(observed.sum()),
                "outcome_unobserved_n": int((1 - observed).sum()),
                "outcome_unobserved_percent": exclusion_percent,
                "triggered_gt20_percent_protocol_gate": int(exclusion_percent > 20),
                "events": int(y.sum()),
                "event_rate": float(np.average(y, weights=sample_weight)),
                **base,
                "calibration_in_the_large": citl.estimate,
                "calibration_slope": slope.slope,
                "observed_expected_ratio": oe.ratio,
                "ici_equal_frequency": ici.ici,
                "weight_truncation_lower": lower if sample_weight is not None else np.nan,
                "weight_truncation_upper": upper if sample_weight is not None else np.nan,
                "effective_sample_size": (
                    float(weight.sum() ** 2 / np.sum(weight**2))
                    if sample_weight is not None
                    else len(y)
                ),
                "selection_model_l2": 0.1,
                "selection_model_converged": selection_model.converged_,
                "target_population": "first pre-outcome-eligible stage2 operation per patient",
                "IPW_replaces_U0": 0,
            }
        )
    pd.DataFrame(rows).to_csv(
        DRIFT_OUT / "outcome_observability_IPW_sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        [
            {
                "selection_denominator_n": len(denominator),
                "observed_n": int(observed.sum()),
                "unobserved_n": int((1 - observed).sum()),
                "unobserved_percent": exclusion_percent,
                "selection_model_auroc": auroc(observed, raw_propensity),
                "selection_model_converged": selection_model.converged_,
                "selection_model_iterations": selection_model.n_iter_,
                "raw_propensity_min": float(raw_propensity.min()),
                "raw_propensity_p01": float(np.quantile(raw_propensity, 0.01)),
                "raw_propensity_p99": float(np.quantile(raw_propensity, 0.99)),
                "raw_propensity_max": float(raw_propensity.max()),
                "propensity_min": float(propensity.min()),
                "propensity_median": float(np.median(propensity)),
                "propensity_max": float(propensity.max()),
                "propensity_clipped_low_n": int((raw_propensity < 0.01).sum()),
                "propensity_clipped_high_n": int((raw_propensity > 0.99).sum()),
                "observed_propensity_median": float(np.median(propensity[observed == 1])),
                "unobserved_propensity_median": float(np.median(propensity[observed == 0])),
                "raw_weight_p01": lower,
                "raw_weight_p99": upper,
                "truncated_weight_min": float(weight.min()),
                "truncated_weight_median": float(np.median(weight)),
                "truncated_weight_max": float(weight.max()),
                "truncated_weight_coefficient_of_variation": float(
                    np.std(weight, ddof=1) / np.mean(weight)
                ),
                "effective_sample_size": float(weight.sum() ** 2 / np.sum(weight**2)),
            }
        ]
    ).to_csv(
        DRIFT_OUT / "outcome_observability_gate_and_weight_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )


def early_vasopressor_descriptive(
    mover: pd.DataFrame,
    probability: np.ndarray,
) -> None:
    """Describe recorded systemic vasopressor administrations during [t0,t0+5]."""

    case_time = mover[["case_id", "patient_id", "anesthesia_start", "t0_min"]].copy()
    case_time["case_id"] = case_time["case_id"].astype(str)
    case_time["anesthesia_start"] = pd.to_datetime(case_time["anesthesia_start"], errors="coerce")
    case_index = case_time.set_index("case_id")
    if case_index.index.duplicated().any():
        raise ValueError("early-treatment mapping requires unique main-cohort case IDs")
    wanted_ids = set(case_index.index)
    retained = []
    raw_rows = 0
    candidate_rows = 0
    candidate_unparseable_med_time_rows = 0
    candidate_missing_anesthesia_start_rows = 0
    with tarfile.open(PATHS["mover_epic_emr"], "r:gz") as archive:
        handle = archive.extractfile("EPIC_EMR/EMR/patient_medications.csv")
        if handle is None:
            raise FileNotFoundError("MOVER patient_medications.csv")
        usecols = [
            "LOG_ID",
            "DISPLAY_NAME",
            "MEDICATION_NM",
            "RECORD_TYPE",
            "MAR_ACTION_NM",
            "MED_ACTION_TIME",
            "MED_ROUTE_NM",
        ]
        for chunk in pd.read_csv(handle, usecols=usecols, chunksize=500_000, low_memory=False):
            raw_rows += len(chunk)
            chunk["LOG_ID"] = chunk["LOG_ID"].astype(str)
            chunk = chunk[chunk["LOG_ID"].isin(wanted_ids)].copy()
            if chunk.empty:
                continue
            text = (
                chunk["DISPLAY_NAME"].fillna("").astype(str)
                + " "
                + chunk["MEDICATION_NM"].fillna("").astype(str)
            )
            route = chunk["MED_ROUTE_NM"].fillna("").astype(str)
            action = chunk["MAR_ACTION_NM"].fillna("").astype(str)
            keep = (
                text.str.contains(VASOPRESSOR_PATTERN, na=False)
                & ~text.str.contains(VASOPRESSOR_EXCLUDE, na=False)
                & route.str.contains("intravenous|IV", case=False, regex=True, na=False)
                & action.str.contains(
                    "given|new bag|bolus|rate change|rate verify|restarted",
                    case=False,
                    regex=True,
                    na=False,
                )
            )
            chunk = chunk[keep].copy()
            if chunk.empty:
                continue
            candidate_rows += len(chunk)
            action_lower = chunk["MAR_ACTION_NM"].fillna("").astype(str).str.lower()
            chunk["action_category"] = np.select(
                [
                    action_lower.str.contains("rate verify", regex=False),
                    action_lower.str.contains("rate change", regex=False),
                    action_lower.str.contains(
                        "given|new bag|bolus|restarted", regex=True, na=False
                    ),
                ],
                ["rate_verify_only", "rate_change", "new_or_bolus_administration"],
                default="other_candidate_action",
            )
            text_lower = text.loc[chunk.index].str.lower()
            chunk["vasopressor_agent"] = np.select(
                [
                    text_lower.str.contains("phenylephrine|neo[- ]?synephrine", regex=True),
                    text_lower.str.contains("ephedrine", regex=False),
                    text_lower.str.contains("norepinephrine|levophed", regex=True),
                    text_lower.str.contains(r"\bepinephrine\b|adrenaline", regex=True),
                    text_lower.str.contains("vasopressin", regex=False),
                    text_lower.str.contains(r"\bdopamine\b", regex=True),
                    text_lower.str.contains(r"\bdobutamine\b", regex=True),
                ],
                [
                    "phenylephrine",
                    "ephedrine",
                    "norepinephrine",
                    "epinephrine",
                    "vasopressin",
                    "dopamine",
                    "dobutamine",
                ],
                default="unresolved_systemic_vasopressor",
            )
            chunk["med_time"] = pd.to_datetime(chunk["MED_ACTION_TIME"], errors="coerce")
            chunk = chunk.join(case_index[["anesthesia_start", "t0_min"]], on="LOG_ID")
            candidate_unparseable_med_time_rows += int(chunk["med_time"].isna().sum())
            candidate_missing_anesthesia_start_rows += int(chunk["anesthesia_start"].isna().sum())
            chunk["minutes_from_t0"] = (
                chunk["med_time"] - chunk["anesthesia_start"]
            ).dt.total_seconds() / 60 - chunk["t0_min"]
            retained.append(chunk[chunk["minutes_from_t0"].between(0, 5, inclusive="both")])
    events = pd.concat(retained, ignore_index=True) if retained else pd.DataFrame(columns=["LOG_ID"])
    confirmed_events = (
        events[
            events["action_category"].isin(
                ["new_or_bolus_administration", "rate_change"]
            )
        ].copy()
        if not events.empty
        else events.copy()
    )
    confirmed_ids = (
        set(confirmed_events["LOG_ID"].astype(str)) if not confirmed_events.empty else set()
    )
    verify_ids = (
        set(events.loc[events["action_category"].eq("rate_verify_only"), "LOG_ID"].astype(str))
        if not events.empty
        else set()
    )
    case_ids = mover["case_id"].astype(str)
    confirmed_flag = case_ids.isin(confirmed_ids).to_numpy()
    verify_only_flag = case_ids.isin(verify_ids - confirmed_ids).to_numpy()
    no_candidate_flag = ~(confirmed_flag | verify_only_flag)
    y_full = pd.to_numeric(mover["primary_outcome"]).astype(int).to_numpy()
    rows = []
    bootstrap_parts = []
    for label, mask in [
        ("confirmed_new_bolus_or_rate_change_0_5", confirmed_flag),
        ("rate_verify_only_without_confirmed_action_0_5", verify_only_flag),
        ("no_candidate_vasopressor_action_recorded_0_5", no_candidate_flag),
    ]:
        y = y_full[mask]
        if len(y) and y.sum() >= 20 and len(y) - y.sum() >= 20:
            result, bootstrap = performance_with_patient_ci(
                y,
                probability[mask],
                mover.loc[mask, "patient_id"].to_numpy(),
                label=f"early_vasopressor|{label}",
            )
            bootstrap.insert(0, "treatment_stratum", label)
            bootstrap_parts.append(bootstrap)
        else:
            result = {
                "n": len(y),
                "events": int(y.sum()),
                "estimability_status": "nonestimable",
                "nonestimable_reason": "fewer than 20 events or 20 non-events",
            }
        result.update(
            {
                "treatment_stratum": label,
                "interpretation": "descriptive recorded treatment only; never causal",
            }
        )
        rows.append(result)
    pd.DataFrame(rows).to_csv(
        DRIFT_OUT / "early_vasopressor_descriptive_performance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if bootstrap_parts:
        pd.concat(bootstrap_parts, ignore_index=True).to_csv(
            DRIFT_OUT / f"early_vasopressor_patient_bootstrap_{N_BOOT}.csv",
            index=False,
            encoding="utf-8-sig",
        )
    if not events.empty:
        action_audit = (
            events.groupby("action_category", dropna=False)
            .agg(action_rows=("LOG_ID", "size"), cases=("LOG_ID", "nunique"))
            .reset_index()
        )
        agent_audit = (
            events.groupby("vasopressor_agent", dropna=False)
            .agg(action_rows=("LOG_ID", "size"), cases=("LOG_ID", "nunique"))
            .reset_index()
        )
    else:
        action_audit = pd.DataFrame(
            columns=["action_category", "action_rows", "cases"]
        )
        agent_audit = pd.DataFrame(
            columns=["vasopressor_agent", "action_rows", "cases"]
        )
    action_audit.to_csv(
        DRIFT_OUT / "early_vasopressor_action_category_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    agent_audit.to_csv(
        DRIFT_OUT / "early_vasopressor_agent_mapping_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        [
            {
                "medication_rows_scanned": raw_rows,
                "systemic_IV_vasopressor_candidate_rows_in_main_cases": candidate_rows,
                "candidate_rows_with_unparseable_med_time": candidate_unparseable_med_time_rows,
                "candidate_rows_with_missing_case_anesthesia_start": candidate_missing_anesthesia_start_rows,
                "candidate_action_rows_during_t0_to_t0_plus_5": len(events),
                "confirmed_new_bolus_or_rate_change_rows": len(confirmed_events),
                "confirmed_action_cases": len(confirmed_ids),
                "rate_verify_only_without_confirmed_action_cases": len(verify_ids - confirmed_ids),
                "no_candidate_action_cases": int(no_candidate_flag.sum()),
                "rate_verify_is_treatment_exposure": 0,
                "classification": "descriptive recorded treatment; not a causal contrast",
            }
        ]
    ).to_csv(
        DRIFT_OUT / "early_vasopressor_mapping_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )


def main() -> None:
    lock = json.loads((ROOT / "10_run_logs_manifest/U0_execution_lock.json").read_text(encoding="utf-8"))
    if lock.get("status") != "COMPLETE":
        raise SystemExit("Secondary analyses require a completed, locked U0")
    for directory in [R1_OUT, OBS_OUT, DRIFT_OUT]:
        directory.mkdir(parents=True, exist_ok=True)
    models, preprocess, _ = load_bundle(ROOT / "04_frozen_INSPIRE_LM5_model")
    primary_name = models["primary_model"]
    primary_model = models["models"][primary_name]
    primary_preprocess = preprocess["models"][primary_name]
    mover = pd.read_csv(MOVER / "mover_main_stage2.csv.gz", low_memory=False)
    inspire = pd.read_csv(INSPIRE / "inspire_main_stage2.csv.gz", low_memory=False)
    u0_probability = frozen_primary_probability(mover, primary_model, primary_preprocess)

    log("Running fixed-patient/t0 H5/R1 observation 2x2")
    observation_2x2(primary_model, primary_preprocess, mover[["case_id", "patient_id"]])

    source_cohort = pd.read_csv(MOVER / "mover_basic_common_cohort_all_operations.csv.gz", low_memory=False)
    cases = canonical_cases(source_cohort)
    source_map = pd.read_csv(
        MOVER / "mover_observed_map_20_200.csv.gz",
        usecols=["encounter_id", "minute_from_anesthesia_start", "map_value", "map_source"],
        low_memory=False,
    )
    raw = canonical_source_map(source_map)
    raw_formal = raw[raw["source"].isin(["ART", "NIBP"])].copy()
    del source_map, raw
    gc.collect()
    h5 = pd.read_csv(MOVER / "mover_H5_grid.csv.gz", low_memory=False)
    r1 = pd.read_csv(MOVER / "mover_R1_grid.csv.gz", low_memory=False)

    log("Running phase, induction-buffer and R1-redetected-t0 sensitivities")
    sensitivity_cohorts(cases, raw_formal, h5, r1, primary_model, primary_preprocess)
    all_surgeries_patient_cluster_sensitivity(mover, primary_model, primary_preprocess)
    endpoint_sensitivities(mover, h5, primary_model, primary_preprocess)
    del raw_formal, r1, cases
    gc.collect()

    log("Running drift, subgroup and independent U1/U2 analyses")
    drift_and_subgroups(inspire, mover, u0_probability)
    update_analysis(mover, u0_probability)
    outcome_observability_ipw(primary_model, primary_preprocess, mover)
    early_vasopressor_descriptive(mover, u0_probability)
    log("Secondary analyses complete")


if __name__ == "__main__":
    main()

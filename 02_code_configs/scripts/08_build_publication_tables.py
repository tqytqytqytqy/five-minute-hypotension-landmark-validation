#!/usr/bin/env python3
"""Build publication-ready CSV tables from locked cohorts and U0 outputs.

This script is deliberately post-estimation only.  It never fits a model,
generates predictions, or writes under 04_frozen_INSPIRE_LM5_model or
05_MOVER_validation.  The only outputs are three manuscript tables and their
source-lineage CSV under 06_tables.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INSPIRE_MAIN = ROOT / "03_derived_cohorts/INSPIRE/inspire_main_stage2.csv.gz"
MOVER_MAIN = ROOT / "03_derived_cohorts/MOVER/mover_main_stage2.csv.gz"
MOVER_FIRST = (
    ROOT
    / "03_derived_cohorts/MOVER/mover_first_fully_evaluable_operation.csv.gz"
)
U0 = ROOT / "05_MOVER_validation/H5_primary"
U0_POINT = U0 / "U0_external_validation_point_estimates.csv"
U0_BOOT_SUMMARY = U0 / "U0_paired_patient_bootstrap_2000_summary.csv"
U0_BOOT_REPLICATES = U0 / "U0_paired_patient_bootstrap_2000_replicates.csv.gz"
U0_THRESHOLDS = U0 / "U0_fixed_thresholds_applied.csv"
U0_WORKLOAD = U0 / "U0_fixed_threshold_alert_workload.csv"
U0_NET_BENEFIT = U0 / "U0_fixed_threshold_net_benefit.csv"
U0_TWO_STAGE_SUMMARY = U0 / "U0_two_stage_fixed_strategy_bootstrap_2000_summary.csv"

TABLE_DIR = ROOT / "06_tables"
TABLE1 = TABLE_DIR / "Table1_cohort_characteristics.csv"
TABLE2 = TABLE_DIR / "Table2_U0_external_validation_with_95CI.csv"
TABLE3 = TABLE_DIR / "Table3_fixed_threshold_clinical_utility.csv"
LINEAGE = TABLE_DIR / "publication_tables_source_lineage.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False, **kwargs)


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def binary(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    mapped = (
        series.astype("string")
        .str.strip()
        .str.lower()
        .map({"true": 1.0, "false": 0.0, "yes": 1.0, "no": 0.0})
    )
    parsed = numeric(series)
    return parsed.where(parsed.notna(), mapped)


def continuous_smd(left: pd.Series, right: pd.Series) -> float:
    left = numeric(left).dropna()
    right = numeric(right).dropna()
    if len(left) < 2 or len(right) < 2:
        return math.nan
    denominator = math.sqrt((left.var(ddof=1) + right.var(ddof=1)) / 2.0)
    if not math.isfinite(denominator) or denominator <= 0:
        return math.nan
    return abs(float(left.mean() - right.mean())) / denominator


def binary_smd(left: pd.Series, right: pd.Series) -> float:
    left = binary(left).dropna()
    right = binary(right).dropna()
    if len(left) == 0 or len(right) == 0:
        return math.nan
    p_left = float(left.mean())
    p_right = float(right.mean())
    denominator = math.sqrt(
        (p_left * (1.0 - p_left) + p_right * (1.0 - p_right)) / 2.0
    )
    if not math.isfinite(denominator) or denominator <= 0:
        return 0.0 if p_left == p_right else math.nan
    return abs(p_left - p_right) / denominator


def fmt_mean_sd(series: pd.Series, digits: int = 1) -> str:
    values = numeric(series).dropna()
    return f"{values.mean():.{digits}f} ({values.std(ddof=1):.{digits}f})"


def fmt_n_percent(series: pd.Series, digits: int = 1) -> str:
    values = binary(series).dropna()
    count = int(values.eq(1).sum())
    percent = 100.0 * count / len(values) if len(values) else math.nan
    return f"{count:,} ({percent:.{digits}f}%)"


def fmt_number(value: object, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "not estimable"
    if not math.isfinite(number):
        return "not estimable"
    return f"{number:.{digits}f}"


def fmt_estimate_ci(
    estimate: object,
    lower: object,
    upper: object,
    *,
    digits: int = 3,
) -> str:
    values = []
    for value in (estimate, lower, upper):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "not estimable"
        if not math.isfinite(number):
            return "not estimable"
        values.append(number)
    values = [0.0 if round(value, digits) == 0 else value for value in values]
    return (
        f"{values[0]:.{digits}f} "
        f"({values[1]:.{digits}f} to {values[2]:.{digits}f})"
    )


def percentile_summary(values: pd.Series) -> dict[str, float | int]:
    values = numeric(values)
    valid = values[np.isfinite(values.to_numpy(dtype=float))]
    if len(valid) == 0:
        return {
            "estimate": math.nan,
            "ci95_lower": math.nan,
            "ci95_upper": math.nan,
            "valid_replicates": 0,
        }
    return {
        "estimate": float(valid.mean()),
        "ci95_lower": float(np.percentile(valid, 2.5)),
        "ci95_upper": float(np.percentile(valid, 97.5)),
        "valid_replicates": int(len(valid)),
    }


def build_table1(inspire: pd.DataFrame, mover: pd.DataFrame) -> pd.DataFrame:
    for label, frame in (("INSPIRE", inspire), ("MOVER", mover)):
        if frame["patient_id"].isna().any() or frame["case_id"].isna().any():
            raise ValueError(f"{label}: missing patient or case identifiers")
        if frame["patient_id"].duplicated().any():
            raise ValueError(f"{label}: main cohort is not one operation per patient")

    rows: list[dict[str, object]] = []

    def common_fields(
        section: str,
        characteristic: str,
        level: str,
        statistic: str,
        left: pd.Series,
        right: pd.Series,
        left_value: str,
        right_value: str,
        smd: float,
        smd_method: str,
    ) -> dict[str, object]:
        left_missing = int(left.isna().sum())
        right_missing = int(right.isna().sum())
        return {
            "section": section,
            "characteristic": characteristic,
            "level": level,
            "summary_statistic": statistic,
            "INSPIRE_nonmissing_n": int(left.notna().sum()),
            "INSPIRE_value": left_value,
            "INSPIRE_missing_n": left_missing,
            "INSPIRE_missing_percent": round(100.0 * left_missing / len(left), 1),
            "MOVER_nonmissing_n": int(right.notna().sum()),
            "MOVER_value": right_value,
            "MOVER_missing_n": right_missing,
            "MOVER_missing_percent": round(100.0 * right_missing / len(right), 1),
            "absolute_SMD": (
                f"{float(smd):.3f}"
                if math.isfinite(float(smd))
                else "not estimable"
            ),
            "SMD_method": smd_method,
        }

    rows.append(
        {
            "section": "Cohort",
            "characteristic": "Participants",
            "level": "",
            "summary_statistic": "n",
            "INSPIRE_nonmissing_n": len(inspire),
            "INSPIRE_value": f"{len(inspire):,}",
            "INSPIRE_missing_n": 0,
            "INSPIRE_missing_percent": 0.0,
            "MOVER_nonmissing_n": len(mover),
            "MOVER_value": f"{len(mover):,}",
            "MOVER_missing_n": 0,
            "MOVER_missing_percent": 0.0,
            "absolute_SMD": "not applicable",
            "SMD_method": "not applicable",
        }
    )

    def add_continuous(
        section: str,
        label: str,
        column: str,
        unit: str,
        digits: int = 1,
    ) -> None:
        left = numeric(inspire[column])
        right = numeric(mover[column])
        rows.append(
            common_fields(
                section,
                label,
                unit,
                "Mean (SD)",
                left,
                right,
                fmt_mean_sd(left, digits),
                fmt_mean_sd(right, digits),
                continuous_smd(left, right),
                "absolute difference in means divided by pooled SD",
            )
        )

    def add_binary(
        section: str,
        label: str,
        column: str,
        level: str = "Yes",
    ) -> None:
        left = binary(inspire[column])
        right = binary(mover[column])
        rows.append(
            common_fields(
                section,
                label,
                level,
                "n (%)",
                left,
                right,
                fmt_n_percent(left),
                fmt_n_percent(right),
                binary_smd(left, right),
                "absolute standardized difference in proportions",
            )
        )

    add_binary("Cohort", "Primary outcome", "primary_outcome", "Present")
    add_continuous("Demographics", "Age", "age_years", "years")
    add_binary("Demographics", "Sex", "male", "Male")
    add_continuous("Demographics", "Body mass index", "bmi", "kg/m^2")

    inspire_asa = numeric(inspire["asa"])
    mover_asa = numeric(mover["asa"])
    asa_levels = [
        ("I", inspire_asa.eq(1), mover_asa.eq(1)),
        ("II", inspire_asa.eq(2), mover_asa.eq(2)),
        ("III", inspire_asa.eq(3), mover_asa.eq(3)),
        ("IV or higher", inspire_asa.ge(4), mover_asa.ge(4)),
    ]
    for level, left_indicator, right_indicator in asa_levels:
        left_indicator = left_indicator.astype(float).where(inspire_asa.notna())
        right_indicator = right_indicator.astype(float).where(mover_asa.notna())
        rows.append(
            common_fields(
                "Demographics",
                "ASA physical status",
                level,
                "n (%)",
                left_indicator,
                right_indicator,
                fmt_n_percent(left_indicator),
                fmt_n_percent(right_indicator),
                binary_smd(left_indicator, right_indicator),
                "absolute standardized difference in category proportions",
            )
        )

    inspire_emergency = binary(inspire["emergency"])
    # MOVER's harmonized source lacks a reliable emergency-status field.  The
    # technical False placeholder must not be interpreted as a true 0% rate.
    rows.append(
        {
            "section": "Operation",
            "characteristic": "Emergency operation",
            "level": "Yes",
            "summary_statistic": "n (%)",
            "INSPIRE_nonmissing_n": int(inspire_emergency.notna().sum()),
            "INSPIRE_value": fmt_n_percent(inspire_emergency),
            "INSPIRE_missing_n": int(inspire_emergency.isna().sum()),
            "INSPIRE_missing_percent": round(
                100.0 * inspire_emergency.isna().sum() / len(inspire_emergency), 1
            ),
            "MOVER_nonmissing_n": 0,
            "MOVER_value": "not available (structurally unavailable)",
            "MOVER_missing_n": len(mover),
            "MOVER_missing_percent": 100.0,
            "absolute_SMD": "not estimable",
            "SMD_method": (
                "not estimable because MOVER emergency status is structurally unavailable"
            ),
            "data_availability_note": (
                "MOVER's technical False placeholder was not interpreted as observed "
                "non-emergency status"
            ),
        }
    )
    add_continuous(
        "Operation",
        "Anaesthesia duration",
        "anesthesia_duration_min",
        "min",
    )
    add_binary(
        "Index hypotension",
        "Arterial-line MAP selected at t0",
        "t0_arterial_source",
        "Yes",
    )
    add_continuous(
        "Index hypotension",
        "Time from anaesthesia start to t0",
        "anesthesia_start_to_t0_min",
        "min",
    )
    add_continuous("Index hypotension", "MAP at t0", "t0_map", "mm Hg")
    add_continuous(
        "Pre-t0 MAP history",
        "MAP records in preceding 10 min",
        "pre10_map_record_count",
        "count",
    )
    add_continuous(
        "Pre-t0 MAP history",
        "Gap from last MAP to t0",
        "pre10_last_measurement_gap_min",
        "min",
    )
    add_continuous(
        "Pre-t0 MAP history", "Last MAP", "pre10_last_map", "mm Hg"
    )
    add_continuous(
        "Pre-t0 MAP history", "Mean MAP", "pre10_mean_map", "mm Hg"
    )
    add_continuous(
        "Pre-t0 MAP history",
        "OLS MAP slope",
        "pre10_map_ols_slope_per_min",
        "mm Hg/min",
        digits=2,
    )
    add_binary(
        "Early response (0-5 min)",
        "Recovered by 5 min",
        "recovered_by_5min",
        "Yes",
    )
    add_continuous(
        "Early response (0-5 min)",
        "AUC below 65 mm Hg",
        "early_auc65_0_5_mmhg_min",
        "mm Hg x min",
    )
    add_continuous(
        "Early response (0-5 min)",
        "Minimum MAP",
        "early_min_map_0_5",
        "mm Hg",
    )
    add_continuous(
        "Early response (0-5 min)",
        "Mean MAP",
        "early_mean_map_0_5",
        "mm Hg",
    )
    add_continuous(
        "Early response (0-5 min)",
        "MAP records",
        "early_map_record_count_0_5",
        "count",
    )

    table = pd.DataFrame(rows)
    table["data_availability_note"] = table["data_availability_note"].fillna("")
    if any("p_value" in column.lower() or column.lower() == "p" for column in table):
        raise AssertionError("Table 1 must not contain P values")
    return table


def build_table2(point: pd.DataFrame, bootstrap: pd.DataFrame) -> pd.DataFrame:
    point = point.set_index("model", drop=False)
    bootstrap = bootstrap.set_index("estimand", drop=False)
    models = {
        "recovered_by_5min": "simple_recovered_by_5min",
        "early_mean_MAP": "simple_early_mean_map",
        "t0_MAP": "simple_t0_map",
    }
    metrics = [
        ("AUROC", "auroc", "higher is better"),
        ("AUPRC", "auprc", "higher is better"),
        ("Brier score", "brier", "lower is better"),
        ("Scaled Brier score", "scaled_brier", "higher is better"),
        ("Log loss", "log_loss", "lower is better"),
        ("Calibration slope", "calibration_slope", "target is 1"),
        (
            "Calibration-in-the-large",
            "calibration_in_the_large",
            "target is 0",
        ),
        ("Observed-to-expected ratio", "observed_expected_ratio", "target is 1"),
        ("Integrated calibration index", "ici_equal_frequency", "lower is better"),
    ]

    if (set(models.values()) | {"LM5_common18"}) - set(point.index):
        raise ValueError("U0 point-estimate file is missing a required model")
    lm5 = point.loc["LM5_common18"]
    n = int(lm5["n"])
    events = int(lm5["events"])
    event_rate = float(lm5["event_rate"])
    rows = []

    for metric_label, metric, direction in metrics:
        lm5_key = f"LM5_common18__{metric}"
        if lm5_key not in bootstrap.index:
            raise ValueError(f"missing bootstrap estimand: {lm5_key}")
        lm5_boot = bootstrap.loc[lm5_key]
        if not np.isclose(float(lm5[metric]), float(lm5_boot["estimate"]), atol=1e-12):
            raise AssertionError(f"point/bootstrap mismatch for {lm5_key}")

        row: dict[str, object] = {
            "metric": metric_label,
            "preferred_direction": direction,
            "n": n,
            "events": events,
            "event_rate": round(event_rate, 4),
            "LM5_common18_estimate_95CI": fmt_estimate_ci(
                lm5[metric], lm5_boot["ci95_lower"], lm5_boot["ci95_upper"]
            ),
            "LM5_valid_bootstrap_replicates": int(lm5_boot["valid_replicates"]),
        }
        for display, model in models.items():
            model_key = f"{model}__{metric}"
            delta_key = f"delta__LM5_common18_minus_{model}__{metric}"
            if model_key not in bootstrap.index or delta_key not in bootstrap.index:
                raise ValueError(f"missing paired bootstrap estimand for {model}/{metric}")
            model_boot = bootstrap.loc[model_key]
            delta_boot = bootstrap.loc[delta_key]
            if not np.isclose(
                float(point.loc[model, metric]),
                float(model_boot["estimate"]),
                atol=1e-12,
            ):
                raise AssertionError(f"point/bootstrap mismatch for {model_key}")
            row[f"{display}_only_estimate_95CI"] = fmt_estimate_ci(
                point.loc[model, metric],
                model_boot["ci95_lower"],
                model_boot["ci95_upper"],
            )
            row[f"LM5_minus_{display}_paired_difference_95CI"] = fmt_estimate_ci(
                delta_boot["estimate"],
                delta_boot["ci95_lower"],
                delta_boot["ci95_upper"],
            )
            row[f"{display}_paired_difference_valid_replicates"] = int(
                delta_boot["valid_replicates"]
            )
        row["CI_method"] = (
            "percentile 95% CI; 2000 non-stratified patient-level paired "
            "bootstrap replicates"
        )
        row["paired_difference_definition"] = "LM5_common18 minus simple model"
        rows.append(row)

    return pd.DataFrame(rows)


def stage2_bootstrap_derived(replicates: pd.DataFrame) -> dict[float, dict[str, dict]]:
    # The t0-only model alerts everyone at pt=0.10, so its NB is the treat-all
    # NB.  This identifies event prevalence in each locked bootstrap replicate
    # without resampling, refitting, or regenerating predictions.
    all_nb = numeric(replicates["simple_t0_map__pt_0.100000__net_benefit"])
    odds_010 = 0.10 / 0.90
    prevalence = (all_nb + odds_010) / (1.0 + odds_010)

    thresholds = [0.10, 0.15, 0.20, 0.25, 0.25561590104176246, 0.3033134554600324, 0.39022485751038916]
    derived: dict[float, dict[str, dict]] = {}
    for threshold in thresholds:
        token = f"{threshold:.6f}"
        prefix = f"LM5_common18__pt_{token}__"
        alert_rate = numeric(replicates[prefix + "alert_rate"])
        sensitivity = numeric(replicates[prefix + "sensitivity"])
        true_positive_rate = sensitivity * prevalence
        false_positive_rate = alert_rate - true_positive_rate
        specificity = 1.0 - false_positive_rate / (1.0 - prevalence)
        values: dict[str, pd.Series] = {
            "alerts_per_1000": alert_rate * 1000.0,
            "true_positive_alerts_per_1000": true_positive_rate * 1000.0,
            "false_alerts_per_1000": false_positive_rate * 1000.0,
            "specificity": specificity,
        }
        if threshold in {0.10, 0.15, 0.20, 0.25}:
            model_nb = numeric(replicates[prefix + "net_benefit"])
            all_nb_at_threshold = prevalence - (1.0 - prevalence) * threshold / (
                1.0 - threshold
            )
            values["net_interventions_avoided_per_100_vs_all"] = (
                (model_nb - all_nb_at_threshold)
                * 100.0
                * (1.0 - threshold)
                / threshold
            )
        derived[threshold] = {
            metric: percentile_summary(series) for metric, series in values.items()
        }
    return derived


def build_table3(
    mover_main: pd.DataFrame,
    mover_first: pd.DataFrame,
    thresholds: pd.DataFrame,
    workload: pd.DataFrame,
    net_benefit: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    bootstrap_replicates: pd.DataFrame,
    two_stage_summary: pd.DataFrame,
) -> pd.DataFrame:
    if mover_main["patient_id"].duplicated().any() or mover_first[
        "patient_id"
    ].duplicated().any():
        raise ValueError("MOVER publication cohorts must contain one row per patient")

    stage2_n = len(mover_main)
    stage2_events = int(numeric(mover_main["primary_outcome"]).sum())
    all_n = len(mover_first)
    all_events = int(numeric(mover_first["primary_outcome"]).sum())
    if (stage2_n, stage2_events) != (7177, 1398):
        raise AssertionError("unexpected locked MOVER Stage 2 cohort counts")
    if (all_n, all_events) != (8595, 2248):
        raise AssertionError("unexpected locked MOVER two-stage cohort counts")

    expected_names = {
        "clinical_action_sensitivity_0.10",
        "clinical_action_primary",
        "clinical_action_sensitivity_0.20",
        "clinical_action_sensitivity_0.25",
        "capacity_top_30_percent",
        "capacity_top_20_percent",
        "capacity_top_10_percent",
    }
    if set(thresholds["threshold_name"]) != expected_names:
        raise AssertionError("fixed-threshold contract differs from the expected seven values")

    threshold_order = [
        "clinical_action_sensitivity_0.10",
        "clinical_action_primary",
        "clinical_action_sensitivity_0.20",
        "clinical_action_sensitivity_0.25",
        "capacity_top_30_percent",
        "capacity_top_20_percent",
        "capacity_top_10_percent",
    ]
    threshold_map = thresholds.set_index("threshold_name")
    workload_lm5 = workload[workload["model"].eq("LM5_common18")].set_index(
        "threshold_name"
    )
    net_lm5 = net_benefit[net_benefit["model"].eq("LM5_common18")].set_index(
        "threshold_name"
    )
    boot_map = bootstrap_summary.set_index("estimand")
    two_map = two_stage_summary.set_index(["threshold_name", "metric"])
    derived = stage2_bootstrap_derived(bootstrap_replicates)

    metric_digits = {
        "alert_rate": 3,
        "alerts_per_1000": 1,
        "true_positive_alerts_per_1000": 1,
        "false_alerts_per_1000": 1,
        "sensitivity": 3,
        "specificity": 3,
        "positive_predictive_value": 3,
        "alerts_per_true_positive": 2,
        "false_alerts_per_true_positive": 2,
        "net_benefit": 3,
        "net_interventions_avoided_per_100_vs_all": 1,
    }

    def stage2_ci(threshold: float, metric: str) -> tuple[object, object, int]:
        if metric in {
            "alerts_per_1000",
            "true_positive_alerts_per_1000",
            "false_alerts_per_1000",
            "specificity",
            "net_interventions_avoided_per_100_vs_all",
        }:
            derived_threshold = min(derived, key=lambda value: abs(value - threshold))
            if abs(derived_threshold - threshold) > 1e-9:
                raise AssertionError(f"bootstrap threshold not found: {threshold}")
            summary = derived[derived_threshold].get(metric)
            if summary is None:
                return math.nan, math.nan, 0
            return (
                summary["ci95_lower"],
                summary["ci95_upper"],
                int(summary["valid_replicates"]),
            )
        source_metric = {
            "positive_predictive_value": "ppv",
            "net_benefit": "net_benefit",
        }.get(metric, metric)
        key = f"LM5_common18__pt_{threshold:.6f}__{source_metric}"
        if key not in boot_map.index:
            return math.nan, math.nan, 0
        row = boot_map.loc[key]
        return row["ci95_lower"], row["ci95_upper"], int(row["valid_replicates"])

    stage2_point_columns = {
        "alert_rate": "alert_rate",
        "alerts_per_1000": "alerts_per_1000",
        "true_positive_alerts_per_1000": "true_positive_alerts_per_1000",
        "false_alerts_per_1000": "false_alerts_per_1000",
        "sensitivity": "sensitivity",
        "specificity": "specificity",
        "positive_predictive_value": "positive_predictive_value",
        "alerts_per_true_positive": "alerts_per_true_positive",
        "false_alerts_per_true_positive": "false_alerts_per_true_positive",
    }
    two_stage_metric_names = {
        "alert_rate": "alert_rate",
        "alerts_per_1000": "alerts_per_1000",
        "true_positive_alerts_per_1000": "true_positive_alerts_per_1000",
        "false_alerts_per_1000": "false_alerts_per_1000",
        "sensitivity": "sensitivity",
        "specificity": "specificity",
        "positive_predictive_value": "positive_predictive_value",
        "alerts_per_true_positive": "alerts_per_true_positive",
        "false_alerts_per_true_positive": "false_alerts_per_true_positive",
        "net_benefit": "fixed_binary_strategy_net_benefit",
        "net_interventions_avoided_per_100_vs_all": (
            "net_interventions_avoided_per_100_vs_all"
        ),
    }

    rows: list[dict[str, object]] = []
    for threshold_name in threshold_order:
        threshold_row = threshold_map.loc[threshold_name]
        threshold = float(threshold_row["threshold"])
        threshold_group = (
            "Clinical action threshold"
            if str(threshold_row["threshold_type"]).startswith("prespecified_clinical")
            else "Capacity threshold"
        )
        is_clinical = threshold_group == "Clinical action threshold"

        for strategy in ("stage2", "two_stage"):
            if strategy == "stage2":
                row: dict[str, object] = {
                    "threshold_group": threshold_group,
                    "threshold_name": threshold_name,
                    "threshold": round(threshold, 6),
                    "threshold_source": threshold_row["source"],
                    "strategy": "Stage 2 LM5_common18 only",
                    "analysis_population": (
                        "first fully evaluable operations eligible for Stage 2"
                    ),
                    "n": stage2_n,
                    "events": stage2_events,
                    "event_rate": round(stage2_events / stage2_n, 4),
                }
                valid_counts = []
                point = workload_lm5.loc[threshold_name]
                for metric, column in stage2_point_columns.items():
                    lower, upper, valid = stage2_ci(threshold, metric)
                    row[f"{metric}_95CI"] = fmt_estimate_ci(
                        point[column], lower, upper, digits=metric_digits[metric]
                    )
                    valid_counts.append(valid)
                if is_clinical:
                    nb_point = net_lm5.loc[threshold_name]
                    lower, upper, valid = stage2_ci(threshold, "net_benefit")
                    row["net_benefit_95CI"] = fmt_estimate_ci(
                        nb_point["net_benefit_model"],
                        lower,
                        upper,
                        digits=metric_digits["net_benefit"],
                    )
                    valid_counts.append(valid)
                    lower, upper, valid = stage2_ci(
                        threshold, "net_interventions_avoided_per_100_vs_all"
                    )
                    row[
                        "net_interventions_avoided_per_100_vs_all_95CI"
                    ] = fmt_estimate_ci(
                        nb_point["net_interventions_avoided_per_100_vs_all"],
                        lower,
                        upper,
                        digits=metric_digits[
                            "net_interventions_avoided_per_100_vs_all"
                        ],
                    )
                    valid_counts.append(valid)
                    row["estimability_note"] = "all prespecified metrics estimable"
                else:
                    row["net_benefit_95CI"] = "not estimable"
                    row[
                        "net_interventions_avoided_per_100_vs_all_95CI"
                    ] = "not estimable"
                    row["estimability_note"] = (
                        "net benefit was not prespecified for capacity/workload thresholds"
                    )
                row["minimum_valid_bootstrap_replicates"] = min(
                    count for count in valid_counts if count > 0
                )
                rows.append(row)
            else:
                row = {
                    "threshold_group": threshold_group,
                    "threshold_name": threshold_name,
                    "threshold": round(threshold, 6),
                    "threshold_source": threshold_row["source"],
                    "strategy": "Two-stage fixed strategy (Stage 1 rule + Stage 2 LM5)",
                    "analysis_population": "all first fully evaluable operations",
                    "n": all_n,
                    "events": all_events,
                    "event_rate": round(all_events / all_n, 4),
                }
                valid_counts = []
                for metric, two_metric in two_stage_metric_names.items():
                    if not is_clinical and metric in {
                        "net_benefit",
                        "net_interventions_avoided_per_100_vs_all",
                    }:
                        row[f"{metric}_95CI"] = "not estimable"
                        continue
                    key = (threshold_name, two_metric)
                    if key not in two_map.index:
                        row[f"{metric}_95CI"] = "not estimable"
                        continue
                    source = two_map.loc[key]
                    row[f"{metric}_95CI"] = fmt_estimate_ci(
                        source["estimate"],
                        source["ci95_lower"],
                        source["ci95_upper"],
                        digits=metric_digits[metric],
                    )
                    valid_counts.append(int(source["valid_replicates"]))
                row["estimability_note"] = (
                    "all prespecified metrics estimable"
                    if is_clinical
                    else (
                        "net benefit was not prespecified for capacity/workload thresholds"
                    )
                )
                row["minimum_valid_bootstrap_replicates"] = min(valid_counts)
                rows.append(row)

    table = pd.DataFrame(rows)
    text = "\n".join(table.astype(str).to_numpy().ravel()).lower()
    forbidden = {"inf", "+inf", "-inf", "nan", "na"}
    tokens = set(text.replace("(", " ").replace(")", " ").replace(",", " ").split())
    if tokens & forbidden:
        raise AssertionError(
            "Table 3 contains raw Inf/NaN/NA; use the explicit 'not estimable' label"
        )
    return table


def build_lineage(
    source_hashes: dict[Path, str],
    source_rows: dict[Path, int],
) -> pd.DataFrame:
    script = Path(__file__).resolve()
    mappings: dict[Path, list[tuple[Path, str, str]]] = {
        TABLE1: [
            (
                INSPIRE_MAIN,
                "INSPIRE model-development Stage 2 cohort",
                "descriptive summaries, missingness, and SMD",
            ),
            (
                MOVER_MAIN,
                "MOVER locked U0 Stage 2 cohort",
                (
                    "descriptive summaries, missingness, and SMD; MOVER emergency "
                    "technical placeholder treated as structurally unavailable"
                ),
            ),
        ],
        TABLE2: [
            (
                U0_POINT,
                "locked U0 point estimates",
                "sample counts and point estimates",
            ),
            (
                U0_BOOT_SUMMARY,
                "locked paired patient bootstrap summary",
                "95% CIs and paired LM5-minus-simple-model differences",
            ),
        ],
        TABLE3: [
            (MOVER_MAIN, "MOVER locked U0 Stage 2 cohort", "Stage 2 denominator/events"),
            (
                MOVER_FIRST,
                "MOVER first fully evaluable cohort",
                "two-stage denominator/events",
            ),
            (U0_THRESHOLDS, "locked threshold contract", "threshold names and sources"),
            (U0_WORKLOAD, "locked U0 workload estimates", "Stage 2 point estimates"),
            (
                U0_NET_BENEFIT,
                "locked U0 clinical net-benefit estimates",
                "Stage 2 clinical utility point estimates",
            ),
            (
                U0_BOOT_SUMMARY,
                "locked paired patient bootstrap summary",
                "Stage 2 direct 95% CIs",
            ),
            (
                U0_BOOT_REPLICATES,
                "locked paired patient bootstrap replicates",
                "deterministic Stage 2 specificity/workload/net-intervention CIs",
            ),
            (
                U0_TWO_STAGE_SUMMARY,
                "locked two-stage bootstrap summary",
                "two-stage point estimates and 95% CIs",
            ),
        ],
    }
    rows = []
    script_hash = sha256(script)
    for output, sources in mappings.items():
        output_hash = sha256(output)
        for source, role, transformation in sources:
            rows.append(
                {
                    "output_table": output.name,
                    "output_sha256": output_hash,
                    "source_path": str(source.relative_to(ROOT)),
                    "source_sha256": source_hashes[source],
                    "source_rows": source_rows[source],
                    "source_role": role,
                    "transformation": transformation,
                    "script_path": str(script.relative_to(ROOT)),
                    "script_sha256": script_hash,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    sources = [
        INSPIRE_MAIN,
        MOVER_MAIN,
        MOVER_FIRST,
        U0_POINT,
        U0_BOOT_SUMMARY,
        U0_BOOT_REPLICATES,
        U0_THRESHOLDS,
        U0_WORKLOAD,
        U0_NET_BENEFIT,
        U0_TWO_STAGE_SUMMARY,
    ]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required locked sources missing: {missing}")
    hashes_before = {path: sha256(path) for path in sources}

    inspire = read_csv(INSPIRE_MAIN)
    mover = read_csv(MOVER_MAIN)
    mover_first = read_csv(MOVER_FIRST)
    point = read_csv(U0_POINT)
    bootstrap = read_csv(U0_BOOT_SUMMARY)
    bootstrap_replicates = read_csv(U0_BOOT_REPLICATES)
    thresholds = read_csv(U0_THRESHOLDS)
    workload = read_csv(U0_WORKLOAD)
    net_benefit = read_csv(U0_NET_BENEFIT)
    two_stage = read_csv(U0_TWO_STAGE_SUMMARY)
    frames = {
        INSPIRE_MAIN: inspire,
        MOVER_MAIN: mover,
        MOVER_FIRST: mover_first,
        U0_POINT: point,
        U0_BOOT_SUMMARY: bootstrap,
        U0_BOOT_REPLICATES: bootstrap_replicates,
        U0_THRESHOLDS: thresholds,
        U0_WORKLOAD: workload,
        U0_NET_BENEFIT: net_benefit,
        U0_TWO_STAGE_SUMMARY: two_stage,
    }

    table1 = build_table1(inspire, mover)
    table2 = build_table2(point, bootstrap)
    table3 = build_table3(
        mover,
        mover_first,
        thresholds,
        workload,
        net_benefit,
        bootstrap,
        bootstrap_replicates,
        two_stage,
    )

    table1.to_csv(TABLE1, index=False, encoding="utf-8-sig")
    table2.to_csv(TABLE2, index=False, encoding="utf-8-sig")
    table3.to_csv(TABLE3, index=False, encoding="utf-8-sig")

    hashes_after = {path: sha256(path) for path in sources}
    if hashes_before != hashes_after:
        raise RuntimeError("a locked cohort or U0 source changed while tables were built")

    lineage = build_lineage(
        hashes_before,
        {path: int(len(frames[path])) for path in sources},
    )
    lineage.to_csv(LINEAGE, index=False, encoding="utf-8-sig")

    if len(table1) != 24 or len(table2) != 9 or len(table3) != 14:
        raise AssertionError("unexpected publication-table dimensions")
    lm5_auroc = table2.loc[table2["metric"].eq("AUROC"), "LM5_common18_estimate_95CI"].iat[0]
    lm5_auprc = table2.loc[table2["metric"].eq("AUPRC"), "LM5_common18_estimate_95CI"].iat[0]
    lm5_brier = table2.loc[table2["metric"].eq("Brier score"), "LM5_common18_estimate_95CI"].iat[0]
    print(
        "Publication tables built without model rerun:\n"
        f"  Table 1: {len(table1)} rows; INSPIRE n={len(inspire):,}, "
        f"MOVER n={len(mover):,}\n"
        f"  Table 2: AUROC {lm5_auroc}; AUPRC {lm5_auprc}; "
        f"Brier {lm5_brier}\n"
        f"  Table 3: {len(table3)} rows across seven locked thresholds and two strategies\n"
        f"  Lineage: {len(lineage)} source-to-table links"
    )


if __name__ == "__main__":
    main()

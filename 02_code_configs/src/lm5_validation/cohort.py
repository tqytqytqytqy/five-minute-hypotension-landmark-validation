"""Cohort assembly that composes source and observation contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .observation import (
    FEATURE_COLUMNS,
    build_h5,
    build_lm5_common18,
    build_observation_2x2,
    build_r1,
    classify_future_outcome,
    classify_stage1,
    find_h5_t0,
    normalize_grid_map,
)


DatasetName = Literal["INSPIRE", "MOVER"]


@dataclass
class CohortArtifacts:
    cases: pd.DataFrame
    h5: pd.DataFrame
    r1: pd.DataFrame | None
    t0: pd.DataFrame
    feature_audit: pd.DataFrame
    stage1: pd.DataFrame
    h5_outcome: pd.DataFrame
    all_case_level: pd.DataFrame
    first_eligible_case_level: pd.DataFrame
    main_stage2: pd.DataFrame
    observation_cells: pd.DataFrame | None
    observation_audit: pd.DataFrame | None
    flow: pd.DataFrame


def canonical_cases(source_cohort: pd.DataFrame) -> pd.DataFrame:
    required = {
        "dataset",
        "patient_id",
        "encounter_id",
        "age",
        "male",
        "bmi",
        "asa",
        "anesthesia_start",
        "anesthesia_duration_min",
    }
    missing = sorted(required - set(source_cohort.columns))
    if missing:
        raise ValueError(f"source cohort missing columns: {missing}")
    out = pd.DataFrame(
        {
            "dataset": source_cohort["dataset"].astype(str),
            "patient_id": source_cohort["patient_id"].astype(str),
            "case_id": source_cohort["encounter_id"].astype(str),
            "age_years": pd.to_numeric(source_cohort["age"], errors="coerce"),
            "male": pd.to_numeric(source_cohort["male"], errors="coerce"),
            "bmi": pd.to_numeric(source_cohort["bmi"], errors="coerce"),
            "asa": pd.to_numeric(source_cohort["asa"], errors="coerce"),
            "anesthesia_start": source_cohort["anesthesia_start"],
            "anesthesia_duration_min": pd.to_numeric(
                source_cohort["anesthesia_duration_min"], errors="coerce"
            ),
            "surgery_type": source_cohort.get(
                "surgery_type", pd.Series(pd.NA, index=source_cohort.index)
            ).astype("string"),
            "surgical_service": source_cohort.get(
                "surgical_service", pd.Series(pd.NA, index=source_cohort.index)
            ).astype("string"),
            "emergency": source_cohort.get(
                "emergency", pd.Series(False, index=source_cohort.index)
            ).astype("boolean"),
        }
    )
    if out["case_id"].duplicated().any():
        raise ValueError("canonical cases contain duplicate case_id values")
    return out.reset_index(drop=True)


def canonical_source_map(source_map: pd.DataFrame) -> pd.DataFrame:
    required = {
        "encounter_id",
        "minute_from_anesthesia_start",
        "map_value",
        "map_source",
    }
    missing = sorted(required - set(source_map.columns))
    if missing:
        raise ValueError(f"source MAP missing columns: {missing}")
    out = pd.DataFrame(
        {
            "case_id": source_map["encounter_id"].astype(str),
            "minute_from_anesthesia_start": pd.to_numeric(
                source_map["minute_from_anesthesia_start"], errors="coerce"
            ),
            "map": pd.to_numeric(source_map["map_value"], errors="coerce"),
            "source": source_map["map_source"].astype(str).str.upper(),
        }
    )
    return out.dropna(subset=["case_id", "minute_from_anesthesia_start", "map"])


def build_processes(
    dataset: DatasetName,
    raw_map: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
    """Return H5, optional R1, and source mapping audit."""

    canonical = canonical_source_map(raw_map)
    source_counts = (
        canonical.groupby("source", dropna=False).size().rename("n_rows").reset_index()
    )
    source_counts["dataset"] = dataset
    source_counts["main_process_included"] = source_counts["source"].isin(["ART", "NIBP"])
    formal = canonical[canonical["source"].isin(["ART", "NIBP"])].copy()
    if formal.empty:
        raise ValueError(f"{dataset} has no formally mapped ART/NIBP MAP records")
    if dataset == "INSPIRE":
        grid = formal.rename(
            columns={"minute_from_anesthesia_start": "time_min"}
        )[["case_id", "time_min", "map", "source"]]
        h5 = normalize_grid_map(
            grid,
            case_col="case_id",
            time_col="time_min",
            map_col="map",
            source_col="source",
            minimum_time_inclusive=0.0,
        )
        r1 = None
    elif dataset == "MOVER":
        h5 = build_h5(formal)
        r1 = build_r1(formal)
    else:
        raise ValueError("dataset must be INSPIRE or MOVER")
    return h5, r1, source_counts


def _sort_key(cases: pd.DataFrame, dataset: DatasetName) -> pd.Series:
    if dataset == "MOVER":
        return pd.to_datetime(cases["anesthesia_start"], errors="coerce")
    return pd.to_numeric(cases["anesthesia_start"], errors="coerce")


def assemble_cohort(
    dataset: DatasetName,
    source_cohort: pd.DataFrame,
    source_map: pd.DataFrame,
) -> tuple[CohortArtifacts, pd.DataFrame]:
    """Apply the locked t0, feature, stage-1, endpoint and first-case rules."""

    cases = canonical_cases(source_cohort)
    h5, r1, source_audit = build_processes(dataset, source_map)
    t0_all = find_h5_t0(h5)
    duration = cases[["case_id", "anesthesia_duration_min"]]
    t0 = t0_all.merge(duration, on="case_id", how="inner", validate="one_to_one")
    t0["anesthesia_end_supports_t0_plus_30"] = (
        t0["anesthesia_duration_min"] >= t0["t0_min"] + 30.0
    ).astype(int)
    t0 = t0[t0["anesthesia_end_supports_t0_plus_30"].eq(1)].copy()

    features, feature_audit = build_lm5_common18(cases, h5, t0)
    stage1 = classify_stage1(h5, t0)
    h5_outcome = classify_future_outcome(h5, t0, process="H5")

    stage_keep = stage1[
        [
            "case_id",
            "stage1_evaluable",
            "early_recurrence_after_recovery",
            "stage1_high_alert",
            "stage2_eligible",
            "stage1_exclusion_reason",
        ]
    ].copy()
    case_meta = cases.drop(columns=["age_years", "male", "bmi", "asa"])
    all_case = (
        case_meta.merge(t0[["case_id", "t0_min", "t0_source"]], on="case_id", how="inner")
        .merge(feature_audit, on=["case_id", "t0_min"], how="left")
        .merge(features, on="case_id", how="left")
        .merge(stage_keep, on="case_id", how="left")
        .merge(h5_outcome, on=["case_id", "t0_min"], how="left", suffixes=("", "_outcome"))
    )
    all_case["fully_evaluable_t0_operation"] = (
        all_case["feature_evaluable"].eq(1)
        & all_case["stage1_evaluable"].eq(1)
        & all_case["outcome_evaluable"].eq(1)
    ).astype(int)
    all_case["operation_sort_key"] = _sort_key(all_case, dataset)
    all_case["first_fully_evaluable_operation_per_patient"] = 0
    candidates = all_case[all_case["fully_evaluable_t0_operation"].eq(1)].copy()
    first_indices = (
        candidates.sort_values(
            ["patient_id", "operation_sort_key", "case_id"], kind="mergesort"
        )
        .groupby("patient_id", sort=False)
        .head(1)
        .index
    )
    all_case.loc[first_indices, "first_fully_evaluable_operation_per_patient"] = 1
    all_case["main_stage2_eligible"] = (
        all_case["first_fully_evaluable_operation_per_patient"].eq(1)
        & all_case["stage2_eligible"].eq(1)
    ).astype(int)

    first = all_case[
        all_case["first_fully_evaluable_operation_per_patient"].eq(1)
    ].copy()
    main = first[first["main_stage2_eligible"].eq(1)].copy()

    cells = None
    cell_audit = None
    if r1 is not None:
        cells, cell_audit = build_observation_2x2(cases, h5, r1, t0)
        main_ids = set(main["case_id"].astype(str))
        cells = cells[cells["case_id"].astype(str).isin(main_ids)].copy()
        cell_audit = cell_audit[
            cell_audit["case_id"].astype(str).isin(main_ids)
        ].copy()

    flow_values = [
        ("basic_common_operations", len(cases)),
        ("H5_has_t0_after_3_to_30_min", len(t0_all)),
        ("anesthesia_end_at_or_after_t0_plus_30", len(t0)),
        ("H5_feature_evaluable", int(feature_audit["feature_evaluable"].sum())),
        ("H5_outcome_evaluable", int(h5_outcome["outcome_evaluable"].sum())),
        ("fully_evaluable_t0_operations", int(all_case["fully_evaluable_t0_operation"].sum())),
        ("first_fully_evaluable_operation_per_patient", len(first)),
        ("stage1_direct_alert_first_operations", int(first["stage1_high_alert"].eq(1).sum())),
        ("main_stage2_model_cohort", len(main)),
        ("main_stage2_events", int(pd.to_numeric(main["primary_outcome"], errors="coerce").sum())),
    ]
    flow = pd.DataFrame(flow_values, columns=["step", "n"])
    flow.insert(0, "dataset", dataset)

    artifacts = CohortArtifacts(
        cases=cases,
        h5=h5,
        r1=r1,
        t0=t0,
        feature_audit=feature_audit,
        stage1=stage1,
        h5_outcome=h5_outcome,
        all_case_level=all_case,
        first_eligible_case_level=first,
        main_stage2=main,
        observation_cells=cells,
        observation_audit=cell_audit,
        flow=flow,
    )
    return artifacts, source_audit


def assert_feature_contract(frame: pd.DataFrame) -> None:
    missing = [name for name in FEATURE_COLUMNS if name not in frame.columns]
    if missing:
        raise AssertionError(f"missing frozen feature columns: {missing}")
    structurally_missing = [name for name in FEATURE_COLUMNS if frame[name].notna().sum() == 0]
    if structurally_missing:
        raise AssertionError(
            f"structurally missing features violate the No-Go gate: {structurally_missing}"
        )


__all__ = [
    "CohortArtifacts",
    "assemble_cohort",
    "assert_feature_contract",
    "build_processes",
    "canonical_cases",
    "canonical_source_map",
]

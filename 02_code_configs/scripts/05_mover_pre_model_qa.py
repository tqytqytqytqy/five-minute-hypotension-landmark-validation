#!/usr/bin/env python3
"""Performance-blind MOVER variable/coverage gate; never loads the model."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_code_configs/src"))

from lm5_validation.observation import FEATURE_COLUMNS  # noqa: E402
from lm5_validation.frozen import sha256_file  # noqa: E402


MOVER = ROOT / "03_derived_cohorts/MOVER"
OUT = ROOT / "09_QA_reproducibility/reports"


def check(name: str, passed: bool, detail: object) -> dict:
    return {"check": name, "passed": bool(passed), "detail": detail}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    analysis_path = ROOT / "02_code_configs/configs/analysis.json"
    source_manifest_path = ROOT / "01_source_audit_lineage/source_manifest.json"
    input_paths = {
        "analysis_config": analysis_path,
        "source_manifest": source_manifest_path,
        "observed_map": MOVER / "mover_observed_map_20_200.csv.gz",
        "H5_grid": MOVER / "mover_H5_grid.csv.gz",
        "R1_grid": MOVER / "mover_R1_grid.csv.gz",
        "H5_t0": MOVER / "mover_H5_t0.csv.gz",
        "main_stage2": MOVER / "mover_main_stage2.csv.gz",
        "first_fully_evaluable": MOVER / "mover_first_fully_evaluable_operation.csv.gz",
        "all_t0_operations": MOVER / "mover_all_t0_operations.csv.gz",
        "cohort_flow": MOVER / "mover_cohort_flow.csv",
        "observation_2x2_cells": MOVER / "mover_observation_2x2_cells.csv.gz",
    }
    missing_inputs = [name for name, path in input_paths.items() if not path.exists()]
    if missing_inputs:
        raise FileNotFoundError(f"MOVER QA inputs missing: {missing_inputs}")
    hashes_before = {name: sha256_file(path) for name, path in input_paths.items()}
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_gate = source_manifest["gate"]
    source_map = pd.read_csv(
        input_paths["observed_map"],
        usecols=["map_value", "map_source", "minute_from_anesthesia_start"],
        low_memory=False,
    )
    h5 = pd.read_csv(input_paths["H5_grid"], low_memory=False)
    r1 = pd.read_csv(input_paths["R1_grid"], low_memory=False)
    t0 = pd.read_csv(input_paths["H5_t0"], low_memory=False)
    main = pd.read_csv(
        input_paths["main_stage2"],
        dtype={"patient_id": "string", "case_id": "string"},
        low_memory=False,
    )
    first = pd.read_csv(
        input_paths["first_fully_evaluable"],
        dtype={"patient_id": "string", "case_id": "string"},
        low_memory=False,
    )
    flow = pd.read_csv(input_paths["cohort_flow"])
    outcome_numeric = pd.to_numeric(main["primary_outcome"], errors="coerce")
    binary_variables = ["male", "t0_arterial_source", "recovered_by_5min"]
    feature_nonmissing = {
        feature: int(pd.to_numeric(main[feature], errors="coerce").notna().sum())
        for feature in FEATURE_COLUMNS
        if feature in main
    }
    safe_flow_steps = {
        "basic_common_operations",
        "H5_has_t0_after_3_to_30_min",
        "anesthesia_end_at_or_after_t0_plus_30",
        "H5_feature_evaluable",
        "H5_outcome_evaluable",
        "fully_evaluable_t0_operations",
        "first_fully_evaluable_operation_per_patient",
        "stage1_direct_alert_first_operations",
        "main_stage2_model_cohort",
    }

    checks = [
        check("study_id_matches_locked_analysis", analysis.get("study_id") == "LM5_COMMON18_INSPIRE_MOVER_20260712", analysis.get("study_id")),
        check("raw_source_gate_green", source_gate["primary_source_gate"] == "GREEN", source_gate["primary_source_gate"]),
        check(
            "formal_MAP_range_20_200_inclusive",
            source_map["map_value"].between(20, 200, inclusive="both").all(),
            {"minimum": float(source_map["map_value"].min()), "maximum": float(source_map["map_value"].max())},
        ),
        check(
            "main_H5_R1_sources_ART_or_NIBP_only",
            set(h5["source"]).issubset({"ART", "NIBP"})
            and set(r1["source"]).issubset({"ART", "NIBP"}),
            {"H5": sorted(set(h5["source"])), "R1": sorted(set(r1["source"]))},
        ),
        check(
            "t0_search_window_obeyed",
            t0["t0_min"].gt(3).all() and t0["t0_min"].le(30).all(),
            {"minimum": float(t0["t0_min"].min()), "maximum": float(t0["t0_min"].max())},
        ),
        check(
            "primary_patient_and_case_ids_complete_unique",
            main[["patient_id", "case_id"]].notna().all().all()
            and not main["patient_id"].duplicated().any()
            and not main["case_id"].duplicated().any(),
            {
                "rows": len(main),
                "patients": int(main["patient_id"].nunique()),
                "cases": int(main["case_id"].nunique()),
            },
        ),
        check(
            "all_18_columns_present",
            all(feature in main for feature in FEATURE_COLUMNS),
            {"expected": list(FEATURE_COLUMNS)},
        ),
        check(
            "no_structurally_missing_feature",
            all(feature_nonmissing.get(feature, 0) > 0 for feature in FEATURE_COLUMNS),
            feature_nonmissing,
        ),
        check(
            "binary_feature_encodings_are_0_or_1",
            all(
                set(pd.to_numeric(main[feature], errors="coerce").dropna().unique()).issubset({0.0, 1.0})
                for feature in binary_variables
            ),
            {feature: sorted(pd.to_numeric(main[feature], errors="coerce").dropna().unique().tolist()) for feature in binary_variables},
        ),
        check(
            "clinical_feature_ranges_plausible",
            main["age_years"].between(18, 100, inclusive="both").all()
            and main["bmi"].dropna().between(10, 80, inclusive="both").all()
            and main["asa"].dropna().between(1, 6, inclusive="both").all()
            and main["t0_map"].between(20, 65, inclusive="left").all()
            and main["anesthesia_start_to_t0_min"].gt(3).all()
            and main["anesthesia_start_to_t0_min"].le(30).all()
            and main["early_min_map_0_5"].between(20, 200, inclusive="both").all()
            and main["early_mean_map_0_5"].between(20, 200, inclusive="both").all(),
            "age 18-100; BMI 10-80; ASA 1-6; t0 MAP [20,65); t0 (3,30]; early MAP 20-200",
        ),
        check(
            "H5_grid_phase_zero",
            np.allclose(pd.to_numeric(h5["time_min"]) % 5, 0, atol=1e-9),
            "all H5 labels are multiples of 5 minutes from anaesthesia start",
        ),
        check(
            "R1_grid_phase_zero",
            np.allclose(pd.to_numeric(r1["time_min"]) % 1, 0, atol=1e-9),
            "all R1 labels are whole minutes from anaesthesia start",
        ),
        check(
            "stage2_outcome_contract_binary_without_aggregation",
            outcome_numeric.notna().all()
            and set(outcome_numeric.unique()).issubset({0.0, 1.0}),
            {"nonmissing": int(outcome_numeric.notna().sum()), "n": len(main)},
        ),
        check(
            "main_case_referential_integrity",
            set(main["case_id"].astype(str)).issubset(set(t0["case_id"].astype(str)))
            and set(main["case_id"].astype(str)).issubset(set(h5["case_id"].astype(str)))
            and set(main["case_id"].astype(str)).issubset(set(r1["case_id"].astype(str)))
            and set(main["case_id"].astype(str)).issubset(set(first["case_id"].astype(str))),
            "all main cases occur in H5 t0, H5, R1 and first-operation files",
        ),
        check(
            "flow_uses_only_safe_non_event_steps_for_QA_report",
            set(flow.loc[flow["step"].isin(safe_flow_steps), "step"]) == safe_flow_steps,
            sorted(safe_flow_steps),
        ),
    ]
    hashes_after = {name: sha256_file(path) for name, path in input_paths.items()}
    checks.append(
        check("QA_inputs_unchanged_during_read", hashes_before == hashes_after, "all input SHA-256 values stable")
    )
    # The QA intentionally validates but never aggregates outcome=1 and never opens model.json.
    report = {
        "study_id": analysis["study_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gate": "GREEN" if all(item["passed"] for item in checks) else "RED",
        "performance_blind": True,
        "model_bundle_loaded": False,
        "predictions_generated": False,
        "outcome_event_rate_calculated": False,
        "qa_code_sha256": sha256_file(Path(__file__)),
        "input_sha256": hashes_after,
        "checks": checks,
        "safe_flow_counts": flow[flow["step"].isin(safe_flow_steps)].to_dict("records"),
    }
    (OUT / "MOVER_pre_model_QA.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(checks).assign(detail=lambda x: x["detail"].astype(str)).to_csv(
        OUT / "MOVER_pre_model_QA.csv", index=False, encoding="utf-8-sig"
    )
    print(json.dumps({"gate": report["gate"], "checks": len(checks)}, ensure_ascii=False))
    if report["gate"] != "GREEN":
        raise SystemExit("MOVER pre-model QA gate RED")


if __name__ == "__main__":
    main()

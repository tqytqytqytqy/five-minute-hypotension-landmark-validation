#!/usr/bin/env python3
"""Build INSPIRE or MOVER H5/R1 cohorts under the frozen observation rules."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_code_configs/src"))

from lm5_validation.cohort import assemble_cohort, assert_feature_contract  # noqa: E402
from lm5_validation.observation import FEATURE_COLUMNS  # noqa: E402


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def write_csv(frame: pd.DataFrame, path: Path, *, compress: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compress:
        frame.to_csv(path, index=False, compression={"method": "gzip", "compresslevel": 5})
    else:
        frame.to_csv(path, index=False, encoding="utf-8-sig")


def load(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    if dataset == "INSPIRE":
        out = ROOT / "03_derived_cohorts/INSPIRE"
        cohort_path = out / "inspire_basic_common_cohort_all_operations.csv.gz"
        map_path = out / "inspire_observed_map_20_200.csv.gz"
    else:
        out = ROOT / "03_derived_cohorts/MOVER"
        cohort_path = out / "mover_basic_common_cohort_all_operations.csv.gz"
        map_path = out / "mover_observed_map_20_200.csv.gz"
    if not cohort_path.exists() or not map_path.exists():
        raise FileNotFoundError(f"source-domain files missing for {dataset}")
    log(f"{dataset}: loading source cohort")
    cohort = pd.read_csv(cohort_path, low_memory=False)
    log(f"{dataset}: loading observed MAP rows")
    source_map = pd.read_csv(
        map_path,
        usecols=[
            "encounter_id",
            "minute_from_anesthesia_start",
            "map_value",
            "map_source",
        ],
        dtype={"encounter_id": "string", "map_source": "string"},
        low_memory=False,
    )
    return cohort, source_map, out


def run(dataset: str) -> None:
    cohort, source_map, out = load(dataset)
    log(f"{dataset}: building H5{' and R1' if dataset == 'MOVER' else ''}")
    artifacts, source_audit = assemble_cohort(dataset, cohort, source_map)
    del source_map
    gc.collect()
    assert_feature_contract(artifacts.main_stage2)

    write_csv(artifacts.h5, out / f"{dataset.lower()}_H5_grid.csv.gz")
    if artifacts.r1 is not None:
        write_csv(artifacts.r1, out / "mover_R1_grid.csv.gz")
    write_csv(artifacts.t0, out / f"{dataset.lower()}_H5_t0.csv.gz")
    write_csv(artifacts.feature_audit, out / f"{dataset.lower()}_feature_audit.csv.gz")
    write_csv(artifacts.stage1, out / f"{dataset.lower()}_stage1.csv.gz")
    write_csv(artifacts.h5_outcome, out / f"{dataset.lower()}_H5_outcomes.csv.gz")
    write_csv(artifacts.all_case_level, out / f"{dataset.lower()}_all_t0_operations.csv.gz")
    write_csv(
        artifacts.first_eligible_case_level,
        out / f"{dataset.lower()}_first_fully_evaluable_operation.csv.gz",
    )
    write_csv(artifacts.main_stage2, out / f"{dataset.lower()}_main_stage2.csv.gz")
    write_csv(artifacts.flow, out / f"{dataset.lower()}_cohort_flow.csv", compress=False)
    write_csv(source_audit, out / f"{dataset.lower()}_map_source_audit.csv", compress=False)
    if artifacts.observation_cells is not None:
        write_csv(artifacts.observation_cells, out / "mover_observation_2x2_cells.csv.gz")
        write_csv(artifacts.observation_audit, out / "mover_observation_2x2_audit.csv.gz")

    missingness = []
    for variable in FEATURE_COLUMNS:
        for scope_name, frame in [
            ("first_fully_evaluable", artifacts.first_eligible_case_level),
            ("main_stage2", artifacts.main_stage2),
        ]:
            missingness.append(
                {
                    "dataset": dataset,
                    "scope": scope_name,
                    "variable": variable,
                    "n": len(frame),
                    "missing_n": int(frame[variable].isna().sum()),
                    "missing_percent": float(100 * frame[variable].isna().mean()) if len(frame) else 0.0,
                    "structurally_missing": int(frame[variable].notna().sum() == 0),
                }
            )
    pd.DataFrame(missingness).to_csv(
        out / f"{dataset.lower()}_feature_missingness.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "dataset": dataset,
        "basic_operations": len(artifacts.cases),
        "h5_grid_rows": len(artifacts.h5),
        "r1_grid_rows": len(artifacts.r1) if artifacts.r1 is not None else None,
        "t0_operations": len(artifacts.t0),
        "first_fully_evaluable_operations": len(artifacts.first_eligible_case_level),
        "main_stage2_n": len(artifacts.main_stage2),
        "main_stage2_events": int(pd.to_numeric(artifacts.main_stage2["primary_outcome"]).sum()),
        "feature_contract": list(FEATURE_COLUMNS),
        "structural_missingness_gate": "GREEN",
    }
    (out / f"{dataset.lower()}_cohort_build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(
        f"{dataset}: complete; first evaluable={summary['first_fully_evaluable_operations']:,}, "
        f"stage2={summary['main_stage2_n']:,}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["INSPIRE", "MOVER"], required=True)
    args = parser.parse_args()
    run(args.dataset)


if __name__ == "__main__":
    main()


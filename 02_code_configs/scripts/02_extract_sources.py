#!/usr/bin/env python3
"""Build source-domain cohorts and observed MAP files without outcomes/models."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_code_configs/src"))

from lm5_validation.sources import (  # noqa: E402
    extract_inspire_common_cohort,
    extract_mover_common_cohort,
    iter_inspire_map_observations,
    iter_mover_map_observations,
)


PATHS = json.loads((ROOT / "02_code_configs/configs/paths.json").read_text(encoding="utf-8"))


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def atomic_stream_csv(chunks, target: Path, audit_target: Path, label: str) -> None:
    """Write the canonical subset incrementally; cached shards make reruns safe."""

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    header = True
    total_raw = 0
    total_retained = 0
    chunk_count = 0
    audit_rows: list[dict] = []
    keep_columns = [
        "dataset",
        "patient_id",
        "encounter_id",
        "observed_time",
        "minute_from_anesthesia_start",
        "map_value",
        "map_source",
        "raw_name",
        "raw_display_name",
        "raw_unit",
    ]
    for frame in chunks:
        chunk_count += 1
        raw_rows = int(frame.attrs.get("raw_rows", 0))
        retained = int(len(frame))
        total_raw += raw_rows
        total_retained += retained
        audit_rows.append(
            {
                "chunk": chunk_count,
                "unit": frame.attrs.get("unit"),
                "member_key": frame.attrs.get("member_key"),
                "raw_rows": raw_rows,
                "retained_rows": retained,
                "cache_hit": bool(frame.attrs.get("cache_hit", False)),
                "map_range": "20-200 inclusive",
                "locf": False,
                "interpolation": False,
            }
        )
        if retained:
            frame[keep_columns].to_csv(
                tmp,
                mode="a",
                header=header,
                index=False,
                compression={"method": "gzip", "compresslevel": 5},
            )
            header = False
        if chunk_count % 100 == 0:
            log(
                f"{label}: chunks={chunk_count:,}, raw={total_raw:,}, "
                f"MAP retained={total_retained:,}"
            )
    if header:
        pd.DataFrame(columns=keep_columns).to_csv(tmp, index=False, compression="gzip")
    os.replace(tmp, target)
    pd.DataFrame(audit_rows).to_csv(audit_target, index=False, encoding="utf-8-sig")
    summary = {
        "dataset": label,
        "chunks": chunk_count,
        "raw_rows_scanned": total_raw,
        "observed_map_rows_retained": total_retained,
        "formal_map_range": "20-200 inclusive",
        "locf": False,
        "interpolation": False,
        "output": str(target),
    }
    audit_target.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(f"{label}: complete; retained {total_retained:,} observed MAP rows")


def coverage(cohort: pd.DataFrame, dataset: str) -> pd.DataFrame:
    rows = []
    for column in cohort.columns:
        rows.append(
            {
                "dataset": dataset,
                "variable": column,
                "n": len(cohort),
                "nonmissing_n": int(cohort[column].notna().sum()),
                "nonmissing_percent": float(100 * cohort[column].notna().mean()) if len(cohort) else 0.0,
                "unique_nonmissing": int(cohort[column].nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def extract_inspire() -> None:
    out = ROOT / "03_derived_cohorts/INSPIRE"
    cache = out / "cache"
    log("INSPIRE: extracting all basic common-cohort operations (first operation deferred)")
    cohort = extract_inspire_common_cohort(
        PATHS["inspire_zip"], first_case_per_patient=False
    )
    cohort.to_csv(out / "inspire_basic_common_cohort_all_operations.csv.gz", index=False, compression="gzip")
    coverage(cohort, "INSPIRE_1.4.2").to_csv(
        out / "inspire_basic_field_coverage.csv", index=False, encoding="utf-8-sig"
    )
    log(f"INSPIRE: {len(cohort):,} basic eligible operations, {cohort.patient_id.nunique():,} patients")
    chunks = iter_inspire_map_observations(
        PATHS["inspire_zip"],
        cohort,
        chunksize=500_000,
        cache_dir=cache,
        resume=True,
        hash_source=False,
    )
    atomic_stream_csv(
        chunks,
        out / "inspire_observed_map_20_200.csv.gz",
        out / "inspire_extraction_chunk_audit.csv",
        "INSPIRE_1.4.2",
    )


def extract_mover() -> None:
    out = ROOT / "03_derived_cohorts/MOVER"
    cache = out / "cache"
    log("MOVER: extracting all basic common-cohort operations with WEIGHT locked as ounces")
    cohort = extract_mover_common_cohort(
        PATHS["mover_epic_emr"],
        first_case_per_patient=False,
        unlabelled_weight_unit="oz",
    )
    cohort.to_csv(out / "mover_basic_common_cohort_all_operations.csv.gz", index=False, compression="gzip")
    coverage(cohort, "MOVER_EPIC").to_csv(
        out / "mover_basic_field_coverage.csv", index=False, encoding="utf-8-sig"
    )
    log(f"MOVER: {len(cohort):,} basic eligible operations, {cohort.patient_id.nunique():,} patients")
    chunks = iter_mover_map_observations(
        PATHS["mover_epic_flowsheets_cleaned"],
        cohort,
        chunksize=400_000,
        expected_parts=19,
        cache_dir=cache,
        resume=True,
        hash_sources=False,
    )
    atomic_stream_csv(
        chunks,
        out / "mover_observed_map_20_200.csv.gz",
        out / "mover_extraction_chunk_audit.csv",
        "MOVER_EPIC",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["inspire", "mover", "both"], default="both")
    args = parser.parse_args()
    if args.dataset in {"inspire", "both"}:
        extract_inspire()
    if args.dataset in {"mover", "both"}:
        extract_mover()


if __name__ == "__main__":
    main()


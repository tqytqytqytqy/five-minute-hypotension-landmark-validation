#!/usr/bin/env python3
"""Run the single prespecified no-update MOVER-H5 external validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_code_configs/src"))

from lm5_validation.frozen import (  # noqa: E402
    apply_imputation,
    load_bundle,
    predict_model_dict,
    sha256_file,
    validate_bundle_contract,
    verify_test_vectors,
)
from lm5_validation.validation import (  # noqa: E402
    paired_bootstrap_validation,
    point_performance,
    threshold_point_tables,
    two_stage_strategy_bootstrap,
)
from lm5_validation.statistics import decision_curve  # noqa: E402


CONFIG = json.loads((ROOT / "02_code_configs/configs/analysis.json").read_text(encoding="utf-8"))
MODEL_DIR = ROOT / "04_frozen_INSPIRE_LM5_model"
MOVER_INPUT = ROOT / "03_derived_cohorts/MOVER/mover_main_stage2.csv.gz"
FIRST_INPUT = ROOT / "03_derived_cohorts/MOVER/mover_first_fully_evaluable_operation.csv.gz"
QA_INPUT = ROOT / "09_QA_reproducibility/reports/MOVER_pre_model_QA.json"
QA_SCRIPT = ROOT / "02_code_configs/scripts/05_mover_pre_model_qa.py"
FREEZE_RECEIPT = ROOT / "10_run_logs_manifest/pre_U0_freeze_receipt.json"
OUTPUT = ROOT / "05_MOVER_validation/H5_primary"
LOCK = ROOT / "10_run_logs_manifest/U0_execution_lock.json"

QA_INPUT_PATHS = {
    "analysis_config": ROOT / "02_code_configs/configs/analysis.json",
    "source_manifest": ROOT / "01_source_audit_lineage/source_manifest.json",
    "observed_map": ROOT / "03_derived_cohorts/MOVER/mover_observed_map_20_200.csv.gz",
    "H5_grid": ROOT / "03_derived_cohorts/MOVER/mover_H5_grid.csv.gz",
    "R1_grid": ROOT / "03_derived_cohorts/MOVER/mover_R1_grid.csv.gz",
    "H5_t0": ROOT / "03_derived_cohorts/MOVER/mover_H5_t0.csv.gz",
    "main_stage2": MOVER_INPUT,
    "first_fully_evaluable": FIRST_INPUT,
    "all_t0_operations": ROOT / "03_derived_cohorts/MOVER/mover_all_t0_operations.csv.gz",
    "cohort_flow": ROOT / "03_derived_cohorts/MOVER/mover_cohort_flow.csv",
    "observation_2x2_cells": ROOT
    / "03_derived_cohorts/MOVER/mover_observation_2x2_cells.csv.gz",
}
RECEIPT_CORE_FILES = (
    "model.json",
    "preprocess.json",
    "thresholds.json",
    "feature_contract.json",
    "cohort_endpoint.json",
    "test_vectors.csv",
    "freeze_provenance.json",
    "SHA256SUMS.csv",
    "code_SHA256SUMS.csv",
)
SECONDARY_PRESPEC_FILES = {
    "02_code_configs/scripts/07_secondary_analyses.py": ROOT
    / "02_code_configs/scripts/07_secondary_analyses.py",
    "00_protocol_SAP/secondary_analysis_prespec.json": ROOT
    / "00_protocol_SAP/secondary_analysis_prespec.json",
}
RESULT_MANIFEST_NAME = "U0_result_SHA256SUMS.csv"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def _encoded_json(payload: dict) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_all(fd: int, content: bytes) -> None:
    view = memoryview(content)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("atomic JSON write made no forward progress")
        view = view[written:]


def _fsync_directory(path: Path) -> None:
    """Best-effort durability barrier for a completed rename/create."""

    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_replace_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.{time.time_ns()}")
    encoded = _encoded_json(payload)
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        _write_all(fd, encoded)
        os.fsync(fd)
    except BaseException:
        os.close(fd)
        temporary.unlink(missing_ok=True)
        raise
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
    try:
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_create_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _encoded_json(payload)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        _write_all(fd, encoded)
        os.fsync(fd)
    except BaseException:
        os.close(fd)
        path.unlink(missing_ok=True)
        raise
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
    _fsync_directory(path.parent)


def process_is_running(pid: object) -> bool:
    try:
        numeric = int(pid)
        if numeric <= 0:
            return False
        os.kill(numeric, 0)
        return True
    except (TypeError, ValueError, ProcessLookupError):
        return False
    except PermissionError:
        return True


def _safe_manifest_path(base: Path, relative: object) -> Path:
    value = str(relative)
    candidate = Path(value)
    if not value or candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"unsafe manifest path: {value!r}")
    resolved_base = base.resolve()
    resolved = (base / candidate).resolve()
    if resolved != resolved_base and resolved_base not in resolved.parents:
        raise ValueError(f"manifest path escapes its root: {value!r}")
    return resolved


def _read_hash_manifest(path: Path, path_column: str) -> pd.DataFrame:
    table = pd.read_csv(path, dtype={path_column: "string", "sha256": "string"})
    required = {path_column, "size_bytes", "sha256"}
    if not required.issubset(table.columns) or table.empty:
        raise ValueError(f"invalid or empty hash manifest: {path}")
    if table[path_column].isna().any() or table[path_column].duplicated().any():
        raise ValueError(f"hash manifest has missing or duplicate paths: {path}")
    if not table["sha256"].str.fullmatch(r"[0-9a-f]{64}").all():
        raise ValueError(f"hash manifest has invalid SHA-256 values: {path}")
    return table


def verify_freeze_hashes() -> None:
    manifest = _read_hash_manifest(MODEL_DIR / "SHA256SUMS.csv", "relative_path")
    failures = []
    for row in manifest.itertuples(index=False):
        path = _safe_manifest_path(MODEL_DIR, row.relative_path)
        observed = sha256_file(path) if path.is_file() else None
        size = path.stat().st_size if path.is_file() else None
        if observed != row.sha256 or size != int(row.size_bytes):
            failures.append(
                {
                    "file": row.relative_path,
                    "expected": row.sha256,
                    "observed": observed,
                    "expected_size": int(row.size_bytes),
                    "observed_size": size,
                }
            )
    if failures:
        raise RuntimeError(f"freeze hash verification failed: {failures[:3]}")
    code_manifest = _read_hash_manifest(
        MODEL_DIR / "code_SHA256SUMS.csv", "relative_path_from_project_root"
    )
    code_failures = []
    for row in code_manifest.itertuples(index=False):
        path = _safe_manifest_path(ROOT, row.relative_path_from_project_root)
        observed = sha256_file(path) if path.is_file() else None
        size = path.stat().st_size if path.is_file() else None
        if observed != row.sha256 or size != int(row.size_bytes):
            code_failures.append(
                {
                    "file": row.relative_path_from_project_root,
                    "expected": row.sha256,
                    "observed": observed,
                    "expected_size": int(row.size_bytes),
                    "observed_size": size,
                }
            )
    if code_failures:
        raise RuntimeError(f"primary code hash verification failed: {code_failures[:3]}")


def locked_thresholds(
    threshold_payload: dict,
) -> tuple[list[float], list[float], pd.DataFrame]:
    if threshold_payload.get("study_id") != CONFIG["study_id"]:
        raise ValueError("threshold bundle study_id differs from the locked analysis")
    required = {
        "capacity_thresholds",
        "clinical_action_primary",
        "clinical_action_sensitivity",
        "legacy_30_feature_thresholds_forbidden",
    }
    if not required.issubset(threshold_payload):
        raise ValueError(f"threshold bundle is missing keys: {sorted(required - set(threshold_payload))}")
    if not isinstance(threshold_payload["capacity_thresholds"], dict):
        raise TypeError("capacity_thresholds must be a mapping")
    rows = []
    for name, detail in threshold_payload["capacity_thresholds"].items():
        rows.append(
            {
                "threshold_name": name,
                "threshold": float(detail["risk_threshold"]),
                "threshold_type": "INSPIRE_capacity",
                "source": detail["source"],
            }
        )
    rows.append(
        {
            "threshold_name": "clinical_action_primary",
            "threshold": float(threshold_payload["clinical_action_primary"]),
            "threshold_type": "prespecified_clinical_action",
            "source": "SAP_before_MOVER_performance",
        }
    )
    for value in threshold_payload["clinical_action_sensitivity"]:
        rows.append(
            {
                "threshold_name": f"clinical_action_sensitivity_{float(value):.2f}",
                "threshold": float(value),
                "threshold_type": "prespecified_clinical_action_sensitivity",
                "source": "SAP_before_MOVER_performance",
            }
        )
    table = pd.DataFrame(rows).drop_duplicates(["threshold_name", "threshold"])
    if not table["threshold"].between(0, 1, inclusive="neither").all():
        raise ValueError("every frozen threshold must be strictly between 0 and 1")
    capacity = table[table["threshold_type"].eq("INSPIRE_capacity")]
    if capacity.empty or not capacity["source"].str.contains("INSPIRE_patient_level_holdout").all():
        raise ValueError("capacity thresholds are not demonstrably sourced from the INSPIRE holdout")
    forbidden = set(float(value) for value in threshold_payload["legacy_30_feature_thresholds_forbidden"])
    if any(
        any(np.isclose(row.threshold, value, rtol=0, atol=1e-12) for value in forbidden)
        and row.threshold_type == "INSPIRE_capacity"
        for row in table.itertuples(index=False)
    ):
        raise ValueError("a forbidden legacy 30-feature threshold entered the capacity set")
    workload = sorted(set(table["threshold"].astype(float)))
    clinical = sorted(
        set(
            table.loc[
                table["threshold_type"].str.contains("clinical_action"), "threshold"
            ].astype(float)
        )
    )
    if not workload or not clinical or not set(clinical).issubset(set(workload)):
        raise ValueError("workload/clinical threshold sets are empty or inconsistent")
    return workload, clinical, table


def _load_json(path: Path, label: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must be a JSON object")
    return payload


def _hash_files(paths: dict[str, Path]) -> dict[str, str]:
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"locked U0 inputs are missing: {missing}")
    return {name: sha256_file(path) for name, path in paths.items()}


def verify_qa_contract(qa: dict) -> dict[str, str]:
    if qa.get("study_id") != CONFIG["study_id"]:
        raise RuntimeError("MOVER QA study_id differs from the locked analysis")
    if qa.get("gate") != "GREEN" or qa.get("performance_blind") is not True:
        raise RuntimeError("MOVER performance-blind QA gate is not GREEN")
    forbidden_true = {
        "model_bundle_loaded": qa.get("model_bundle_loaded"),
        "predictions_generated": qa.get("predictions_generated"),
        "outcome_event_rate_calculated": qa.get("outcome_event_rate_calculated"),
    }
    if any(value is not False for value in forbidden_true.values()):
        raise RuntimeError(f"MOVER QA is not demonstrably performance blind: {forbidden_true}")
    recorded = qa.get("input_sha256")
    if not isinstance(recorded, dict) or set(recorded) != set(QA_INPUT_PATHS):
        raise RuntimeError("MOVER QA input hash inventory differs from the locked inventory")
    observed = _hash_files(QA_INPUT_PATHS)
    if observed != recorded:
        failures = {
            key: {"expected": recorded.get(key), "observed": observed.get(key)}
            for key in sorted(set(recorded) | set(observed))
            if recorded.get(key) != observed.get(key)
        }
        raise RuntimeError(f"MOVER QA input hashes no longer match: {failures}")
    qa_code_sha = sha256_file(QA_SCRIPT)
    if qa.get("qa_code_sha256") != qa_code_sha:
        raise RuntimeError("MOVER QA script hash differs from the script that issued the gate")
    checks = qa.get("checks")
    if not isinstance(checks, list) or not checks or any(item.get("passed") is not True for item in checks):
        raise RuntimeError("MOVER QA contains absent or failed component checks")
    return observed


def verify_receipt_contract(receipt: dict, vector_verification: dict) -> None:
    if receipt.get("study_id") != CONFIG["study_id"]:
        raise RuntimeError("pre-U0 receipt study_id differs from the locked analysis")
    if receipt.get("freeze_version") != "1.1.0":
        raise RuntimeError("pre-U0 receipt does not certify freeze version 1.1.0")
    if receipt.get("MOVER_patient_level_predictions_seen") is not False:
        raise RuntimeError("receipt does not certify prediction-blind freezing")
    if receipt.get("MOVER_performance_seen") is not False:
        raise RuntimeError("receipt does not certify performance-blind freezing")
    if FREEZE_RECEIPT.stat().st_mode & 0o222:
        raise RuntimeError("pre-U0 receipt has writable permission bits")

    core = receipt.get("core_file_sha256")
    if not isinstance(core, dict) or set(core) != set(RECEIPT_CORE_FILES):
        raise RuntimeError("pre-U0 receipt core-file inventory is incomplete or unexpected")
    observed_core = _hash_files(
        {name: MODEL_DIR / name for name in RECEIPT_CORE_FILES}
    )
    if core != observed_core:
        raise RuntimeError("frozen model core hashes differ from the pre-U0 receipt")

    secondary = receipt.get("secondary_analysis_file_sha256")
    if not isinstance(secondary, dict) or set(secondary) != set(SECONDARY_PRESPEC_FILES):
        raise RuntimeError("receipt lacks the complete prespecified secondary-analysis hash set")
    observed_secondary = _hash_files(SECONDARY_PRESPEC_FILES)
    if secondary != observed_secondary:
        raise RuntimeError("secondary-analysis files differ from their pre-U0 receipt hashes")
    if receipt.get("test_vector_verification") != vector_verification:
        raise RuntimeError("test-vector verification differs from the pre-U0 receipt")


def verify_feature_order_contract(models: dict, preprocess: dict) -> None:
    feature_contract = _load_json(MODEL_DIR / "feature_contract.json", "feature contract")
    expected = list(CONFIG.get("features_in_order", []))
    primary = models.get("primary_model")
    if not expected or primary != "LM5_common18":
        raise RuntimeError("locked common-18 feature list or primary-model identity is invalid")
    candidates = {
        "analysis_config": expected,
        "feature_contract": list(feature_contract.get("feature_order", [])),
        "model_json": list(models["models"][primary].get("feature_order", [])),
        "preprocess_json": list(preprocess["models"][primary].get("feature_order", [])),
    }
    if feature_contract.get("study_id") != CONFIG["study_id"]:
        raise RuntimeError("feature contract study_id differs from the locked analysis")
    if any(order != expected for order in candidates.values()):
        raise RuntimeError(f"common-18 feature order differs across frozen contracts: {candidates}")


def collect_input_hashes(qa: dict, receipt: dict) -> dict[str, str]:
    """Snapshot every primary input, contract, bundle and prespecified code file."""

    paths: dict[str, Path] = {
        "qa_report": QA_INPUT,
        "qa_script": QA_SCRIPT,
        "freeze_receipt": FREEZE_RECEIPT,
    }
    paths.update({f"qa_input::{name}": path for name, path in QA_INPUT_PATHS.items()})
    paths.update(
        {f"freeze_core::{name}": MODEL_DIR / name for name in RECEIPT_CORE_FILES}
    )
    paths.update(
        {f"secondary_prespec::{name}": path for name, path in SECONDARY_PRESPEC_FILES.items()}
    )

    bundle_manifest = _read_hash_manifest(
        MODEL_DIR / "SHA256SUMS.csv", "relative_path"
    )
    for row in bundle_manifest.itertuples(index=False):
        paths[f"bundle_manifest_entry::{row.relative_path}"] = _safe_manifest_path(
            MODEL_DIR, row.relative_path
        )
    code_manifest = _read_hash_manifest(
        MODEL_DIR / "code_SHA256SUMS.csv", "relative_path_from_project_root"
    )
    for row in code_manifest.itertuples(index=False):
        paths[
            f"primary_code::{row.relative_path_from_project_root}"
        ] = _safe_manifest_path(ROOT, row.relative_path_from_project_root)

    snapshot = _hash_files(paths)
    if {name: snapshot[f"qa_input::{name}"] for name in QA_INPUT_PATHS} != qa["input_sha256"]:
        raise RuntimeError("QA input hashes changed while assembling the U0 lock snapshot")
    if {
        name: snapshot[f"freeze_core::{name}"] for name in RECEIPT_CORE_FILES
    } != receipt["core_file_sha256"]:
        raise RuntimeError("freeze core hashes changed while assembling the U0 lock snapshot")
    if {
        name: snapshot[f"secondary_prespec::{name}"] for name in SECONDARY_PRESPEC_FILES
    } != receipt["secondary_analysis_file_sha256"]:
        raise RuntimeError("secondary prespec hashes changed while assembling the U0 lock snapshot")
    return dict(sorted(snapshot.items()))


def load_pre_u0_contract() -> tuple[dict, dict, dict, dict, dict, dict[str, str]]:
    qa = _load_json(QA_INPUT, "MOVER pre-model QA report")
    receipt = _load_json(FREEZE_RECEIPT, "pre-U0 freeze receipt")
    verify_qa_contract(qa)
    verify_freeze_hashes()
    models, preprocess, thresholds = load_bundle(MODEL_DIR)
    validate_bundle_contract(
        models, preprocess, thresholds, expected_study_id=CONFIG["study_id"]
    )
    verify_feature_order_contract(models, preprocess)
    vector_verification = verify_test_vectors(MODEL_DIR, tolerance=1e-12)
    verify_receipt_contract(receipt, vector_verification)
    input_hashes = collect_input_hashes(qa, receipt)
    return models, preprocess, thresholds, qa, receipt, input_hashes


def _directory_entries(path: Path) -> list[Path]:
    if not path.exists():
        return []
    if not path.is_dir() or path.is_symlink():
        raise RuntimeError(f"U0 output path is not a real directory: {path}")
    return sorted(path.iterdir())


def require_clean_output() -> None:
    entries = _directory_entries(OUTPUT)
    if entries:
        raise RuntimeError(
            "U0 output is not empty; refusing to mix or overwrite results: "
            + ", ".join(item.name for item in entries[:10])
        )


def strict_ids(frame: pd.DataFrame, label: str) -> None:
    required = ["patient_id", "case_id"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"{label} is missing identifier columns: {missing}")
    for column in required:
        values = frame[column]
        if values.isna().any():
            raise ValueError(f"{label} {column} contains missing values")
        as_string = values.astype("string")
        stripped = as_string.str.strip()
        if (stripped != as_string).any() or stripped.eq("").any():
            raise ValueError(f"{label} {column} contains blank or padded identifiers")
        forbidden = stripped.str.lower().isin({"nan", "none", "null", "<na>"})
        if forbidden.any():
            raise ValueError(f"{label} {column} contains placeholder identifiers")
        if stripped.duplicated().any():
            raise ValueError(f"{label} {column} is not unique")
    if frame.duplicated(required).any():
        raise ValueError(f"{label} patient/case keys are not unique")


def strict_binary(series: pd.Series, label: str) -> np.ndarray:
    if series.isna().any():
        raise ValueError(f"{label} contains missing values")
    values = pd.to_numeric(series, errors="raise").to_numpy(dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError(f"{label} contains non-finite values")
    if not np.isin(values, [0.0, 1.0]).all():
        observed = np.unique(values)[:10].tolist()
        raise ValueError(f"{label} is not strictly binary 0/1: {observed}")
    return values.astype(np.int8)


def validate_predictions(y: np.ndarray, predictions: dict[str, np.ndarray]) -> None:
    if len(np.unique(y)) != 2:
        raise ValueError("primary external validation requires both outcome classes")
    if not predictions:
        raise ValueError("frozen bundle generated no model predictions")
    for name, probability in predictions.items():
        values = np.asarray(probability, dtype=float)
        if values.ndim != 1 or len(values) != len(y):
            raise ValueError(f"{name} prediction length or shape is invalid")
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError(f"{name} generated non-finite or out-of-range probabilities")
        logits = np.log(np.clip(values, 1e-12, 1 - 1e-12) / np.clip(1 - values, 1e-12, 1))
        if float(np.var(logits)) <= 1e-12:
            raise ValueError(f"{name} prediction logits are constant; calibration is not identifiable")


def verify_result_manifest(directory: Path, expected_files: list[str]) -> str:
    manifest_path = directory / RESULT_MANIFEST_NAME
    manifest = _read_hash_manifest(manifest_path, "file")
    if manifest["file"].str.contains(r"[/\\]", regex=True).any():
        raise RuntimeError("U0 result manifest contains a non-basename path")
    if set(manifest["file"]) != set(expected_files) or len(manifest) != len(expected_files):
        raise RuntimeError("U0 result manifest file inventory differs from the explicit contract")
    actual_entries = _directory_entries(directory)
    if any(not path.is_file() or path.is_symlink() for path in actual_entries):
        raise RuntimeError("U0 result directory contains a subdirectory or symbolic link")
    actual_names = {path.name for path in actual_entries}
    if actual_names != set(expected_files) | {RESULT_MANIFEST_NAME}:
        raise RuntimeError("U0 result directory contains unmanifested or missing files")
    failures = []
    for row in manifest.itertuples(index=False):
        path = directory / row.file
        observed_sha = sha256_file(path)
        observed_size = path.stat().st_size
        if observed_sha != row.sha256 or observed_size != int(row.size_bytes):
            failures.append(row.file)
    if failures:
        raise RuntimeError(f"U0 result hash verification failed: {failures[:5]}")
    return sha256_file(manifest_path)


def _attempt_history_entry(previous: dict) -> dict:
    return {
        key: previous.get(key)
        for key in [
            "attempt_number",
            "status",
            "pid",
            "hostname",
            "started_at_utc",
            "failed_at_utc",
            "error_type",
            "error",
            "input_sha256",
            "staging_directory",
            "expected_result_manifest_sha256",
        ]
    }


def finalize_already_published(previous: dict, input_hashes: dict[str, str]) -> bool:
    entries = _directory_entries(OUTPUT)
    if not entries:
        return False
    expected_files = previous.get("expected_result_files")
    expected_manifest = previous.get("expected_result_manifest_sha256")
    if not isinstance(expected_files, list) or not expected_manifest:
        raise RuntimeError("nonempty U0 output lacks a recorded atomic-publish contract")
    if previous.get("pre_publish_input_sha256") != input_hashes:
        raise RuntimeError("published output lacks the exact pre-publish input snapshot")
    if not all(previous.get(key) is not None for key in ["n_patients", "n_events", "bootstrap_repetitions"]):
        raise RuntimeError("published output lacks required pre-publish run metadata")
    observed_manifest = verify_result_manifest(OUTPUT, expected_files)
    if observed_manifest != expected_manifest:
        raise RuntimeError("published U0 manifest differs from the pre-publish lock")
    if collect_input_hashes(
        _load_json(QA_INPUT, "MOVER pre-model QA report"),
        _load_json(FREEZE_RECEIPT, "pre-U0 freeze receipt"),
    ) != input_hashes:
        raise RuntimeError("locked inputs changed after the atomic result publish")
    complete = dict(previous)
    complete.update(
        {
            "status": "COMPLETE",
            "completed_at_utc": now(),
            "completion_recovered_from_atomic_publish": True,
            "completion_input_sha256": input_hashes,
            "model_and_threshold_hashes_verified_at_completion": True,
        }
    )
    atomic_replace_json(LOCK, complete)
    log("Recovered a fully verified atomic publish and marked U0 COMPLETE")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume-technical",
        action="store_true",
        help="Permit an exact-input retry only after a recorded technical failure before completion.",
    )
    args = parser.parse_args()
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    # This gate reads contracts and hashes files only.  No MOVER outcome aggregate
    # or model prediction is produced before the one-time lock is acquired.
    models, preprocess, thresholds_payload, qa, receipt, input_hashes = (
        load_pre_u0_contract()
    )

    previous: dict | None = None
    prior_attempts: list[dict] = []
    attempt_number = 1
    if LOCK.exists():
        previous = _load_json(LOCK, "U0 execution lock")
        if previous.get("study_id") != CONFIG["study_id"]:
            raise SystemExit("Existing U0 lock belongs to a different study_id")
        if previous.get("status") == "COMPLETE":
            raise SystemExit("U0 is already complete; rerun is prohibited")
        if not args.resume_technical:
            raise SystemExit(
                "A prior incomplete U0 attempt exists; use --resume-technical only after review."
            )
        allowed = {"RUNNING", "FAILED_TECHNICAL", "PUBLISHING"}
        if previous.get("status") not in allowed:
            raise SystemExit(f"Unsupported prior U0 lock state: {previous.get('status')}")
        if previous.get("status") in {"RUNNING", "PUBLISHING"} and process_is_running(
            previous.get("pid")
        ):
            raise SystemExit("The prior U0 process is still running; concurrent execution is prohibited")
        if previous.get("input_sha256") != input_hashes:
            raise SystemExit("Technical retry inputs differ from the first attempt; rerun prohibited")
        if finalize_already_published(previous, input_hashes):
            return
        require_clean_output()
        recorded_attempts = previous.get("technical_attempts", [])
        if not isinstance(recorded_attempts, list):
            raise SystemExit("Existing U0 lock has an invalid attempt history")
        prior_attempts = list(recorded_attempts)
        prior_attempts.append(_attempt_history_entry(previous))
        attempt_number = int(previous.get("attempt_number", 0)) + 1
    else:
        require_clean_output()
        orphan_staging = sorted(OUTPUT.parent.glob(".H5_primary_U0_staging_attempt_*"))
        if orphan_staging:
            raise SystemExit("Untracked U0 staging directories exist without a lock; review required")

    vector_verification = verify_test_vectors(MODEL_DIR, tolerance=1e-12)
    lock = {
        "study_id": CONFIG["study_id"],
        "status": "RUNNING",
        "attempt_number": attempt_number,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "started_at_utc": now(),
        "input_sha256": input_hashes,
        "qa_gate_sha256": sha256_file(QA_INPUT),
        "pre_U0_freeze_receipt_sha256": sha256_file(FREEZE_RECEIPT),
        "test_vector_verification": vector_verification,
        "model_state": "U0_no_update",
        "primary_observation_process": "H5",
        "model_or_threshold_refitting_in_MOVER": False,
        "technical_attempts": prior_attempts,
    }
    if previous is None:
        atomic_create_json(LOCK, lock)
    else:
        atomic_replace_json(LOCK, lock)

    stage_dir = OUTPUT.parent / (
        f".H5_primary_U0_staging_attempt_{attempt_number:03d}_"
        f"{os.getpid()}_{time.time_ns()}"
    )
    result_files: list[str] = []

    try:
        stage_dir.mkdir(mode=0o700)
        lock["staging_directory"] = stage_dir.name
        atomic_replace_json(LOCK, lock)

        data = pd.read_csv(
            MOVER_INPUT,
            dtype={"patient_id": "string", "case_id": "string"},
            low_memory=False,
        )
        strict_ids(data, "MOVER main stage-2 input")
        if "primary_outcome" not in data:
            raise ValueError("MOVER main stage-2 input lacks primary_outcome")
        y = strict_binary(data["primary_outcome"], "MOVER main primary_outcome")
        patient_ids = data["patient_id"].astype(str).to_numpy()

        predictions: dict[str, np.ndarray] = {}
        for name, model in models["models"].items():
            features = model["feature_order"]
            medians = preprocess["models"][name]["imputation_medians"]
            X = apply_imputation(data, features, medians, enforce_structural_gate=True)
            predictions[name] = predict_model_dict(model, X)
        validate_predictions(y, predictions)
        log(f"U0 raw frozen probabilities generated for {len(data):,} patients")

        workload_thresholds, clinical_thresholds, threshold_table = locked_thresholds(
            thresholds_payload
        )

        def write_csv(frame: pd.DataFrame, filename: str, **kwargs: object) -> None:
            if filename in result_files or filename == RESULT_MANIFEST_NAME:
                raise RuntimeError(f"duplicate or reserved U0 output filename: {filename}")
            frame.to_csv(stage_dir / filename, index=False, **kwargs)
            result_files.append(filename)

        write_csv(
            threshold_table,
            "U0_fixed_thresholds_applied.csv",
            encoding="utf-8-sig",
        )

        point_rows = []
        bin_parts = []
        for name, probability in predictions.items():
            row, bins = point_performance(y, probability)
            row.update(
                {
                    "dataset": "MOVER_EPIC",
                    "observation_process": "H5",
                    "model_state": "U0_no_update",
                    "model": name,
                }
            )
            point_rows.append(row)
            bins.insert(0, "model", name)
            bin_parts.append(bins)
        point_table = pd.DataFrame(point_rows)
        required_point = [
            "auroc",
            "auprc",
            "brier",
            "calibration_in_the_large",
            "calibration_slope",
            "observed_expected_ratio",
            "ici_equal_frequency",
        ]
        if not np.isfinite(point_table[required_point].to_numpy(dtype=float)).all():
            raise RuntimeError("a primary point-performance estimand is non-finite")
        write_csv(
            point_table,
            "U0_external_validation_point_estimates.csv",
            encoding="utf-8-sig",
        )
        write_csv(
            pd.concat(bin_parts, ignore_index=True),
            "U0_calibration_equal_frequency_bins.csv",
            encoding="utf-8-sig",
        )

        workload, _ = threshold_point_tables(
            y, predictions, patient_ids, workload_thresholds
        )
        _, dca_fixed = threshold_point_tables(
            y, predictions, patient_ids, clinical_thresholds
        )
        workload = workload.merge(threshold_table, on="threshold", how="left")
        clinical_table = threshold_table[
            threshold_table["threshold_type"].str.contains("clinical_action")
        ]
        dca_fixed = dca_fixed.merge(clinical_table, on="threshold", how="left")
        if set(dca_fixed["threshold"].astype(float)) != set(clinical_thresholds):
            raise RuntimeError("fixed net-benefit output contains a non-clinical threshold")
        write_csv(
            workload,
            "U0_fixed_threshold_alert_workload.csv",
            encoding="utf-8-sig",
        )
        write_csv(
            dca_fixed,
            "U0_fixed_threshold_net_benefit.csv",
            encoding="utf-8-sig",
        )

        dca_start, dca_end = [float(value) for value in CONFIG["thresholds"]["dca_full_range"]]
        dca_step = float(CONFIG["thresholds"]["dca_step"])
        if not (
            np.isclose(dca_start, 0.05, rtol=0, atol=1e-12)
            and np.isclose(dca_end, 0.50, rtol=0, atol=1e-12)
            and np.isclose(dca_step, 0.01, rtol=0, atol=1e-12)
        ):
            raise ValueError("full decision-curve threshold grid must remain 0.05-0.50 by 0.01")
        dca_grid = np.round(
            np.arange(dca_start, dca_end + 1e-12, dca_step),
            6,
        )
        dca_parts = []
        for name, probability in predictions.items():
            table = decision_curve(y, probability, dca_grid)
            table.insert(0, "model", name)
            dca_parts.append(table)
        write_csv(
            pd.concat(dca_parts, ignore_index=True),
            "U0_decision_curve_0.05_0.50.csv",
            encoding="utf-8-sig",
        )

        configured_n_boot = CONFIG["external_validation"][
            "minimum_patient_bootstrap_repetitions"
        ]
        if isinstance(configured_n_boot, bool) or float(configured_n_boot) != int(configured_n_boot):
            raise ValueError("patient bootstrap repetition count must be an integer")
        n_boot = int(configured_n_boot)
        if n_boot < 2000:
            raise ValueError("primary patient bootstrap must request at least 2,000 repetitions")
        log(f"Starting {n_boot:,}-replicate non-stratified paired patient bootstrap")
        bootstrap_summary, bootstrap_replicates, rcs_curve = paired_bootstrap_validation(
            y,
            predictions,
            patient_ids,
            workload_thresholds,
            primary_model=models["primary_model"],
            net_benefit_thresholds=clinical_thresholds,
            n_boot=n_boot,
            random_state=int(CONFIG["random_seed"]),
        )
        if bootstrap_summary["estimand"].duplicated().any():
            raise RuntimeError("paired bootstrap summary contains duplicate estimands")
        primary_core = bootstrap_summary[
            bootstrap_summary["estimand"].isin(
                [
                    f"{models['primary_model']}__{metric}"
                    for metric in [
                        "auroc",
                        "auprc",
                        "brier",
                        "calibration_slope",
                        "calibration_in_the_large",
                        "observed_expected_ratio",
                        "ici_equal_frequency",
                    ]
                ]
            )
        ]
        if len(primary_core) != 7 or (
            primary_core["valid_replicates"] < math.ceil(0.95 * n_boot)
        ).any():
            raise RuntimeError("fewer than 95% valid replicates for a primary bootstrap estimand")
        if not np.isfinite(primary_core["estimate"].to_numpy(dtype=float)).all():
            raise RuntimeError("a primary bootstrap point estimand is non-finite")
        if (
            "valid_bootstrap_curves" not in rcs_curve
            or int(rcs_curve["valid_bootstrap_curves"].min()) < math.ceil(0.95 * n_boot)
        ):
            raise RuntimeError("fewer than 95% valid RCS calibration bootstrap curves")
        write_csv(
            bootstrap_summary,
            f"U0_paired_patient_bootstrap_{n_boot}_summary.csv",
            encoding="utf-8-sig",
        )
        write_csv(
            bootstrap_replicates,
            f"U0_paired_patient_bootstrap_{n_boot}_replicates.csv.gz",
            compression={"method": "gzip", "compresslevel": 5},
        )
        write_csv(
            rcs_curve,
            "U0_primary_RCS_calibration_curve_with_95CI.csv",
            encoding="utf-8-sig",
        )

        identifiers = pd.DataFrame(
            {
                "patient_hash": data["patient_id"].map(
                    lambda value: hashlib.sha256(
                        f"{CONFIG['study_id']}|MOVER_patient|{value}".encode()
                    ).hexdigest()
                ),
                "case_hash": data["case_id"].map(
                    lambda value: hashlib.sha256(
                        f"{CONFIG['study_id']}|MOVER_case|{value}".encode()
                    ).hexdigest()
                ),
                "primary_outcome": y,
            }
        )
        for name, probability in predictions.items():
            identifiers[f"probability_{name}"] = probability
        write_csv(
            identifiers,
            "U0_hashed_individual_predictions.csv.gz",
            compression={"method": "gzip", "compresslevel": 5},
        )

        # Fixed two-stage clinical flow: stage 1 is always alerted; only stage 2
        # receives the continuous frozen probability.  This is a binary strategy,
        # so it has no mixed-population continuous DCA curve.
        first_all = pd.read_csv(
            FIRST_INPUT,
            dtype={"patient_id": "string", "case_id": "string"},
            low_memory=False,
        )
        strict_ids(first_all, "MOVER first fully evaluable operation input")
        required_first = {"stage1_high_alert", "primary_outcome"}
        if not required_first.issubset(first_all):
            raise ValueError(f"first-operation input lacks columns: {sorted(required_first - set(first_all))}")
        first_y = strict_binary(first_all["primary_outcome"], "two-stage primary_outcome")
        stage1_numeric = strict_binary(first_all["stage1_high_alert"], "stage1_high_alert")
        stage1_alert = stage1_numeric.astype(bool)

        key_columns = ["patient_id", "case_id"]
        expected_stage2 = first_all.loc[~stage1_alert, key_columns]
        key_audit = expected_stage2.merge(
            data[key_columns],
            on=key_columns,
            how="outer",
            validate="one_to_one",
            indicator=True,
        )
        if not key_audit["_merge"].eq("both").all():
            raise AssertionError("main stage-2 keys do not exactly equal non-stage1 first-operation keys")

        prediction_frame = data[key_columns].copy()
        prediction_frame["stage2_primary_outcome"] = y
        prediction_frame["stage2_probability"] = predictions[models["primary_model"]]
        joined = first_all[key_columns].copy()
        joined["first_primary_outcome"] = first_y
        joined["stage1_high_alert"] = stage1_numeric
        joined = joined.merge(
            prediction_frame,
            on=key_columns,
            how="left",
            validate="one_to_one",
            indicator=True,
        )
        if not joined.loc[joined["stage1_high_alert"].eq(0), "_merge"].eq("both").all():
            raise AssertionError("two-stage flow has a stage-2 patient without a frozen probability")
        if not joined.loc[joined["stage1_high_alert"].eq(1), "_merge"].eq("left_only").all():
            raise AssertionError("a stage-1 direct-alert patient incorrectly entered model prediction")
        stage2_rows = joined["stage1_high_alert"].eq(0)
        if not np.array_equal(
            joined.loc[stage2_rows, "first_primary_outcome"].to_numpy(dtype=np.int8),
            joined.loc[stage2_rows, "stage2_primary_outcome"].to_numpy(dtype=np.int8),
        ):
            raise AssertionError("primary outcomes disagree between main and first-operation files")
        stage2_probability = joined["stage2_probability"].to_numpy(dtype=float)

        two_stage_summary, two_stage_replicates = two_stage_strategy_bootstrap(
            first_y,
            stage1_numeric,
            stage2_probability,
            first_all["patient_id"].astype(str).to_numpy(),
            workload_thresholds,
            net_benefit_thresholds=clinical_thresholds,
            n_boot=n_boot,
            random_state=int(CONFIG["random_seed"]),
        )
        two_stage_summary = two_stage_summary.merge(threshold_table, on="threshold", how="left")
        if two_stage_summary.duplicated(["threshold", "metric", "threshold_name"]).any():
            raise RuntimeError("two-stage bootstrap summary contains duplicate estimands")
        essential_two_stage = two_stage_summary[
            two_stage_summary["metric"].isin(
                [
                    "alert_rate",
                    "sensitivity",
                    "specificity",
                    "fixed_binary_strategy_net_benefit",
                ]
            )
        ]
        if essential_two_stage.empty or (
            essential_two_stage["valid_replicates"] < math.ceil(0.95 * n_boot)
        ).any():
            raise RuntimeError("fewer than 95% valid two-stage bootstrap replicates")
        if not np.isfinite(essential_two_stage["estimate"].to_numpy(dtype=float)).all():
            raise RuntimeError("an essential two-stage estimand is non-finite")
        write_csv(
            two_stage_summary,
            f"U0_two_stage_fixed_strategy_bootstrap_{n_boot}_summary.csv",
            encoding="utf-8-sig",
        )
        write_csv(
            two_stage_replicates,
            f"U0_two_stage_fixed_strategy_bootstrap_{n_boot}_replicates.csv.gz",
            compression={"method": "gzip", "compresslevel": 5},
        )

        if len(result_files) != len(set(result_files)):
            raise RuntimeError("explicit U0 result inventory contains duplicate names")
        result_hashes = [
            {
                "file": filename,
                "size_bytes": (stage_dir / filename).stat().st_size,
                "sha256": sha256_file(stage_dir / filename),
            }
            for filename in result_files
        ]
        pd.DataFrame(result_hashes).to_csv(
            stage_dir / RESULT_MANIFEST_NAME, index=False, encoding="utf-8-sig"
        )
        result_manifest_sha = verify_result_manifest(stage_dir, result_files)

        input_hashes_before_publish = collect_input_hashes(qa, receipt)
        if input_hashes_before_publish != input_hashes:
            raise RuntimeError("a locked U0 input changed during computation; publish prohibited")
        lock.update(
            {
                "status": "PUBLISHING",
                "publish_started_at_utc": now(),
                "n_patients": len(data),
                "n_events": int(y.sum()),
                "bootstrap_repetitions": n_boot,
                "expected_result_files": result_files,
                "expected_result_manifest_sha256": result_manifest_sha,
                "pre_publish_input_sha256": input_hashes_before_publish,
            }
        )
        atomic_replace_json(LOCK, lock)

        require_clean_output()
        if OUTPUT.exists():
            OUTPUT.rmdir()
        os.replace(stage_dir, OUTPUT)
        _fsync_directory(OUTPUT.parent)
        if verify_result_manifest(OUTPUT, result_files) != result_manifest_sha:
            raise RuntimeError("atomic-published result manifest failed verification")
        completion_hashes = collect_input_hashes(qa, receipt)
        if completion_hashes != input_hashes:
            raise RuntimeError("a locked U0 input changed before completion was recorded")
        lock.update(
            {
                "status": "COMPLETE",
                "completed_at_utc": now(),
                "completion_input_sha256": completion_hashes,
                "model_and_threshold_hashes_verified_at_completion": True,
                "completion_recovered_from_atomic_publish": False,
            }
        )
        atomic_replace_json(LOCK, lock)
        log("U0 one-time external validation COMPLETE and locked")
    except BaseException as error:
        lock.update(
            {
                "status": "FAILED_TECHNICAL",
                "failed_at_utc": now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "performance_guided_changes_authorized": False,
            }
        )
        try:
            atomic_replace_json(LOCK, lock)
        except BaseException:
            pass
        raise


if __name__ == "__main__":
    main()

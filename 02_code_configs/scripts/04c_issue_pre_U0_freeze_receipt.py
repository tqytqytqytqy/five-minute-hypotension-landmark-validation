#!/usr/bin/env python3
"""Issue an append-only, external-to-bundle receipt before MOVER U0."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_code_configs/src"))

from lm5_validation.frozen import sha256_file, verify_test_vectors  # noqa: E402


MODEL = ROOT / "04_frozen_INSPIRE_LM5_model"
RECEIPT = ROOT / "10_run_logs_manifest/pre_U0_freeze_receipt.json"


def main() -> None:
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    verification = verify_test_vectors(MODEL, tolerance=1e-12)
    core = [
        "model.json",
        "preprocess.json",
        "thresholds.json",
        "feature_contract.json",
        "cohort_endpoint.json",
        "test_vectors.csv",
        "freeze_provenance.json",
        "SHA256SUMS.csv",
        "code_SHA256SUMS.csv",
    ]
    secondary_prespec = {
        "00_protocol_SAP/secondary_analysis_prespec.json": (
            ROOT / "00_protocol_SAP/secondary_analysis_prespec.json"
        ),
        "02_code_configs/scripts/07_secondary_analyses.py": (
            ROOT / "02_code_configs/scripts/07_secondary_analyses.py"
        ),
    }
    payload = {
        "study_id": "LM5_COMMON18_INSPIRE_MOVER_20260712",
        "receipt_version": "1.0",
        "freeze_version": "1.1.0",
        "issued_at_utc": datetime.now(timezone.utc).isoformat(),
        "receipt_location": "outside writable model bundle",
        "MOVER_patient_level_predictions_seen": False,
        "MOVER_performance_seen": False,
        "core_file_sha256": {name: sha256_file(MODEL / name) for name in core},
        "secondary_analysis_file_sha256": {
            name: sha256_file(path) for name, path in secondary_prespec.items()
        },
        "test_vector_verification": verification,
    }
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(RECEIPT, flags, 0o444)
    try:
        os.write(fd, encoded)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.chmod(RECEIPT, 0o444)
    print(json.dumps({"status": "ISSUED", "path": str(RECEIPT), "sha256": sha256_file(RECEIPT)}))


if __name__ == "__main__":
    main()

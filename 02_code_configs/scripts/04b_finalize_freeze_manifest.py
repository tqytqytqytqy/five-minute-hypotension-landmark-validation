#!/usr/bin/env python3
"""Finalize code/config lineage without changing any fitted model object."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_code_configs/src"))

from lm5_validation.frozen import sha256_file, sha256_tree, verify_test_vectors  # noqa: E402


MODEL_DIR = ROOT / "04_frozen_INSPIRE_LM5_model"


def main() -> None:
    immutable_model_files = ["model.json", "preprocess.json", "thresholds.json", "test_vectors.csv"]
    before = {name: sha256_file(MODEL_DIR / name) for name in immutable_model_files}
    code_paths = [
        ROOT / "00_protocol_SAP/SAP_v1.0.md",
        ROOT / "00_protocol_SAP/deviation_log.csv",
        ROOT / "02_code_configs/configs/analysis.json",
        ROOT / "02_code_configs/configs/paths.json",
    ]
    code_paths += sorted((ROOT / "02_code_configs/src/lm5_validation").glob("*.py"))
    code_paths += [
        ROOT / f"02_code_configs/scripts/{name}"
        for name in [
            "01_source_audit.py",
            "02_extract_sources.py",
            "03_build_cohort.py",
            "04_fit_freeze_inspire.py",
            "04b_finalize_freeze_manifest.py",
            "04c_issue_pre_U0_freeze_receipt.py",
            "05_mover_pre_model_qa.py",
            "06_run_mover_u0_once.py",
        ]
    ]
    rows = []
    for path in code_paths:
        rows.append(
            {
                "relative_path_from_project_root": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "role": "primary_protocol_code_config",
            }
        )
    pd.DataFrame(rows).to_csv(
        MODEL_DIR / "code_SHA256SUMS.csv", index=False, encoding="utf-8-sig"
    )
    provenance_path = MODEL_DIR / "freeze_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance.update(
        {
            "technical_freeze_finalized_at_utc": datetime.now(timezone.utc).isoformat(),
            "technical_freeze_version": "1.1.0",
            "technical_freeze_extension": "post-audit INSPIRE-only refreeze with expanded ridge grid and corrected precision formula; this finalization step itself adds the primary code/config hash manifest without changing the newly refitted objects",
            "immutable_model_file_hashes_before_extension": before,
            "code_manifest_file": "code_SHA256SUMS.csv",
        }
    )
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    after = {name: sha256_file(MODEL_DIR / name) for name in immutable_model_files}
    if before != after:
        raise AssertionError("technical freeze extension changed a model object")
    verification = verify_test_vectors(MODEL_DIR, tolerance=1e-12)
    provenance["test_vector_verification_after_extension"] = verification
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    sha256_tree(MODEL_DIR).to_csv(
        MODEL_DIR / "SHA256SUMS.csv", index=False, encoding="utf-8-sig"
    )
    print(json.dumps({"status": "FINALIZED", "code_files": len(rows), **verification}))


if __name__ == "__main__":
    main()

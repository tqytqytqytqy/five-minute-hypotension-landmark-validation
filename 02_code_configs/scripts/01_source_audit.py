#!/usr/bin/env python3
"""Create a read-only source gate and cryptographic lineage manifest."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PATHS = json.loads((ROOT / "02_code_configs/configs/paths.json").read_text(encoding="utf-8"))
OUT = ROOT / "01_source_audit_lineage"

EXPECTED_MD5 = {
    "mover_epic_emr": "ad8756e62a8316a896580f6c6946ec2b",
    "mover_epic_flowsheets_cleaned": "c4f86c9618d07636ba69f1c27f25025a",
    # This hash belongs to the complete file. The local partial download must not match it.
    "mover_epic_patient_measurements_forbidden": "629dcfba88878cf407ca4e61fdf23fb5",
}

EXPECTED_INSPIRE_SHA256 = "abfe6fd97ec902caab9fe7d75a32090a31d8e0691bcddd1326b7b8360e1d0a4d"


def hashes(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> tuple[str, str]:
    md5 = hashlib.md5(usedforsecurity=False)
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            md5.update(block)
            sha.update(block)
    return md5.hexdigest(), sha.hexdigest()


def file_record(role: str, path_string: str) -> dict:
    path = Path(path_string)
    record = {
        "role": role,
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat() if path.exists() else None,
        "md5": None,
        "sha256": None,
        "expected_md5": EXPECTED_MD5.get(role),
        "expected_sha256": EXPECTED_INSPIRE_SHA256 if role == "inspire_zip" else None,
        "hash_match": False,
        "permitted_for_primary_analysis": role not in {
            "mover_epic_patient_measurements_forbidden",
            "mover_sis_emr_sensitivity_only",
            "legacy_inspire_v3_read_only",
            "legacy_mover_map_cache_qa_only",
            "vitaldb_raw_root",
        },
    }
    if path.exists() and path.is_file():
        record["md5"], record["sha256"] = hashes(path)
        if record["expected_md5"]:
            record["hash_match"] = record["md5"] == record["expected_md5"]
        elif record["expected_sha256"]:
            record["hash_match"] = record["sha256"] == record["expected_sha256"]
        else:
            record["hash_match"] = True
    elif path.exists() and path.is_dir():
        record["hash_match"] = True
    return record


def inspire_members(path: Path) -> list[dict]:
    with zipfile.ZipFile(path) as archive:
        return [
            {
                "member": item.filename,
                "uncompressed_bytes": item.file_size,
                "compressed_bytes": item.compress_size,
                "crc32": f"{item.CRC:08x}",
            }
            for item in archive.infolist()
            if not item.is_dir()
        ]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    roles = [
        "inspire_zip",
        "mover_epic_emr",
        "mover_epic_flowsheets_cleaned",
        "mover_epic_patient_measurements_forbidden",
        "mover_sis_emr_sensitivity_only",
        "vitaldb_raw_root",
        "legacy_inspire_v3_read_only",
        "legacy_mover_map_cache_qa_only",
    ]
    records = [file_record(role, PATHS[role]) for role in roles]
    by_role = {row["role"]: row for row in records}

    core_ok = all(
        by_role[name]["exists"] and by_role[name]["hash_match"]
        for name in ["inspire_zip", "mover_epic_emr", "mover_epic_flowsheets_cleaned"]
    )
    forbidden_partial = by_role["mover_epic_patient_measurements_forbidden"]
    forbidden_status = (
        "present_but_incomplete_and_forbidden"
        if forbidden_partial["exists"] and not forbidden_partial["hash_match"]
        else "forbidden_even_if_complete"
    )
    gate = {
        "study_id": "LM5_COMMON18_INSPIRE_MOVER_20260712",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "primary_source_gate": "GREEN" if core_ok else "RED",
        "core_sources": ["inspire_zip", "mover_epic_emr", "mover_epic_flowsheets_cleaned"],
        "forbidden_patient_measurements_status": forbidden_status,
        "mover_map_source": "Epic_flowsheets_cleaned.tar.gz only",
        "legacy_caches": "QA-only; never accepted as the formal H5/R1 source",
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    (OUT / "source_manifest.json").write_text(
        json.dumps({"gate": gate, "files": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (OUT / "source_manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    if by_role["inspire_zip"]["exists"]:
        (OUT / "inspire_zip_members.json").write_text(
            json.dumps(inspire_members(Path(PATHS["inspire_zip"])), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    lines = [
        "# 原始数据源门报告",
        "",
        f"- 主数据源门：**{gate['primary_source_gate']}**",
        f"- INSPIRE 1.4.2 SHA-256匹配：{by_role['inspire_zip']['hash_match']}",
        f"- MOVER EPIC_EMR官方MD5匹配：{by_role['mover_epic_emr']['hash_match']}",
        f"- MOVER cleaned flowsheets官方MD5匹配：{by_role['mover_epic_flowsheets_cleaned']['hash_match']}",
        f"- 禁用patient_measurments状态：{forbidden_status}",
        "- 正式MOVER MAP输入仅为完整cleaned flowsheets；旧20–180缓存、LOCF分钟缓存和残缺patient_measurments均不得进入主分析。",
    ]
    (OUT / "source_gate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    if not core_ok:
        raise SystemExit("Primary source gate RED")


if __name__ == "__main__":
    main()


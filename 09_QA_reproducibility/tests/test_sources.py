from __future__ import annotations

import gzip
import io
import json
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "02_code_configs" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lm5_validation.sources import (  # noqa: E402
    CacheMismatchError,
    SourceSchemaError,
    extract_inspire_source,
    extract_mover_source,
    iter_mover_map_observations,
    parse_height_m,
    parse_mover_weight_kg,
)


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _write_tar(path: Path, members: dict[str, pd.DataFrame]) -> None:
    with tarfile.open(path, mode="w:gz") as archive:
        for name, frame in members.items():
            payload = _csv_bytes(frame)
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def _inspire_operations() -> pd.DataFrame:
    base = {
        "hadm_id": 1,
        "case_id": np.nan,
        "opdate": 0,
        "sex": "M",
        "weight": 80,
        "height": 180,
        "race": "Other",
        "asa": 2,
        "emop": 0,
        "icd10_pcs": "0ABC0",
        "orin_time": 95,
        "orout_time": 230,
        "opstart_time": 120,
        "opend_time": 210,
        "admission_time": 0,
        "discharge_time": 1000,
        "cpbon_time": np.nan,
        "cpboff_time": np.nan,
        "icuin_time": np.nan,
        "icuout_time": np.nan,
        "inhosp_death_time": np.nan,
        "allcause_death_time": np.nan,
    }
    rows = []

    def add(op_id: int, subject: int, **changes: object) -> None:
        row = {
            **base,
            "op_id": op_id,
            "subject_id": subject,
            "age": 60,
            "department": "GS",
            "antype": "General",
            "anstart_time": 100,
            "anend_time": 220,
        }
        row.update(changes)
        rows.append(row)

    add(401, 101)
    add(402, 102, age=17)
    add(403, 103, department="CTS")
    add(404, 104, department="OG")
    add(405, 105, antype="Neuraxial")
    add(406, 106, anend_time=np.nan)
    add(407, 101, opdate=5000, anstart_time=5100, anend_time=5220)
    return pd.DataFrame(rows)


def _write_inspire_zip(path: Path) -> None:
    vitals = pd.DataFrame(
        [
            [401, 101, 105, "art_mbp", 20],
            [401, 101, 110, "nibp_mbp", 200],
            [401, 101, 115, "art_mbp", 19],
            [401, 101, 120, "nibp_mbp", 201],
            [401, 101, 125, "hr", 70],
            [401, 101, 95, "art_mbp", 60],
            [402, 102, 105, "art_mbp", 55],
            [407, 101, 5105, "art_mbp", 50],
        ],
        columns=["op_id", "subject_id", "chart_time", "item_name", "value"],
    )
    operations = _inspire_operations()
    prefix = "inspire-a-publicly-available-research-dataset-for-perioperative-medicine-1.4.2"
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(f"{prefix}/operations.csv.gz", gzip.compress(_csv_bytes(operations)))
        archive.writestr(f"{prefix}/vitals.csv.gz", gzip.compress(_csv_bytes(vitals)))
        # The real release also contains ward_vitals.csv.gz; exact basename
        # matching must not mistake it for a second intraoperative vitals file.
        archive.writestr(
            f"{prefix}/ward_vitals.csv.gz",
            gzip.compress(_csv_bytes(vitals.iloc[0:0])),
        )


def _mover_patients() -> pd.DataFrame:
    base = {
        "HEIGHT": "5' 10\"",
        "WEIGHT": "2821.9176",
        "SEX": "M",
        "ASA_RATING_C": 2,
        "PATIENT_CLASS_GROUP": "Inpatient",
        "PATIENT_CLASS_NM": "Elective",
        "IN_OR_DTTM": "2024-01-01 07:50:00",
        "OUT_OR_DTTM": "2024-01-01 10:10:00",
    }
    rows = []

    def add(log_id: str, mrn: str, **changes: object) -> None:
        row = {
            **base,
            "LOG_ID": log_id,
            "MRN": mrn,
            "BIRTH_DATE": 60,
            "PRIMARY_ANES_TYPE_NM": "General",
            "PRIMARY_PROCEDURE_NM": "TOTAL HIP ARTHROPLASTY",
            "AN_START_DATETIME": "2024-01-01 08:00:00",
            "AN_STOP_DATETIME": "2024-01-01 10:00:00",
        }
        row.update(changes)
        rows.append(row)

    add("E1", "P1")
    add("E2", "P2", BIRTH_DATE=16)
    add("E3", "P3", PRIMARY_PROCEDURE_NM="CABG WITH CARDIOPULMONARY BYPASS")
    add("E4", "P4", PRIMARY_PROCEDURE_NM="CESAREAN SECTION")
    add("E5", "P5", PRIMARY_ANES_TYPE_NM="MAC")
    add("E6", "P6", AN_STOP_DATETIME="2024-01-01 07:59:00")
    add(
        "E7",
        "P1",
        AN_START_DATETIME="2024-02-01 08:00:00",
        AN_STOP_DATETIME="2024-02-01 10:00:00",
    )
    return pd.DataFrame(rows)


def _write_mover_archives(epic_path: Path, flows_path: Path) -> None:
    _write_tar(
        epic_path,
        {"EPIC_EMR/EMR/patient_information.csv": _mover_patients()},
    )
    columns = [
        "LOG_ID",
        "MRN",
        "FLO_NAME",
        "FLO_DISPLAY_NAME",
        "RECORD_TYPE",
        "RECORDED_TIME",
        "MEAS_VALUE",
        "UNITS",
    ]
    part1 = pd.DataFrame(
        [
            ["E1", "P1", "ART MAP", "Arterial MAP", "value", "2024-01-01 08:05:00", "60", "mmHg"],
            ["E1", "P1", "NIBP MAP", "Cuff MAP", "value", "2024-01-01 08:10:00", "70", "mmHg"],
            ["E1", "P1", "MAP", "Mean arterial pressure", "value", "2024-01-01 08:20:00", "80", "mmHg"],
            ["E1", "P1", "MAP goal", "MAP target", "value", "2024-01-01 08:25:00", "75", "mmHg"],
            ["E1", "P1", "BP", "Blood pressure", "value", "2024-01-01 08:30:00", "120/80", "mmHg"],
            ["E2", "P2", "ART MAP", "Arterial MAP", "value", "2024-01-01 08:05:00", "55", "mmHg"],
        ],
        columns=columns,
    )
    part2 = pd.DataFrame(
        [
            ["E1", "P1", "MAP", "Mean blood pressure", "value", "2024-01-01 08:40:00", "20", "mmHg"],
            ["E1", "P1", "MAP", "Mean blood pressure", "value", "2024-01-01 08:50:00", "200", "mmHg"],
            ["E1", "P1", "MAP", "Mean blood pressure", "value", "2024-01-01 09:00:00", "19", "mmHg"],
            ["E1", "P1", "MAP", "Mean blood pressure", "value", "2024-01-01 09:10:00", "201", "mmHg"],
            ["E1", "P1", "MAP", "Mean blood pressure", "value", "2024-01-01 10:05:00", "65", "mmHg"],
        ],
        columns=columns,
    )
    _write_tar(
        flows_path,
        {
            "Epic_flowsheets_cleaned/part_01.csv": part1,
            "Epic_flowsheets_cleaned/part_02.csv": part2,
        },
    )


class TestSourceExtractors(unittest.TestCase):
    def test_height_and_mover_ounce_weight_are_explicit(self) -> None:
        self.assertAlmostEqual(parse_height_m("5' 10\""), 1.778, places=3)
        self.assertAlmostEqual(parse_height_m("178 cm"), 1.78, places=6)
        self.assertAlmostEqual(parse_mover_weight_kg("2821.9176"), 80.0, places=4)
        self.assertAlmostEqual(parse_mover_weight_kg("80 kg"), 80.0, places=6)
        self.assertAlmostEqual(
            parse_mover_weight_kg("80000", unlabelled_unit="g"), 80.0, places=6
        )
        self.assertTrue(np.isnan(parse_mover_weight_kg("80")))

    def test_real_mover_scale_2687_85_ounces_and_5ft7_has_plausible_bmi(self) -> None:
        weight_kg = parse_mover_weight_kg("2687.85")
        height_m = parse_height_m("5' 7\"")
        bmi = weight_kg / (height_m**2)

        self.assertAlmostEqual(weight_kg, 76.2, places=1)
        self.assertAlmostEqual(height_m, 1.7018, places=4)
        self.assertGreater(bmi, 20.0)
        self.assertLess(bmi, 35.0)

    def test_inspire_zip_chunking_eligibility_range_no_locf_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "inspire.zip"
            cache = root / "cache"
            _write_inspire_zip(source)
            first = extract_inspire_source(
                source,
                chunksize_operations=2,
                chunksize_vitals=2,
                cache_dir=cache,
                collect_map=True,
            )
            self.assertEqual(first.cohort["encounter_id"].tolist(), ["401"])
            self.assertEqual(first.map_observations["map_value"].tolist(), [20, 200])
            self.assertEqual(first.map_observations["map_source"].tolist(), ["ART", "NIBP"])
            self.assertEqual(
                first.map_observations["minute_from_anesthesia_start"].tolist(), [5, 10]
            )
            self.assertTrue(first.map_observations["observed_flag"].all())
            self.assertEqual(len(first.map_observations), 2)
            self.assertTrue(first.cache_manifest_path.exists())
            payload = json.loads(first.cache_manifest_path.read_text(encoding="utf-8"))
            self.assertTrue(payload["settings"]["no_locf"])
            self.assertEqual(payload["settings"]["map_range_inclusive"], [20.0, 200.0])

            second = extract_inspire_source(
                source,
                chunksize_operations=2,
                chunksize_vitals=2,
                cache_dir=cache,
                collect_map=True,
            )
            assert_frame_equal(
                first.map_observations.reset_index(drop=True),
                second.map_observations.reset_index(drop=True),
                check_dtype=False,
            )
            self.assertTrue(second.manifest["cache_hit"].all())

    def test_mover_tar_stream_cohort_map_sources_range_no_locf_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            epic = root / "EPIC_EMR.tar.gz"
            flows = root / "Epic_flowsheets_cleaned.tar.gz"
            cache = root / "cache"
            _write_mover_archives(epic, flows)
            first = extract_mover_source(
                epic,
                flows,
                chunksize_patients=2,
                chunksize_flowsheet=2,
                expected_flowsheet_parts=2,
                cache_dir=cache,
                collect_map=True,
            )
            self.assertEqual(first.cohort["encounter_id"].tolist(), ["E1"])
            self.assertAlmostEqual(float(first.cohort.loc[0, "weight_kg"]), 80.0, places=3)
            self.assertAlmostEqual(float(first.cohort.loc[0, "height_m"]), 1.778, places=3)
            self.assertAlmostEqual(float(first.cohort.loc[0, "bmi"]), 25.3, places=1)
            self.assertEqual(first.map_observations["map_value"].tolist(), [60, 70, 80, 20, 200])
            self.assertEqual(
                first.map_observations["map_source"].tolist(),
                ["ART", "NIBP", "DIRECT", "DIRECT", "DIRECT"],
            )
            self.assertEqual(
                first.map_observations["minute_from_anesthesia_start"].tolist(),
                [5, 10, 20, 40, 50],
            )
            self.assertEqual(len(first.map_observations), 5)
            self.assertTrue(first.map_observations["observed_flag"].all())

            second = extract_mover_source(
                epic,
                flows,
                chunksize_patients=2,
                chunksize_flowsheet=2,
                expected_flowsheet_parts=2,
                cache_dir=cache,
                collect_map=True,
            )
            assert_frame_equal(
                first.map_observations.reset_index(drop=True),
                second.map_observations.reset_index(drop=True),
                check_dtype=False,
            )
            self.assertTrue(second.manifest["cache_hit"].all())

    def test_cache_contract_and_expected_part_count_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            epic = root / "EPIC_EMR.tar.gz"
            flows = root / "Epic_flowsheets_cleaned.tar.gz"
            cache = root / "cache"
            _write_mover_archives(epic, flows)
            result = extract_mover_source(
                epic,
                flows,
                chunksize_patients=2,
                chunksize_flowsheet=2,
                expected_flowsheet_parts=2,
                cache_dir=cache,
                collect_map=True,
            )
            with self.assertRaises(CacheMismatchError):
                list(
                    iter_mover_map_observations(
                        flows,
                        result.cohort,
                        chunksize=3,
                        expected_parts=2,
                        cache_dir=cache,
                    )
                )
            with self.assertRaises(SourceSchemaError):
                list(
                    iter_mover_map_observations(
                        flows,
                        result.cohort,
                        chunksize=2,
                        expected_parts=19,
                        cache_dir=None,
                    )
                )


if __name__ == "__main__":
    unittest.main()

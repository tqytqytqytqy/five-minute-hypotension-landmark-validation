"""Read-only source extraction for the INSPIRE -> MOVER LM5 study.

This module deliberately stops at two auditable source-domain tables:

* a harmonised, eligibility-filtered case table; and
* a long table containing only actually observed MAP measurements.

It never bins MAP, carries values forward, interpolates, constructs outcomes,
loads a model, or writes model predictions/performance.  H5/R1 construction is
therefore downstream of this module and cannot silently contaminate extraction.

The public iterators are the preferred API for full MOVER scans.  Optional
chunk caches are resumable and carry a JSON manifest tied to source file
fingerprints, extraction settings, and the eligible-cohort signature.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
import tarfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd


MAP_MIN = 20.0
MAP_MAX = 200.0
DEFAULT_MAX_ANESTHESIA_MIN = 24.0 * 60.0
INSPIRE_OPERATIONS_MEMBER = "operations.csv.gz"
INSPIRE_VITALS_MEMBER = "vitals.csv.gz"
MOVER_PATIENT_MEMBER = "EPIC_EMR/EMR/patient_information.csv"

COMMON_COHORT_COLUMNS = [
    "dataset",
    "patient_id",
    "encounter_id",
    "age",
    "sex",
    "male",
    "height_m",
    "weight_kg",
    "bmi",
    "asa",
    "emergency",
    "surgery_type",
    "surgical_service",
    "primary_anesthesia",
    "anesthesia_start",
    "anesthesia_end",
    "anesthesia_duration_min",
    "time_scale",
]

COMMON_MAP_COLUMNS = [
    "dataset",
    "patient_id",
    "encounter_id",
    "observed_time",
    "minute_from_anesthesia_start",
    "map_value",
    "map_source",
    "map_measurement",
    "observed_flag",
    "raw_name",
    "raw_display_name",
    "raw_unit",
]

_CARDIAC_RE = re.compile(
    r"\b(?:cardiac|cardiothoracic|open[ -]?heart|cabg|coronary(?: artery)? bypass|"
    r"cardiopulmonary(?: bypass)?|heart transplant|ventricular assist|vad|ecmo|"
    r"valve (?:repair|replacement)|aortic root)\b",
    re.I,
)
_OBSTETRIC_RE = re.compile(
    r"\b(?:obstetric|cesarean|caesarean|c[ -]?section|vaginal delivery|labor and delivery|"
    r"fetal surgery|placenta|postpartum)\b",
    re.I,
)
_GENERAL_RE = re.compile(r"\bgeneral\b", re.I)
_MAP_LABEL_RE = re.compile(
    r"\bmap\b|mean arterial(?: pressure)?|mean blood pressure|\babp\s*(?:mean|m)\b|"
    r"arterial (?:blood )?pressure mean",
    re.I,
)
_SUMMARY_LABEL_RE = re.compile(
    r"\b(?:min(?:imum)?|max(?:imum)?|goal|target|alarm|limit|parameter)\b", re.I
)
_ART_RE = re.compile(
    r"\b(?:art(?:erial)?[ _-]?(?:line|bp|map)|a[ _-]?line|abp|"
    r"arterial(?: line)? (?:map|blood pressure)|invasive (?:map|blood pressure))\b",
    re.I,
)
_NIBP_RE = re.compile(
    r"\b(?:nibp|nbp|non[ -]?invasive(?: blood pressure)?|cuff(?: blood pressure)?)\b", re.I
)
_SINGLE_NUMBER_RE = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*(?:mm\s*hg|mmhg)?\s*$", re.I
)


class SourceSchemaError(ValueError):
    """Raised when an archive or CSV does not satisfy the locked source contract."""


class CacheMismatchError(RuntimeError):
    """Raised rather than mixing shards produced from different source contracts."""


class _NonSeekableReader(io.RawIOBase):
    """Give pandas the RawIO interface missing from tarfile streaming members."""

    def __init__(self, source) -> None:
        self.source = source

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def readinto(self, buffer) -> int:
        chunk = self.source.read(len(buffer))
        size = len(chunk)
        buffer[:size] = chunk
        return size


def _buffer_tar_member(handle) -> io.BufferedReader:
    return io.BufferedReader(_NonSeekableReader(handle), buffer_size=8 * 1024 * 1024)


@dataclass(frozen=True)
class SourceExtraction:
    """Small-study convenience result; use iterators for memory-safe full scans."""

    cohort: pd.DataFrame
    map_observations: pd.DataFrame | None
    manifest: pd.DataFrame
    cache_manifest_path: Path | None


def _require_columns(frame: pd.DataFrame, required: Sequence[str], source: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise SourceSchemaError(f"{source} is missing required columns: {missing}")


def _normalise_sex(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip().str.upper()
    out = pd.Series(pd.NA, index=values.index, dtype="string")
    out.loc[text.isin(["M", "MALE"])] = "M"
    out.loc[text.isin(["F", "FEMALE"])] = "F"
    return out


def _male_numeric(sex: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=sex.index, dtype=float)
    out.loc[sex.eq("M")] = 1.0
    out.loc[sex.eq("F")] = 0.0
    return out


def _truthy(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip().str.lower()
    return text.isin(["1", "true", "t", "yes", "y", "emergency", "urgent"])


def parse_height_m(value: object) -> float:
    """Parse centimetres/metres/inches or a feet-and-inches height string."""

    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower().replace("′", "'").replace("″", '"')
    feet = re.match(r"^(\d+(?:\.\d+)?)\s*(?:'|ft|feet)\s*(\d+(?:\.\d+)?)?", text)
    if feet:
        return (float(feet.group(1)) * 12.0 + float(feet.group(2) or 0.0)) * 0.0254
    number_match = re.search(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", text.replace(",", ""))
    if not number_match:
        return np.nan
    number = float(number_match.group(0))
    if "cm" in text or (120.0 <= number <= 230.0 and not re.search(r"\bin\b", text)):
        return number / 100.0
    if re.search(r"\b(?:in|inch|inches)\b", text) or 48.0 <= number <= 90.0:
        return number * 0.0254
    if " m" in f" {text}" or 1.2 <= number <= 2.3:
        return number
    return np.nan


def parse_mover_weight_kg(value: object, *, unlabelled_unit: str = "oz") -> float:
    """Parse MOVER WEIGHT, whose unlabelled locked unit is ounces, into kg.

    Explicit ``kg``, gram, pound, and ounce labels are honoured.  Unlabelled
    values are interpreted as ounces by default, matching the audited MOVER
    source convention and its official processing (``WEIGHT * 0.0283495``).
    The unit remains an explicit argument so alternate documented extracts
    cannot silently reuse the wrong conversion.
    """

    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower().replace(",", "")
    match = re.search(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", text)
    if not match:
        return np.nan
    number = float(match.group(0))
    if number <= 0:
        return np.nan
    if re.search(r"\bkg\b|kilogram", text):
        kg = number
    elif re.search(r"\b(?:lb|lbs)\b|pound", text):
        kg = number * 0.45359237
    elif re.search(r"\b(?:oz)\b|ounce", text):
        kg = number * 0.028349523125
    elif re.search(r"\b(?:g|gm|grams?)\b", text):
        kg = number / 1000.0
    else:
        unit = unlabelled_unit.lower()
        if unit in {"oz", "ounce", "ounces"}:
            kg = number * 0.028349523125
        elif unit in {"g", "gram", "grams"}:
            kg = number / 1000.0
        elif unit in {"kg", "kilogram", "kilograms"}:
            kg = number
        else:
            raise ValueError("unlabelled_unit must be 'oz', 'g', or 'kg'")
    return kg if 20.0 <= kg <= 400.0 else np.nan


def _inspire_weight_kg(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if 20.0 <= number <= 400.0 else np.nan


def _valid_bmi(weight_kg: pd.Series, height_m: pd.Series) -> pd.Series:
    bmi = pd.to_numeric(weight_kg, errors="coerce") / np.square(
        pd.to_numeric(height_m, errors="coerce")
    )
    return bmi.where(bmi.between(10.0, 80.0, inclusive="both"))


def _finalise_common_cohort(
    frame: pd.DataFrame, *, first_case_per_patient: bool
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=COMMON_COHORT_COLUMNS)
    out = frame.copy()
    identity = ["patient_id", "anesthesia_start", "anesthesia_end"]
    conflicts = (
        out.groupby("encounter_id", dropna=False)[identity]
        .nunique(dropna=True)
        .gt(1)
        .any(axis=1)
    )
    conflict_ids = set(conflicts.index[conflicts])
    out = out.loc[~out["encounter_id"].isin(conflict_ids)].copy()
    out["_completeness"] = out[COMMON_COHORT_COLUMNS].notna().sum(axis=1)
    out = (
        out.sort_values(["encounter_id", "_completeness"], ascending=[True, False], kind="mergesort")
        .drop_duplicates("encounter_id", keep="first")
        .drop(columns="_completeness")
    )
    if first_case_per_patient:
        out = (
            out.sort_values(["patient_id", "anesthesia_start", "encounter_id"], kind="mergesort")
            .drop_duplicates("patient_id", keep="first")
        )
    return out[COMMON_COHORT_COLUMNS].reset_index(drop=True)


def build_inspire_common_cohort(
    operations: pd.DataFrame,
    *,
    max_anesthesia_duration_min: float = DEFAULT_MAX_ANESTHESIA_MIN,
    first_case_per_patient: bool = True,
) -> pd.DataFrame:
    """Map and filter INSPIRE operations to the locked common target population."""

    required = [
        "op_id",
        "subject_id",
        "age",
        "sex",
        "weight",
        "height",
        "asa",
        "emop",
        "department",
        "antype",
        "icd10_pcs",
        "anstart_time",
        "anend_time",
        "cpbon_time",
    ]
    _require_columns(operations, required, "INSPIRE operations.csv.gz")
    raw = operations.copy()
    age = pd.to_numeric(raw["age"], errors="coerce")
    start = pd.to_numeric(raw["anstart_time"], errors="coerce")
    end = pd.to_numeric(raw["anend_time"], errors="coerce")
    duration = end - start
    department = raw["department"].astype("string").str.strip().str.upper()
    anesthesia = raw["antype"].astype("string").str.strip()
    cpb = pd.to_numeric(raw["cpbon_time"], errors="coerce").notna()
    known_service = department.notna() & department.ne("")
    eligible = (
        age.ge(18.0)
        & anesthesia.str.contains(_GENERAL_RE, na=False)
        & known_service
        & ~department.isin(["CTS", "CS", "OG"])
        & ~cpb
        & start.notna()
        & end.notna()
        & duration.gt(0.0)
        & duration.le(float(max_anesthesia_duration_min))
    )
    raw = raw.loc[eligible].copy()
    if raw.empty:
        return pd.DataFrame(columns=COMMON_COHORT_COLUMNS)
    start = pd.to_numeric(raw["anstart_time"], errors="coerce")
    end = pd.to_numeric(raw["anend_time"], errors="coerce")
    height = raw["height"].map(parse_height_m)
    weight = raw["weight"].map(_inspire_weight_kg)
    sex = _normalise_sex(raw["sex"])
    out = pd.DataFrame(
        {
            "dataset": "INSPIRE_1.4.2",
            "patient_id": raw["subject_id"].astype("string"),
            "encounter_id": raw["op_id"].astype("string"),
            "age": pd.to_numeric(raw["age"], errors="coerce"),
            "sex": sex,
            "male": _male_numeric(sex),
            "height_m": height,
            "weight_kg": weight,
            "bmi": _valid_bmi(weight, height),
            "asa": pd.to_numeric(raw["asa"], errors="coerce"),
            "emergency": _truthy(raw["emop"]).astype(bool),
            "surgery_type": raw["icd10_pcs"].astype("string"),
            "surgical_service": department.loc[raw.index],
            "primary_anesthesia": raw["antype"].astype("string"),
            "anesthesia_start": start,
            "anesthesia_end": end,
            "anesthesia_duration_min": end - start,
            "time_scale": "relative_minutes",
        },
        index=raw.index,
    )
    return _finalise_common_cohort(out, first_case_per_patient=first_case_per_patient)


def build_mover_common_cohort(
    patient_information: pd.DataFrame,
    *,
    max_anesthesia_duration_min: float = DEFAULT_MAX_ANESTHESIA_MIN,
    first_case_per_patient: bool = True,
    unlabelled_weight_unit: str = "oz",
) -> pd.DataFrame:
    """Map/filter EPIC patient_information to the same target population."""

    required = [
        "LOG_ID",
        "MRN",
        "BIRTH_DATE",
        "HEIGHT",
        "WEIGHT",
        "SEX",
        "ASA_RATING_C",
        "PRIMARY_ANES_TYPE_NM",
        "PRIMARY_PROCEDURE_NM",
        "AN_START_DATETIME",
        "AN_STOP_DATETIME",
    ]
    _require_columns(patient_information, required, "MOVER patient_information.csv")
    raw = patient_information.copy()
    age = pd.to_numeric(raw["BIRTH_DATE"], errors="coerce")
    start = pd.to_datetime(raw["AN_START_DATETIME"], errors="coerce")
    end = pd.to_datetime(raw["AN_STOP_DATETIME"], errors="coerce")
    duration = (end - start).dt.total_seconds() / 60.0
    anesthesia = raw["PRIMARY_ANES_TYPE_NM"].astype("string").str.strip()
    procedure = raw["PRIMARY_PROCEDURE_NM"].astype("string").str.strip()
    known_procedure = procedure.notna() & procedure.ne("")
    excluded = procedure.str.contains(_CARDIAC_RE, na=False) | procedure.str.contains(
        _OBSTETRIC_RE, na=False
    )
    eligible = (
        age.ge(18.0)
        & anesthesia.str.contains(_GENERAL_RE, na=False)
        & known_procedure
        & ~excluded
        & start.notna()
        & end.notna()
        & duration.gt(0.0)
        & duration.le(float(max_anesthesia_duration_min))
    )
    raw = raw.loc[eligible].copy()
    if raw.empty:
        return pd.DataFrame(columns=COMMON_COHORT_COLUMNS)
    start = pd.to_datetime(raw["AN_START_DATETIME"], errors="coerce")
    end = pd.to_datetime(raw["AN_STOP_DATETIME"], errors="coerce")
    height = raw["HEIGHT"].map(parse_height_m)
    weight = raw["WEIGHT"].map(
        lambda value: parse_mover_weight_kg(value, unlabelled_unit=unlabelled_weight_unit)
    )
    sex = _normalise_sex(raw["SEX"])
    if "PATIENT_CLASS_NM" in raw:
        emergency = raw["PATIENT_CLASS_NM"].astype("string").str.contains(
            r"emergency|urgent|\bed\b", case=False, regex=True, na=False
        )
    else:
        emergency = pd.Series(False, index=raw.index)
    service = (
        raw["PATIENT_CLASS_GROUP"].astype("string")
        if "PATIENT_CLASS_GROUP" in raw
        else pd.Series(pd.NA, index=raw.index, dtype="string")
    )
    out = pd.DataFrame(
        {
            "dataset": "MOVER_EPIC",
            "patient_id": raw["MRN"].astype("string"),
            "encounter_id": raw["LOG_ID"].astype("string"),
            "age": pd.to_numeric(raw["BIRTH_DATE"], errors="coerce"),
            "sex": sex,
            "male": _male_numeric(sex),
            "height_m": height,
            "weight_kg": weight,
            "bmi": _valid_bmi(weight, height),
            "asa": pd.to_numeric(raw["ASA_RATING_C"], errors="coerce"),
            "emergency": emergency.astype(bool),
            "surgery_type": raw["PRIMARY_PROCEDURE_NM"].astype("string"),
            "surgical_service": service,
            "primary_anesthesia": raw["PRIMARY_ANES_TYPE_NM"].astype("string"),
            "anesthesia_start": start,
            "anesthesia_end": end,
            "anesthesia_duration_min": (end - start).dt.total_seconds() / 60.0,
            "time_scale": "datetime",
        },
        index=raw.index,
    )
    return _finalise_common_cohort(out, first_case_per_patient=first_case_per_patient)


def _find_zip_member(zf: zipfile.ZipFile, suffix: str) -> str:
    # Match the archive member basename exactly.  A suffix match incorrectly
    # treats ward_vitals.csv.gz as a second copy of vitals.csv.gz in INSPIRE.
    matches = [
        name
        for name in zf.namelist()
        if Path(name).name == Path(suffix).name
    ]
    if len(matches) != 1:
        raise SourceSchemaError(
            f"Expected exactly one ZIP member ending in {suffix!r}; found {len(matches)}"
        )
    return matches[0]


def _iter_zip_csv_chunks(
    zip_path: str | Path,
    member_suffix: str,
    *,
    chunksize: int,
    usecols: Sequence[str] | None = None,
) -> Iterator[pd.DataFrame]:
    with zipfile.ZipFile(Path(zip_path), mode="r") as zf:
        member_name = _find_zip_member(zf, member_suffix)
        with zf.open(member_name, mode="r") as raw:
            if member_name.endswith(".gz"):
                handle = gzip.GzipFile(fileobj=raw, mode="rb")
            else:
                handle = raw
            try:
                yield from pd.read_csv(
                    handle,
                    usecols=usecols,
                    chunksize=int(chunksize),
                    low_memory=False,
                )
            finally:
                if handle is not raw:
                    handle.close()


def iter_inspire_operation_chunks(
    zip_path: str | Path, *, chunksize: int = 250_000
) -> Iterator[pd.DataFrame]:
    """Stream raw INSPIRE operations chunks from the 1.4.2 ZIP."""

    yield from _iter_zip_csv_chunks(
        zip_path, INSPIRE_OPERATIONS_MEMBER, chunksize=chunksize
    )


def extract_inspire_common_cohort(
    zip_path: str | Path,
    *,
    chunksize: int = 250_000,
    max_anesthesia_duration_min: float = DEFAULT_MAX_ANESTHESIA_MIN,
    first_case_per_patient: bool = True,
) -> pd.DataFrame:
    """Chunk-read operations and return one harmonised eligible cohort table."""

    chunks = list(iter_inspire_operation_chunks(zip_path, chunksize=chunksize))
    raw = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    return build_inspire_common_cohort(
        raw,
        max_anesthesia_duration_min=max_anesthesia_duration_min,
        first_case_per_patient=first_case_per_patient,
    )


def _tar_member_matches(name: str, suffix: str) -> bool:
    return name.lstrip("./").endswith(suffix.lstrip("./"))


def iter_mover_patient_chunks(
    epic_tar_path: str | Path, *, chunksize: int = 100_000
) -> Iterator[pd.DataFrame]:
    """Stream EPIC patient_information.csv without extracting the tar archive."""

    found = 0
    with tarfile.open(Path(epic_tar_path), mode="r|*") as archive:
        for member in archive:
            if not member.isfile() or not _tar_member_matches(member.name, MOVER_PATIENT_MEMBER):
                continue
            found += 1
            handle = archive.extractfile(member)
            if handle is None:
                raise SourceSchemaError(f"Cannot read tar member {member.name}")
            yield from pd.read_csv(
                _buffer_tar_member(handle), chunksize=int(chunksize), low_memory=False
            )
    if found != 1:
        raise SourceSchemaError(
            f"Expected exactly one {MOVER_PATIENT_MEMBER}; found {found} in {epic_tar_path}"
        )


def extract_mover_common_cohort(
    epic_tar_path: str | Path,
    *,
    chunksize: int = 100_000,
    max_anesthesia_duration_min: float = DEFAULT_MAX_ANESTHESIA_MIN,
    first_case_per_patient: bool = True,
    unlabelled_weight_unit: str = "oz",
) -> pd.DataFrame:
    """Chunk-read EPIC patient information and return the common cohort."""

    chunks = list(iter_mover_patient_chunks(epic_tar_path, chunksize=chunksize))
    raw = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    return build_mover_common_cohort(
        raw,
        max_anesthesia_duration_min=max_anesthesia_duration_min,
        first_case_per_patient=first_case_per_patient,
        unlabelled_weight_unit=unlabelled_weight_unit,
    )


def _single_numeric(values: pd.Series) -> pd.Series:
    extracted = values.astype("string").str.extract(_SINGLE_NUMBER_RE, expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def normalise_inspire_map_chunk(
    raw_vitals: pd.DataFrame, cohort: pd.DataFrame
) -> pd.DataFrame:
    """Return observed INSPIRE MAP rows only; no fusion, binning, or filling."""

    _require_columns(
        raw_vitals,
        ["op_id", "subject_id", "chart_time", "item_name", "value"],
        "INSPIRE vitals.csv.gz",
    )
    _require_columns(
        cohort,
        ["encounter_id", "patient_id", "anesthesia_start", "anesthesia_end"],
        "common cohort",
    )
    raw = raw_vitals.copy()
    raw["encounter_id"] = raw["op_id"].astype("string")
    raw["raw_name"] = raw["item_name"].astype("string").str.strip().str.lower()
    raw = raw.loc[raw["raw_name"].isin(["art_mbp", "nibp_mbp"])].copy()
    raw["map_value"] = pd.to_numeric(raw["value"], errors="coerce")
    raw["observed_time"] = pd.to_numeric(raw["chart_time"], errors="coerce")
    raw = raw.loc[
        raw["map_value"].between(MAP_MIN, MAP_MAX, inclusive="both")
        & raw["observed_time"].notna()
    ].copy()
    index = cohort[
        ["encounter_id", "patient_id", "anesthesia_start", "anesthesia_end"]
    ].drop_duplicates("encounter_id")
    out = raw.merge(index, on="encounter_id", how="inner", validate="many_to_one")
    out["anesthesia_start"] = pd.to_numeric(out["anesthesia_start"], errors="coerce")
    out["anesthesia_end"] = pd.to_numeric(out["anesthesia_end"], errors="coerce")
    out = out.loc[
        out["observed_time"].ge(out["anesthesia_start"])
        & out["observed_time"].le(out["anesthesia_end"])
    ].copy()
    out["dataset"] = "INSPIRE_1.4.2"
    out["minute_from_anesthesia_start"] = out["observed_time"] - out["anesthesia_start"]
    out["map_source"] = out["raw_name"].map(
        {"art_mbp": "ART", "nibp_mbp": "NIBP"}
    )
    out["map_measurement"] = "direct"
    out["observed_flag"] = True
    out["raw_display_name"] = out["raw_name"]
    out["raw_unit"] = "mmHg"
    result = out[COMMON_MAP_COLUMNS].reset_index(drop=True)
    result.attrs.update(
        {
            "raw_rows": len(raw_vitals),
            "retained_rows": len(result),
            "locf": False,
            "interpolation": False,
        }
    )
    return result


def normalise_mover_map_chunk(
    raw_flowsheet: pd.DataFrame, cohort: pd.DataFrame
) -> pd.DataFrame:
    """Classify direct observed MAP rows and attach the anaesthesia window."""

    required = ["LOG_ID", "FLO_NAME", "FLO_DISPLAY_NAME", "RECORDED_TIME", "MEAS_VALUE"]
    _require_columns(raw_flowsheet, required, "MOVER cleaned flowsheet")
    _require_columns(
        cohort,
        ["encounter_id", "patient_id", "anesthesia_start", "anesthesia_end"],
        "common cohort",
    )
    raw = raw_flowsheet.copy()
    raw["encounter_id"] = raw["LOG_ID"].astype("string")
    raw_name = raw["FLO_NAME"].astype("string").fillna("")
    raw_display = raw["FLO_DISPLAY_NAME"].astype("string").fillna("")
    label = (raw_name + " | " + raw_display).str.strip()
    candidate = label.str.contains(_MAP_LABEL_RE, na=False) & ~label.str.contains(
        _SUMMARY_LABEL_RE, na=False
    )
    raw = raw.loc[candidate].copy()
    label = label.loc[raw.index]
    raw["map_value"] = _single_numeric(raw["MEAS_VALUE"])
    raw["observed_time"] = pd.to_datetime(raw["RECORDED_TIME"], errors="coerce")
    raw = raw.loc[
        raw["map_value"].between(MAP_MIN, MAP_MAX, inclusive="both")
        & raw["observed_time"].notna()
    ].copy()
    label = label.loc[raw.index]
    source = pd.Series("DIRECT", index=raw.index, dtype="string")
    source.loc[label.str.contains(_NIBP_RE, na=False)] = "NIBP"
    source.loc[label.str.contains(_ART_RE, na=False)] = "ART"
    raw["map_source"] = source
    index = cohort[
        ["encounter_id", "patient_id", "anesthesia_start", "anesthesia_end"]
    ].drop_duplicates("encounter_id")
    index = index.copy()
    index["anesthesia_start"] = pd.to_datetime(index["anesthesia_start"], errors="coerce")
    index["anesthesia_end"] = pd.to_datetime(index["anesthesia_end"], errors="coerce")
    out = raw.merge(index, on="encounter_id", how="inner", validate="many_to_one")
    out = out.loc[
        out["observed_time"].ge(out["anesthesia_start"])
        & out["observed_time"].le(out["anesthesia_end"])
    ].copy()
    out["dataset"] = "MOVER_EPIC"
    out["minute_from_anesthesia_start"] = (
        out["observed_time"] - out["anesthesia_start"]
    ).dt.total_seconds() / 60.0
    out["map_measurement"] = "direct"
    out["observed_flag"] = True
    out["raw_name"] = out["FLO_NAME"].astype("string")
    out["raw_display_name"] = out["FLO_DISPLAY_NAME"].astype("string")
    out["raw_unit"] = (
        out["UNITS"].astype("string") if "UNITS" in out else pd.Series(pd.NA, index=out.index)
    )
    result = out[COMMON_MAP_COLUMNS].reset_index(drop=True)
    result.attrs.update(
        {
            "raw_rows": len(raw_flowsheet),
            "retained_rows": len(result),
            "locf": False,
            "interpolation": False,
        }
    )
    return result


def _sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _source_fingerprint(path: str | Path, *, hash_source: bool) -> dict:
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _sha256_file(resolved) if hash_source else None,
    }


def _cohort_signature(cohort: pd.DataFrame) -> dict:
    required = ["encounter_id", "patient_id", "anesthesia_start", "anesthesia_end"]
    _require_columns(cohort, required, "common cohort")
    stable = cohort[required].copy().astype("string").sort_values(required, kind="mergesort")
    payload = stable.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return {"rows": int(len(stable)), "sha256": hashlib.sha256(payload).hexdigest()}


def _safe_unit_name(unit: str) -> str:
    return hashlib.sha256(unit.encode("utf-8")).hexdigest()[:24] + ".csv.gz"


def _restore_map_types(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    for column in ["patient_id", "encounter_id"]:
        out[column] = out[column].astype("string")
    for column in ["minute_from_anesthesia_start", "map_value"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    if out["dataset"].astype(str).eq("MOVER_EPIC").all():
        out["observed_time"] = pd.to_datetime(out["observed_time"], errors="coerce")
    else:
        out["observed_time"] = pd.to_numeric(out["observed_time"], errors="coerce")
    out["observed_flag"] = out["observed_flag"].astype("string").str.lower().eq("true")
    return out


class _ChunkCache:
    """Private cache whose contract prevents cross-source shard reuse."""

    def __init__(
        self,
        root: str | Path,
        *,
        source_name: str,
        sources: list[dict],
        settings: dict,
        resume: bool,
    ) -> None:
        self.root = Path(root) / source_name
        self.root.mkdir(parents=True, exist_ok=True)
        self.shards = self.root / "shards"
        self.shards.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        contract = {
            "format_version": 1,
            "source_name": source_name,
            "sources": sources,
            "settings": settings,
        }
        if self.manifest_path.exists():
            if not resume:
                raise CacheMismatchError(
                    f"Cache already exists at {self.root}; choose a new cache directory"
                )
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            observed = {key: payload.get(key) for key in contract}
            if observed != contract:
                raise CacheMismatchError(
                    f"Cache contract mismatch at {self.root}; do not mix source versions/settings"
                )
            self.payload = payload
        else:
            self.payload = {
                **contract,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "updated_utc": None,
                "completed_chunks": {},
                "completed_members": {},
            }
            self._save()

    def _save(self) -> None:
        self.payload["updated_utc"] = datetime.now(timezone.utc).isoformat()
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self.payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(self.manifest_path)

    def get(self, unit: str) -> pd.DataFrame | None:
        metadata = self.payload["completed_chunks"].get(unit)
        if metadata is None:
            return None
        path = self.root / metadata["relative_path"]
        if not path.exists() or _sha256_file(path) != metadata["sha256"]:
            raise CacheMismatchError(f"Cached shard missing or corrupt: {path}")
        frame = _restore_map_types(pd.read_csv(path, low_memory=False))
        frame.attrs.update(metadata)
        frame.attrs["cache_hit"] = True
        return frame

    def put(
        self,
        unit: str,
        frame: pd.DataFrame,
        *,
        member_key: str,
        raw_rows: int,
    ) -> None:
        filename = _safe_unit_name(unit)
        path = self.shards / filename
        temporary = self.shards / (filename + ".tmp")
        frame.to_csv(temporary, index=False, compression="gzip")
        temporary.replace(path)
        self.payload["completed_chunks"][unit] = {
            "relative_path": str(path.relative_to(self.root)),
            "sha256": _sha256_file(path),
            "member_key": member_key,
            "raw_rows": int(raw_rows),
            "retained_rows": int(len(frame)),
            "locf": False,
            "interpolation": False,
        }
        self._save()

    def member_complete(self, member_key: str) -> bool:
        return member_key in self.payload["completed_members"]

    def member_units(self, member_key: str) -> list[str]:
        return list(self.payload["completed_members"].get(member_key, {}).get("units", []))

    def mark_member_complete(self, member_key: str, units: list[str]) -> None:
        self.payload["completed_members"][member_key] = {"units": list(units)}
        self._save()


def _attach_audit(
    frame: pd.DataFrame,
    *,
    unit: str,
    member_key: str,
    raw_rows: int,
    cache_hit: bool,
) -> pd.DataFrame:
    frame.attrs.update(
        {
            "unit": unit,
            "member_key": member_key,
            "raw_rows": int(raw_rows),
            "retained_rows": int(len(frame)),
            "cache_hit": bool(cache_hit),
            "locf": False,
            "interpolation": False,
        }
    )
    return frame


def iter_inspire_map_observations(
    zip_path: str | Path,
    cohort: pd.DataFrame,
    *,
    chunksize: int = 500_000,
    cache_dir: str | Path | None = None,
    resume: bool = True,
    hash_source: bool = False,
) -> Iterator[pd.DataFrame]:
    """Stream eligible observed INSPIRE MAP rows, optionally caching each chunk."""

    cache = None
    member_key = f"zip::{INSPIRE_VITALS_MEMBER}"
    if cache_dir is not None:
        cache = _ChunkCache(
            cache_dir,
            source_name="inspire_vitals_observed_map",
            sources=[_source_fingerprint(zip_path, hash_source=hash_source)],
            settings={
                "chunksize": int(chunksize),
                "cohort": _cohort_signature(cohort),
                "map_range_inclusive": [MAP_MIN, MAP_MAX],
                "no_locf": True,
                "classifier": "INSPIRE_item_name_art_mbp_or_nibp_mbp_v1",
            },
            resume=resume,
        )
        if cache.member_complete(member_key):
            for unit in cache.member_units(member_key):
                frame = cache.get(unit)
                if frame is None:
                    raise CacheMismatchError(f"Manifest lists a missing chunk: {unit}")
                metadata = cache.payload["completed_chunks"][unit]
                yield _attach_audit(
                    frame,
                    unit=unit,
                    member_key=member_key,
                    raw_rows=metadata["raw_rows"],
                    cache_hit=True,
                )
            return
    units: list[str] = []
    raw_iterator = _iter_zip_csv_chunks(
        zip_path,
        INSPIRE_VITALS_MEMBER,
        chunksize=int(chunksize),
        usecols=["op_id", "subject_id", "chart_time", "item_name", "value"],
    )
    for chunk_number, raw in enumerate(raw_iterator):
        unit = f"{member_key}::chunk_{chunk_number:08d}"
        units.append(unit)
        cached = cache.get(unit) if cache is not None else None
        if cached is not None:
            metadata = cache.payload["completed_chunks"][unit]
            yield _attach_audit(
                cached,
                unit=unit,
                member_key=member_key,
                raw_rows=metadata["raw_rows"],
                cache_hit=True,
            )
            continue
        frame = normalise_inspire_map_chunk(raw, cohort)
        if cache is not None:
            cache.put(unit, frame, member_key=member_key, raw_rows=len(raw))
        yield _attach_audit(
            frame,
            unit=unit,
            member_key=member_key,
            raw_rows=len(raw),
            cache_hit=False,
        )
    if cache is not None:
        cache.mark_member_complete(member_key, units)


def _normalise_tar_paths(paths: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        result = [Path(paths)]
    else:
        result = [Path(path) for path in paths]
    if not result:
        raise ValueError("At least one cleaned flowsheet tar path is required")
    return result


def iter_mover_map_observations(
    cleaned_flowsheet_tar_paths: str | Path | Sequence[str | Path],
    cohort: pd.DataFrame,
    *,
    chunksize: int = 400_000,
    expected_parts: int | None = 19,
    cache_dir: str | Path | None = None,
    resume: bool = True,
    hash_sources: bool = False,
) -> Iterator[pd.DataFrame]:
    """Stream all cleaned MOVER flowsheet CSV parts and retain observed MAP only.

    The real release is expected to contain 19 CSV parts.  Tests or documented
    alternate releases may pass another count (or ``None``), but the observed
    count remains represented by completed member records in the cache manifest.
    """

    paths = _normalise_tar_paths(cleaned_flowsheet_tar_paths)
    cache = None
    if cache_dir is not None:
        cache = _ChunkCache(
            cache_dir,
            source_name="mover_cleaned_flowsheet_observed_map",
            sources=[
                _source_fingerprint(path, hash_source=hash_sources) for path in paths
            ],
            settings={
                "chunksize": int(chunksize),
                "expected_parts": expected_parts,
                "cohort": _cohort_signature(cohort),
                "map_range_inclusive": [MAP_MIN, MAP_MAX],
                "no_locf": True,
                "classifier": "MOVER_direct_MAP_ART_NIBP_DIRECT_v1",
            },
            resume=resume,
        )
    part_count = 0
    wanted = {
        "LOG_ID",
        "MRN",
        "FLO_NAME",
        "FLO_DISPLAY_NAME",
        "RECORD_TYPE",
        "RECORDED_TIME",
        "MEAS_VALUE",
        "UNITS",
    }
    for archive_number, path in enumerate(paths):
        with tarfile.open(path, mode="r|*") as archive:
            for member in archive:
                if not member.isfile() or not member.name.lower().endswith(".csv"):
                    continue
                part_count += 1
                member_key = f"archive_{archive_number:03d}::{member.name}"
                if cache is not None and cache.member_complete(member_key):
                    for unit in cache.member_units(member_key):
                        frame = cache.get(unit)
                        if frame is None:
                            raise CacheMismatchError(f"Manifest lists a missing chunk: {unit}")
                        metadata = cache.payload["completed_chunks"][unit]
                        yield _attach_audit(
                            frame,
                            unit=unit,
                            member_key=member_key,
                            raw_rows=metadata["raw_rows"],
                            cache_hit=True,
                        )
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    raise SourceSchemaError(f"Cannot read tar member {member.name}")
                units: list[str] = []
                chunks = pd.read_csv(
                    _buffer_tar_member(handle),
                    usecols=lambda name: name in wanted,
                    chunksize=int(chunksize),
                    low_memory=False,
                )
                for chunk_number, raw in enumerate(chunks):
                    unit = f"{member_key}::chunk_{chunk_number:08d}"
                    units.append(unit)
                    cached = cache.get(unit) if cache is not None else None
                    if cached is not None:
                        metadata = cache.payload["completed_chunks"][unit]
                        yield _attach_audit(
                            cached,
                            unit=unit,
                            member_key=member_key,
                            raw_rows=metadata["raw_rows"],
                            cache_hit=True,
                        )
                        continue
                    frame = normalise_mover_map_chunk(raw, cohort)
                    if cache is not None:
                        cache.put(unit, frame, member_key=member_key, raw_rows=len(raw))
                    yield _attach_audit(
                        frame,
                        unit=unit,
                        member_key=member_key,
                        raw_rows=len(raw),
                        cache_hit=False,
                    )
                if cache is not None:
                    cache.mark_member_complete(member_key, units)
    if part_count == 0:
        raise SourceSchemaError("No CSV members found in cleaned MOVER flowsheet archive(s)")
    if expected_parts is not None and part_count != int(expected_parts):
        raise SourceSchemaError(
            f"Expected {expected_parts} cleaned flowsheet CSV parts; found {part_count}"
        )


def _collect_map_stream(
    chunks: Iterator[pd.DataFrame], *, collect_map: bool
) -> tuple[pd.DataFrame | None, pd.DataFrame]:
    retained: list[pd.DataFrame] = []
    audit: list[dict] = []
    for chunk in chunks:
        audit.append(
            {
                "unit": chunk.attrs.get("unit"),
                "member_key": chunk.attrs.get("member_key"),
                "raw_rows": int(chunk.attrs.get("raw_rows", 0)),
                "retained_rows": int(len(chunk)),
                "cache_hit": bool(chunk.attrs.get("cache_hit", False)),
                "map_min_inclusive": MAP_MIN,
                "map_max_inclusive": MAP_MAX,
                "locf": False,
                "interpolation": False,
            }
        )
        if collect_map and not chunk.empty:
            retained.append(chunk)
    if collect_map:
        maps = (
            pd.concat(retained, ignore_index=True)
            if retained
            else pd.DataFrame(columns=COMMON_MAP_COLUMNS)
        )
    else:
        maps = None
    return maps, pd.DataFrame(audit)


def extract_inspire_source(
    zip_path: str | Path,
    *,
    chunksize_operations: int = 250_000,
    chunksize_vitals: int = 500_000,
    cache_dir: str | Path | None = None,
    resume: bool = True,
    collect_map: bool = True,
    hash_source: bool = False,
    max_anesthesia_duration_min: float = DEFAULT_MAX_ANESTHESIA_MIN,
    first_case_per_patient: bool = True,
) -> SourceExtraction:
    """Convenience wrapper for small/moderate INSPIRE extraction."""

    cohort = extract_inspire_common_cohort(
        zip_path,
        chunksize=chunksize_operations,
        max_anesthesia_duration_min=max_anesthesia_duration_min,
        first_case_per_patient=first_case_per_patient,
    )
    stream = iter_inspire_map_observations(
        zip_path,
        cohort,
        chunksize=chunksize_vitals,
        cache_dir=cache_dir,
        resume=resume,
        hash_source=hash_source,
    )
    maps, manifest = _collect_map_stream(stream, collect_map=collect_map)
    cache_manifest = (
        Path(cache_dir) / "inspire_vitals_observed_map" / "manifest.json"
        if cache_dir is not None
        else None
    )
    return SourceExtraction(cohort, maps, manifest, cache_manifest)


def extract_mover_source(
    epic_tar_path: str | Path,
    cleaned_flowsheet_tar_paths: str | Path | Sequence[str | Path],
    *,
    chunksize_patients: int = 100_000,
    chunksize_flowsheet: int = 400_000,
    expected_flowsheet_parts: int | None = 19,
    cache_dir: str | Path | None = None,
    resume: bool = True,
    collect_map: bool = True,
    hash_sources: bool = False,
    max_anesthesia_duration_min: float = DEFAULT_MAX_ANESTHESIA_MIN,
    first_case_per_patient: bool = True,
    unlabelled_weight_unit: str = "oz",
) -> SourceExtraction:
    """Convenience wrapper; full scans should set a cache directory."""

    if not collect_map and cache_dir is None:
        raise ValueError("collect_map=False requires cache_dir so retained shards are not discarded")
    cohort = extract_mover_common_cohort(
        epic_tar_path,
        chunksize=chunksize_patients,
        max_anesthesia_duration_min=max_anesthesia_duration_min,
        first_case_per_patient=first_case_per_patient,
        unlabelled_weight_unit=unlabelled_weight_unit,
    )
    stream = iter_mover_map_observations(
        cleaned_flowsheet_tar_paths,
        cohort,
        chunksize=chunksize_flowsheet,
        expected_parts=expected_flowsheet_parts,
        cache_dir=cache_dir,
        resume=resume,
        hash_sources=hash_sources,
    )
    maps, manifest = _collect_map_stream(stream, collect_map=collect_map)
    cache_manifest = (
        Path(cache_dir) / "mover_cleaned_flowsheet_observed_map" / "manifest.json"
        if cache_dir is not None
        else None
    )
    return SourceExtraction(cohort, maps, manifest, cache_manifest)


__all__ = [
    "CacheMismatchError",
    "COMMON_COHORT_COLUMNS",
    "COMMON_MAP_COLUMNS",
    "MAP_MAX",
    "MAP_MIN",
    "SourceExtraction",
    "SourceSchemaError",
    "build_inspire_common_cohort",
    "build_mover_common_cohort",
    "extract_inspire_common_cohort",
    "extract_inspire_source",
    "extract_mover_common_cohort",
    "extract_mover_source",
    "iter_inspire_map_observations",
    "iter_inspire_operation_chunks",
    "iter_mover_map_observations",
    "iter_mover_patient_chunks",
    "normalise_inspire_map_chunk",
    "normalise_mover_map_chunk",
    "parse_height_m",
    "parse_mover_weight_kg",
]

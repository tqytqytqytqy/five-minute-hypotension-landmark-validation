"""Observation-process engine for the INSPIRE -> MOVER LM5 study.

This module implements only prespecified observation and endpoint rules.  It
does not fit a model, impute a missing dynamic observation, interpolate a MAP
trajectory, or carry an observation forward.

Canonical input contracts
-------------------------
Raw MAP records use ``case_id``, ``minute_from_anesthesia_start``, ``map`` and
``source``.  The H5/R1 builders return one selected observation per case/grid
time with columns ``case_id``, ``time_min``, ``map`` and ``source`` (plus raw
record counts).  Static case data use ``case_id``, ``age_years``, ``male``,
``bmi`` and ``asa``.

The main H5 grid has phase 0 and right-closed intervals (0, 5], (5, 10], ... .
For a phase ``p`` the grid labels are ``p + k * width`` and each record is put
in the first right boundary greater than or equal to its time.  Records at or
before anaesthesia start are always excluded.  Thus phase sensitivity changes
the grid origin without importing pre-anaesthesia measurements.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import math

import numpy as np
import pandas as pd


MAP_MIN = 20.0
MAP_MAX = 200.0
HYPOTENSION_THRESHOLD = 65.0

FEATURE_COLUMNS: tuple[str, ...] = (
    "age_years",
    "male",
    "bmi",
    "asa",
    "t0_map",
    "t0_map_squared",
    "t0_arterial_source",
    "anesthesia_start_to_t0_min",
    "pre10_map_record_count",
    "pre10_last_measurement_gap_min",
    "pre10_last_map",
    "pre10_mean_map",
    "pre10_map_ols_slope_per_min",
    "recovered_by_5min",
    "early_auc65_0_5_mmhg_min",
    "early_min_map_0_5",
    "early_mean_map_0_5",
    "early_map_record_count_0_5",
)

_DEFAULT_ART_LABELS = (
    "ART",
    "ARTERIAL",
    "ART_MBP",
    "ABP",
    "A-LINE",
    "ALINE",
    "INVASIVE",
)
_DEFAULT_NIBP_LABELS = (
    "NIBP",
    "NIBP_MBP",
    "NBP",
    "NONINVASIVE",
    "NON-INVASIVE",
    "CUFF",
)
_ATOL = 1e-9


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _token(value: object) -> str:
    return "".join(character for character in str(value).upper() if character.isalnum())


def _source_lookup(
    art_labels: Iterable[object], nibp_labels: Iterable[object]
) -> dict[str, str]:
    lookup = {_token(value): "ART" for value in art_labels}
    for value in nibp_labels:
        key = _token(value)
        if key in lookup and lookup[key] != "NIBP":
            raise ValueError(f"source label is assigned to both ART and NIBP: {value!r}")
        lookup[key] = "NIBP"
    return lookup


def _validate_phase(width_min: float, phase_min: float) -> tuple[float, float]:
    width = float(width_min)
    phase = float(phase_min)
    if not np.isfinite(width) or width <= 0:
        raise ValueError("width_min must be a finite positive number")
    if not np.isfinite(phase) or not 0 <= phase < width:
        raise ValueError("phase_min must satisfy 0 <= phase_min < width_min")
    return width, phase


def harmonize_map(
    raw: pd.DataFrame,
    *,
    width_min: float,
    phase_min: float = 0.0,
    case_col: str = "case_id",
    time_col: str = "minute_from_anesthesia_start",
    map_col: str = "map",
    source_col: str = "source",
    art_labels: Iterable[object] = _DEFAULT_ART_LABELS,
    nibp_labels: Iterable[object] = _DEFAULT_NIBP_LABELS,
) -> pd.DataFrame:
    """QC and aggregate raw MAP records to a right-closed observation grid.

    MAP values 20--200 mmHg are retained, inclusive.  Within a case/interval,
    ART and NIBP medians are first calculated separately; ART is then selected
    whenever it exists, otherwise NIBP is selected.  Unknown source labels are
    a hard mapping error rather than being silently treated as NIBP.

    No interpolation or LOCF is performed.  An interval without a real record
    is absent from the returned frame.
    """

    _require_columns(raw, (case_col, time_col, map_col, source_col), "raw")
    width, phase = _validate_phase(width_min, phase_min)
    columns = [
        "case_id",
        "time_min",
        "map",
        "source",
        "n_raw_records",
        "n_raw_records_all_sources",
    ]
    if raw.empty:
        return pd.DataFrame(columns=columns)

    data = raw[[case_col, time_col, map_col, source_col]].copy()
    data.columns = ["case_id", "raw_time", "map", "raw_source"]
    data["raw_time"] = pd.to_numeric(data["raw_time"], errors="coerce")
    data["map"] = pd.to_numeric(data["map"], errors="coerce")
    data = data[
        data["case_id"].notna()
        & data["raw_time"].notna()
        & data["map"].notna()
        & data["raw_time"].gt(0.0)
        & data["map"].between(MAP_MIN, MAP_MAX, inclusive="both")
    ].copy()
    if data.empty:
        return pd.DataFrame(columns=columns)

    lookup = _source_lookup(art_labels, nibp_labels)
    data["source"] = data["raw_source"].map(lambda value: lookup.get(_token(value)))
    unknown = data.loc[data["source"].isna(), "raw_source"].drop_duplicates().tolist()
    if unknown:
        raise ValueError(f"unmapped MAP source labels: {unknown[:10]}")

    # Subtracting a tiny tolerance keeps a floating representation of an exact
    # boundary (e.g. 5.000000000000001) in its intended right-closed interval.
    scaled = (data["raw_time"].to_numpy(dtype=float) - phase) / width
    grid_number = np.ceil(scaled - 1e-12)
    data["time_min"] = phase + grid_number * width

    by_source = (
        data.groupby(["case_id", "time_min", "source"], as_index=False, sort=True)
        .agg(map=("map", "median"), n_raw_records=("map", "size"))
    )
    total = (
        data.groupby(["case_id", "time_min"], as_index=False, sort=True)
        .size()
        .rename(columns={"size": "n_raw_records_all_sources"})
    )
    by_source["source_priority"] = by_source["source"].map({"ART": 0, "NIBP": 1})
    selected = (
        by_source.sort_values(["case_id", "time_min", "source_priority"])
        .drop_duplicates(["case_id", "time_min"], keep="first")
        .merge(total, on=["case_id", "time_min"], how="left", validate="one_to_one")
    )
    return selected[columns].sort_values(["case_id", "time_min"]).reset_index(drop=True)


def normalize_grid_map(
    raw: pd.DataFrame,
    *,
    case_col: str = "case_id",
    time_col: str = "time_min",
    map_col: str = "map",
    source_col: str = "source",
    minimum_time_inclusive: float | None = 0.0,
    maximum_time_inclusive: float | None = None,
    art_labels: Iterable[object] = _DEFAULT_ART_LABELS,
    nibp_labels: Iterable[object] = _DEFAULT_NIBP_LABELS,
) -> pd.DataFrame:
    """Normalize an already-published grid without applying another binning.

    This is the INSPIRE-facing path: published grid timestamps are preserved
    exactly.  QC 20--200 and same-timestamp ART-over-NIBP selection still apply.
    The optional time limits restrict the anaesthesia-relative analysis period;
    they never shift or round a timestamp.
    """

    _require_columns(raw, (case_col, time_col, map_col, source_col), "raw_grid")
    columns = [
        "case_id",
        "time_min",
        "map",
        "source",
        "n_raw_records",
        "n_raw_records_all_sources",
    ]
    if raw.empty:
        return pd.DataFrame(columns=columns)
    data = raw[[case_col, time_col, map_col, source_col]].copy()
    data.columns = ["case_id", "time_min", "map", "raw_source"]
    data["time_min"] = pd.to_numeric(data["time_min"], errors="coerce")
    data["map"] = pd.to_numeric(data["map"], errors="coerce")
    keep = (
        data["case_id"].notna()
        & data["time_min"].notna()
        & data["map"].notna()
        & data["map"].between(MAP_MIN, MAP_MAX, inclusive="both")
    )
    if minimum_time_inclusive is not None:
        keep &= data["time_min"].ge(float(minimum_time_inclusive))
    if maximum_time_inclusive is not None:
        keep &= data["time_min"].le(float(maximum_time_inclusive))
    data = data[keep].copy()
    if data.empty:
        return pd.DataFrame(columns=columns)

    lookup = _source_lookup(art_labels, nibp_labels)
    data["source"] = data["raw_source"].map(lambda value: lookup.get(_token(value)))
    unknown = data.loc[data["source"].isna(), "raw_source"].drop_duplicates().tolist()
    if unknown:
        raise ValueError(f"unmapped MAP source labels: {unknown[:10]}")

    by_source = (
        data.groupby(["case_id", "time_min", "source"], as_index=False, sort=True)
        .agg(map=("map", "median"), n_raw_records=("map", "size"))
    )
    total = (
        data.groupby(["case_id", "time_min"], as_index=False, sort=True)
        .size()
        .rename(columns={"size": "n_raw_records_all_sources"})
    )
    by_source["source_priority"] = by_source["source"].map({"ART": 0, "NIBP": 1})
    selected = (
        by_source.sort_values(["case_id", "time_min", "source_priority"])
        .drop_duplicates(["case_id", "time_min"], keep="first")
        .merge(total, on=["case_id", "time_min"], how="left", validate="one_to_one")
    )
    return selected[columns].sort_values(["case_id", "time_min"]).reset_index(drop=True)


def build_h5(raw: pd.DataFrame, *, phase_min: float = 0.0, **kwargs: object) -> pd.DataFrame:
    """Build the prespecified 5-minute H5 series: (0, 5], (5, 10], ... ."""

    return harmonize_map(raw, width_min=5.0, phase_min=phase_min, **kwargs)


def build_r1(raw: pd.DataFrame, *, phase_min: float = 0.0, **kwargs: object) -> pd.DataFrame:
    """Build the prespecified 1-minute R1 series: (0, 1], (1, 2], ... ."""

    return harmonize_map(raw, width_min=1.0, phase_min=phase_min, **kwargs)


def _validate_series(series: pd.DataFrame, name: str = "series") -> pd.DataFrame:
    _require_columns(series, ("case_id", "time_min", "map", "source"), name)
    data = series[["case_id", "time_min", "map", "source"]].copy()
    data["time_min"] = pd.to_numeric(data["time_min"], errors="coerce")
    data["map"] = pd.to_numeric(data["map"], errors="coerce")
    data = data.dropna(subset=["case_id", "time_min", "map", "source"])
    data = data[data["map"].between(MAP_MIN, MAP_MAX, inclusive="both")].copy()
    duplicated = data.duplicated(["case_id", "time_min"], keep=False)
    if duplicated.any():
        examples = data.loc[duplicated, ["case_id", "time_min"]].head().to_dict("records")
        raise ValueError(f"{name} has duplicate case/grid times: {examples}")
    data["source"] = data["source"].astype(str).str.upper()
    invalid_sources = sorted(set(data["source"]) - {"ART", "NIBP"})
    if invalid_sources:
        raise ValueError(f"{name} contains non-canonical sources: {invalid_sources}")
    return data.sort_values(["case_id", "time_min"]).reset_index(drop=True)


def find_h5_t0(
    h5: pd.DataFrame,
    *,
    threshold: float = HYPOTENSION_THRESHOLD,
    search_start_exclusive: float = 3.0,
    search_end_inclusive: float = 30.0,
) -> pd.DataFrame:
    """Find the first H5 grid MAP <65 after >3 and at <=30 anaesthesia minutes."""

    data = _validate_series(h5, "h5")
    candidates = data[
        data["time_min"].gt(float(search_start_exclusive))
        & data["time_min"].le(float(search_end_inclusive))
        & data["map"].lt(float(threshold))
    ].copy()
    if candidates.empty:
        return pd.DataFrame(columns=["case_id", "t0_min", "t0_map", "t0_source"])
    first = (
        candidates.sort_values(["case_id", "time_min"])
        .drop_duplicates("case_id", keep="first")
        .rename(columns={"time_min": "t0_min", "map": "t0_map", "source": "t0_source"})
    )
    return first[["case_id", "t0_min", "t0_map", "t0_source"]].reset_index(drop=True)


def _validate_t0(t0: pd.DataFrame) -> pd.DataFrame:
    _require_columns(t0, ("case_id", "t0_min"), "t0")
    data = t0.copy()
    data["t0_min"] = pd.to_numeric(data["t0_min"], errors="coerce")
    if data[["case_id", "t0_min"]].isna().any().any():
        raise ValueError("t0 contains missing case_id or t0_min")
    if data["case_id"].duplicated().any():
        raise ValueError("t0 must contain at most one row per case_id")
    return data.reset_index(drop=True)


def _point_at(group: pd.DataFrame, time_min: float) -> pd.Series | None:
    match = group[np.isclose(group["time_min"], float(time_min), atol=_ATOL, rtol=0.0)]
    if match.empty:
        return None
    if len(match) != 1:
        raise ValueError("observation series must contain one value per grid time")
    return match.iloc[0]


def _trapezoid_auc_below(points: pd.DataFrame, threshold: float) -> float:
    """Trapezoidal deficit integral over actual adjacent observations only."""

    ordered = points.sort_values("time_min")
    if len(ordered) < 2:
        return math.nan
    times = ordered["time_min"].to_numpy(dtype=float)
    deficits = np.maximum(float(threshold) - ordered["map"].to_numpy(dtype=float), 0.0)
    return float(np.sum(np.diff(times) * (deficits[:-1] + deficits[1:]) / 2.0))


def _reason(**conditions: bool) -> str:
    return ";".join(name for name, failed in conditions.items() if failed)


def build_lm5_common18(
    cases: pd.DataFrame,
    series: pd.DataFrame,
    t0: pd.DataFrame,
    *,
    threshold: float = HYPOTENSION_THRESHOLD,
    landmark_offset_min: float = 5.0,
    include_ineligible: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct the prespecified 18 features and a row-level eligibility audit.

    The pre-t0 slope is the ordinary least-squares slope of MAP on time using
    only actual observations in [t0-10, t0).  It requires at least two distinct
    observed times; otherwise it is NaN.  Individual missing static values
    remain NaN for later use of the frozen INSPIRE imputation parameters.

    t0 and t0+5 must both be real observations in the selected process.  With
    ``include_ineligible=False`` (default), rows failing this boundary rule are
    present only in the returned audit.  No value after t0+5 is inspected.
    """

    _require_columns(cases, ("case_id", "age_years", "male", "bmi", "asa"), "cases")
    if cases["case_id"].duplicated().any():
        raise ValueError("cases must contain one row per case_id")
    observations = _validate_series(series)
    landmarks = _validate_t0(t0)
    static = cases.set_index("case_id", drop=False)
    grouped = {case_id: group for case_id, group in observations.groupby("case_id", sort=False)}

    feature_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for landmark in landmarks.itertuples(index=False):
        case_id = landmark.case_id
        zero = float(landmark.t0_min)
        prediction_time = zero + float(landmark_offset_min)
        group = grouped.get(case_id, observations.iloc[0:0])
        t0_point = _point_at(group, zero)
        landmark_point = _point_at(group, prediction_time)
        has_static = case_id in static.index
        evaluable = bool(has_static and t0_point is not None and landmark_point is not None)
        reason = _reason(
            missing_case_covariates_row=not has_static,
            missing_t0_observation=t0_point is None,
            missing_t0_plus_5_observation=landmark_point is None,
        )
        audit_rows.append(
            {
                "case_id": case_id,
                "t0_min": zero,
                "landmark_min": prediction_time,
                "has_case_covariates_row": int(has_static),
                "has_t0_observation": int(t0_point is not None),
                "has_landmark_observation": int(landmark_point is not None),
                "feature_evaluable": int(evaluable),
                "feature_exclusion_reason": reason,
            }
        )

        static_row: Mapping[str, object] = static.loc[case_id] if has_static else {}
        row: dict[str, object] = {
            "case_id": case_id,
            "age_years": pd.to_numeric(static_row.get("age_years"), errors="coerce"),
            "male": pd.to_numeric(static_row.get("male"), errors="coerce"),
            "bmi": pd.to_numeric(static_row.get("bmi"), errors="coerce"),
            "asa": pd.to_numeric(static_row.get("asa"), errors="coerce"),
            "t0_map": math.nan,
            "t0_map_squared": math.nan,
            "t0_arterial_source": math.nan,
            "anesthesia_start_to_t0_min": zero,
            "pre10_map_record_count": 0,
            "pre10_last_measurement_gap_min": math.nan,
            "pre10_last_map": math.nan,
            "pre10_mean_map": math.nan,
            "pre10_map_ols_slope_per_min": math.nan,
            "recovered_by_5min": math.nan,
            "early_auc65_0_5_mmhg_min": math.nan,
            "early_min_map_0_5": math.nan,
            "early_mean_map_0_5": math.nan,
            "early_map_record_count_0_5": math.nan,
        }

        pre = group[
            group["time_min"].ge(zero - 10.0 - _ATOL)
            & group["time_min"].lt(zero - _ATOL)
        ].sort_values("time_min")
        if not pre.empty:
            row["pre10_map_record_count"] = int(len(pre))
            row["pre10_last_measurement_gap_min"] = float(zero - pre.iloc[-1]["time_min"])
            row["pre10_last_map"] = float(pre.iloc[-1]["map"])
            row["pre10_mean_map"] = float(pre["map"].mean())
            x = pre["time_min"].to_numpy(dtype=float)
            y = pre["map"].to_numpy(dtype=float)
            if len(x) >= 2 and np.unique(x).size >= 2:
                x_centered = x - float(np.mean(x))
                denominator = float(np.sum(x_centered**2))
                if denominator > 0:
                    row["pre10_map_ols_slope_per_min"] = float(
                        np.sum(x_centered * (y - float(np.mean(y)))) / denominator
                    )

        if t0_point is not None:
            t0_map = float(t0_point["map"])
            row["t0_map"] = t0_map
            row["t0_map_squared"] = t0_map**2
            row["t0_arterial_source"] = float(str(t0_point["source"]).upper() == "ART")

        if t0_point is not None and landmark_point is not None:
            early = group[
                group["time_min"].ge(zero - _ATOL)
                & group["time_min"].le(prediction_time + _ATOL)
            ].sort_values("time_min")
            after_t0 = early[early["time_min"].gt(zero + _ATOL)]
            row["recovered_by_5min"] = int(after_t0["map"].ge(float(threshold)).any())
            row["early_auc65_0_5_mmhg_min"] = _trapezoid_auc_below(early, float(threshold))
            row["early_min_map_0_5"] = float(early["map"].min())
            row["early_mean_map_0_5"] = float(early["map"].mean())
            row["early_map_record_count_0_5"] = int(len(early))

        if include_ineligible or evaluable:
            feature_rows.append(row)

    features = pd.DataFrame(feature_rows, columns=("case_id",) + FEATURE_COLUMNS)
    audit = pd.DataFrame(audit_rows)
    return features, audit


def classify_stage1(
    series: pd.DataFrame,
    t0: pd.DataFrame,
    *,
    threshold: float = HYPOTENSION_THRESHOLD,
    landmark_offset_min: float = 5.0,
    auc_alert_threshold: float = 50.0,
) -> pd.DataFrame:
    """Apply the fixed stage-1 recurrence/high-burden rule in [t0, t0+5]."""

    observations = _validate_series(series)
    landmarks = _validate_t0(t0)
    grouped = {case_id: group for case_id, group in observations.groupby("case_id", sort=False)}
    rows: list[dict[str, object]] = []
    for landmark in landmarks.itertuples(index=False):
        case_id = landmark.case_id
        zero = float(landmark.t0_min)
        prediction_time = zero + float(landmark_offset_min)
        group = grouped.get(case_id, observations.iloc[0:0])
        t0_point = _point_at(group, zero)
        landmark_point = _point_at(group, prediction_time)
        evaluable = t0_point is not None and landmark_point is not None
        recovered: object = pd.NA
        recurrent: object = pd.NA
        auc = math.nan
        high_alert: object = pd.NA
        stage2: object = pd.NA
        if evaluable:
            early = group[
                group["time_min"].ge(zero - _ATOL)
                & group["time_min"].le(prediction_time + _ATOL)
            ].sort_values("time_min")
            recovered_points = early[
                early["time_min"].gt(zero + _ATOL)
                & early["map"].ge(float(threshold))
            ]
            recovered = int(not recovered_points.empty)
            if recovered_points.empty:
                recurrent = 0
            else:
                recovery_time = float(recovered_points.iloc[0]["time_min"])
                recurrent = int(
                    early[
                        early["time_min"].gt(recovery_time + _ATOL)
                        & early["map"].lt(float(threshold))
                    ].shape[0]
                    > 0
                )
            auc = _trapezoid_auc_below(early, float(threshold))
            high_alert = int(bool(recurrent) or auc >= float(auc_alert_threshold))
            stage2 = int(not bool(high_alert))
        rows.append(
            {
                "case_id": case_id,
                "t0_min": zero,
                "landmark_min": prediction_time,
                "stage1_evaluable": int(evaluable),
                "recovered_by_5min": recovered,
                "early_recurrence_after_recovery": recurrent,
                "auc_below_65_0_5": auc,
                "stage1_high_alert": high_alert,
                "stage2_eligible": stage2,
                "stage1_exclusion_reason": _reason(
                    missing_t0_observation=t0_point is None,
                    missing_t0_plus_5_observation=landmark_point is None,
                ),
            }
        )
    return pd.DataFrame(rows)


def classify_future_outcome(
    series: pd.DataFrame,
    t0: pd.DataFrame,
    *,
    process: str,
    threshold: float = HYPOTENSION_THRESHOLD,
    landmark_offset_min: float = 5.0,
    horizon_offset_min: float = 30.0,
    persistent_recovery_after_min: float = 10.0,
    auc_event_threshold: float = 75.0,
) -> pd.DataFrame:
    """Classify the prespecified future endpoint under H5 or R1 observation.

    The t0+5 point is an AUC anchor but is not searched for future recovery.
    Recovery is the first *actual* grid observation >=65 strictly after the
    landmark.  H5 requires exact t0+5/t0+30 anchors, >=5 of its six expected
    grids and maximum adjacent available gap <=10 minutes.  R1 requires the
    exact anchors and maximum adjacent actual gap <=5 minutes.
    """

    mode = str(process).upper()
    if mode not in {"H5", "R1"}:
        raise ValueError("process must be 'H5' or 'R1'")
    observations = _validate_series(series)
    landmarks = _validate_t0(t0)
    grouped = {case_id: group for case_id, group in observations.groupby("case_id", sort=False)}
    rows: list[dict[str, object]] = []
    for landmark in landmarks.itertuples(index=False):
        case_id = landmark.case_id
        zero = float(landmark.t0_min)
        landmark_time = zero + float(landmark_offset_min)
        horizon_time = zero + float(horizon_offset_min)
        group = grouped.get(case_id, observations.iloc[0:0]).copy()
        window = group[
            group["time_min"].ge(landmark_time - _ATOL)
            & group["time_min"].le(horizon_time + _ATOL)
        ].sort_values("time_min")
        start_point = _point_at(group, landmark_time)
        end_point = _point_at(group, horizon_time)

        if mode == "H5":
            expected_times = zero + np.arange(
                float(landmark_offset_min), float(horizon_offset_min) + _ATOL, 5.0
            )
            mask = np.zeros(len(window), dtype=bool)
            for expected in expected_times:
                mask |= np.isclose(
                    window["time_min"].to_numpy(dtype=float), expected, atol=_ATOL, rtol=0.0
                )
            integration = window.loc[mask].copy()
            n_required_grid_points = int(len(integration))
            minimum_points_ok = n_required_grid_points >= 5
            allowed_gap = 10.0
        else:
            integration = window.copy()
            n_required_grid_points = int(len(integration))
            minimum_points_ok = len(integration) >= 2
            allowed_gap = 5.0

        times = integration["time_min"].to_numpy(dtype=float)
        max_gap = float(np.max(np.diff(times))) if len(times) >= 2 else math.nan
        gap_ok = bool(len(times) >= 2 and max_gap <= allowed_gap + _ATOL)
        evaluable = bool(
            start_point is not None
            and end_point is not None
            and minimum_points_ok
            and gap_ok
        )

        recovery_after_landmark = math.nan
        persistent: object = pd.NA
        auc = math.nan
        high_burden: object = pd.NA
        primary: object = pd.NA
        if evaluable:
            future_recovery = integration[
                integration["time_min"].gt(landmark_time + _ATOL)
                & integration["map"].ge(float(threshold))
            ]
            if not future_recovery.empty:
                recovery_after_landmark = float(
                    future_recovery.iloc[0]["time_min"] - landmark_time
                )
            persistent = int(
                math.isnan(recovery_after_landmark)
                or recovery_after_landmark > float(persistent_recovery_after_min)
            )
            auc = _trapezoid_auc_below(integration, float(threshold))
            high_burden = int(auc >= float(auc_event_threshold))
            primary = int(bool(persistent) or bool(high_burden))

        rows.append(
            {
                "case_id": case_id,
                "t0_min": zero,
                "landmark_min": landmark_time,
                "horizon_min": horizon_time,
                "outcome_process": mode,
                "has_t0_plus_5_anchor": int(start_point is not None),
                "has_t0_plus_30_anchor": int(end_point is not None),
                "n_outcome_grid_points": n_required_grid_points,
                "max_adjacent_gap_min": max_gap,
                "outcome_evaluable": int(evaluable),
                "first_recovery_after_landmark_min": recovery_after_landmark,
                "future_persistent_recovery_gt10": persistent,
                "future_auc_below_65_5_30": auc,
                "future_high_burden_auc_ge75": high_burden,
                "primary_outcome": primary,
                "outcome_exclusion_reason": _reason(
                    missing_t0_plus_5_anchor=start_point is None,
                    missing_t0_plus_30_anchor=end_point is None,
                    insufficient_grid_points=not minimum_points_ok,
                    excessive_adjacent_gap=not gap_ok,
                ),
            }
        )
    return pd.DataFrame(rows)


def build_observation_2x2(
    cases: pd.DataFrame,
    h5: pd.DataFrame,
    r1: pd.DataFrame,
    h5_t0: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build fixed-H5-patient/t0 H5/R1 input-by-label analysis cells.

    The fixed base cohort is the H5 main stage-2 risk set: H5 features and H5
    outcome are evaluable and the H5 stage-1 rule does not directly alert.  All
    four cells retain exactly those patients and the same H5-derived t0.  If an
    R1 feature or outcome cannot be formed, its row remains in the cell table
    with an explicit eligibility flag; it is never silently removed or filled.
    """

    landmarks = _validate_t0(h5_t0)
    h5_features, h5_feature_audit = build_lm5_common18(
        cases, h5, landmarks, include_ineligible=True
    )
    r1_features, r1_feature_audit = build_lm5_common18(
        cases, r1, landmarks, include_ineligible=True
    )
    h5_stage1 = classify_stage1(h5, landmarks)
    h5_outcome = classify_future_outcome(h5, landmarks, process="H5")
    r1_outcome = classify_future_outcome(r1, landmarks, process="R1")

    base = (
        landmarks[["case_id", "t0_min"]]
        .merge(h5_feature_audit[["case_id", "feature_evaluable"]], on="case_id", how="left")
        .merge(h5_stage1[["case_id", "stage1_evaluable", "stage2_eligible"]], on="case_id", how="left")
        .merge(h5_outcome[["case_id", "outcome_evaluable"]], on="case_id", how="left")
    )
    keep = (
        base["feature_evaluable"].eq(1)
        & base["stage1_evaluable"].eq(1)
        & base["stage2_eligible"].eq(1)
        & base["outcome_evaluable"].eq(1)
    )
    fixed = base.loc[keep, ["case_id", "t0_min"]].copy()

    feature_sets = {
        "H5": (h5_features, h5_feature_audit),
        "R1": (r1_features, r1_feature_audit),
    }
    outcome_sets = {"H5": h5_outcome, "R1": r1_outcome}
    cells: list[pd.DataFrame] = []
    audits: list[pd.DataFrame] = []
    for feature_process, (features, feature_audit) in feature_sets.items():
        feature_block = fixed.merge(features, on="case_id", how="left", validate="one_to_one")
        feature_block = feature_block.merge(
            feature_audit[["case_id", "feature_evaluable", "feature_exclusion_reason"]],
            on="case_id",
            how="left",
            validate="one_to_one",
        )
        for outcome_process, outcomes in outcome_sets.items():
            block = feature_block.merge(
                outcomes.drop(columns=["t0_min"], errors="ignore"),
                on="case_id",
                how="left",
                validate="one_to_one",
            )
            block["feature_process"] = feature_process
            block["outcome_process"] = outcome_process
            block["cell"] = f"{feature_process}_features__{outcome_process}_outcome"
            block["cell_evaluable"] = (
                block["feature_evaluable"].eq(1) & block["outcome_evaluable"].eq(1)
            ).astype(int)
            cells.append(block)
            audits.append(
                block[
                    [
                        "case_id",
                        "t0_min",
                        "feature_process",
                        "outcome_process",
                        "cell",
                        "feature_evaluable",
                        "feature_exclusion_reason",
                        "outcome_evaluable",
                        "outcome_exclusion_reason",
                        "cell_evaluable",
                    ]
                ].copy()
            )

    cell_frame = pd.concat(cells, ignore_index=True) if cells else pd.DataFrame()
    audit_frame = pd.concat(audits, ignore_index=True) if audits else pd.DataFrame()
    return cell_frame, audit_frame


__all__ = [
    "FEATURE_COLUMNS",
    "HYPOTENSION_THRESHOLD",
    "MAP_MAX",
    "MAP_MIN",
    "build_h5",
    "build_lm5_common18",
    "build_observation_2x2",
    "build_r1",
    "classify_future_outcome",
    "classify_stage1",
    "find_h5_t0",
    "harmonize_map",
    "normalize_grid_map",
]

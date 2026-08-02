"""Contract tests for the prespecified H5/R1 observation engine."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


DELIVERY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DELIVERY_ROOT / "02_code_configs" / "src"))

from lm5_validation.observation import (  # noqa: E402
    FEATURE_COLUMNS,
    build_h5,
    build_lm5_common18,
    build_observation_2x2,
    build_r1,
    classify_future_outcome,
    classify_stage1,
    find_h5_t0,
    normalize_grid_map,
)


def canonical_series(case_id: str, times: list[float], maps: list[float], source: str = "ART") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": case_id,
            "time_min": times,
            "map": maps,
            "source": source,
        }
    )


class HarmonizationTests(unittest.TestCase):
    def test_map_qc_is_inclusive_and_right_boundary_is_closed(self) -> None:
        raw = pd.DataFrame(
            {
                "case_id": ["A"] * 8,
                "minute_from_anesthesia_start": [0.0, 0.1, 1.0, 2.0, 5.0, 5.0001, 6.0, 7.0],
                "map": [60, 19.9, 20, 200, 70, 80, 200.1, np.nan],
                "source": ["ART"] * 8,
            }
        )
        h5 = build_h5(raw)

        self.assertEqual(h5["time_min"].tolist(), [5.0, 10.0])
        # t=0, MAP<20, MAP>200 and missing MAP are absent.  t=5 belongs to
        # (0,5], while t=5.0001 belongs to (5,10].
        self.assertAlmostEqual(float(h5.iloc[0]["map"]), 70.0)
        self.assertAlmostEqual(float(h5.iloc[1]["map"]), 80.0)

    def test_median_is_computed_within_source_then_art_is_selected(self) -> None:
        raw = pd.DataFrame(
            {
                "case_id": ["A"] * 5 + ["B"] * 2,
                "minute_from_anesthesia_start": [1, 2, 3, 4, 4.5, 1, 4],
                "map": [50, 70, 90, 120, 140, 80, 100],
                "source": ["ART", "ART", "ART", "NIBP", "NIBP", "NIBP", "NIBP"],
            }
        )
        h5 = build_h5(raw).set_index("case_id")

        self.assertEqual(h5.loc["A", "source"], "ART")
        self.assertEqual(float(h5.loc["A", "map"]), 70.0)
        self.assertEqual(int(h5.loc["A", "n_raw_records"]), 3)
        self.assertEqual(int(h5.loc["A", "n_raw_records_all_sources"]), 5)
        self.assertEqual(h5.loc["B", "source"], "NIBP")
        self.assertEqual(float(h5.loc["B", "map"]), 90.0)

    def test_unknown_source_is_a_hard_mapping_failure(self) -> None:
        raw = pd.DataFrame(
            {
                "case_id": ["A"],
                "minute_from_anesthesia_start": [1.0],
                "map": [60.0],
                "source": ["mystery"],
            }
        )
        with self.assertRaisesRegex(ValueError, "unmapped MAP source"):
            build_h5(raw)

    def test_phase_changes_grid_origin_without_using_pre_anesthesia_data(self) -> None:
        raw = pd.DataFrame(
            {
                "case_id": ["A"] * 4,
                "minute_from_anesthesia_start": [-0.5, 0.5, 4.0, 4.1],
                "map": [20, 60, 70, 80],
                "source": ["ART"] * 4,
            }
        )
        h5_phase_0 = build_h5(raw, phase_min=0)
        h5_phase_4 = build_h5(raw, phase_min=4)

        self.assertEqual(h5_phase_0["time_min"].tolist(), [5.0])
        self.assertEqual(h5_phase_4["time_min"].tolist(), [4.0, 9.0])
        self.assertEqual(float(h5_phase_4.iloc[0]["map"]), 65.0)
        self.assertNotIn(20.0, h5_phase_4["map"].tolist())

    def test_r1_uses_right_closed_one_minute_bins(self) -> None:
        raw = pd.DataFrame(
            {
                "case_id": ["A"] * 3,
                "minute_from_anesthesia_start": [0.2, 1.0, 1.001],
                "map": [60, 70, 80],
                "source": ["NIBP"] * 3,
            }
        )
        r1 = build_r1(raw)
        self.assertEqual(r1["time_min"].tolist(), [1.0, 2.0])
        self.assertEqual(r1["map"].tolist(), [65.0, 80.0])

    def test_published_grid_is_normalized_without_rebinning(self) -> None:
        published = pd.DataFrame(
            {
                "op": ["A", "A", "A"],
                "chart": [0.0, 7.25, 7.25],
                "value": [70, 60, 100],
                "item": ["NIBP", "ART", "NIBP"],
            }
        )
        grid = normalize_grid_map(
            published, case_col="op", time_col="chart", map_col="value", source_col="item"
        )
        self.assertEqual(grid["time_min"].tolist(), [0.0, 7.25])
        self.assertEqual(grid["map"].tolist(), [70.0, 60.0])
        self.assertEqual(grid["source"].tolist(), ["NIBP", "ART"])


class T0AndFeatureTests(unittest.TestCase):
    def test_t0_is_detected_only_after_h5_formation_and_within_search_bounds(self) -> None:
        # The two raw lows in (0,5] are outweighed by a high value, so the H5
        # median is not hypotensive.  The first H5 low is therefore at 10 min.
        raw = pd.DataFrame(
            {
                "case_id": ["A"] * 6 + ["B"] * 2,
                "minute_from_anesthesia_start": [1, 2, 4, 6, 7, 31, 1, 4],
                "map": [60, 70, 100, 60, 62, 50, 50, 55],
                "source": ["ART"] * 8,
            }
        )
        h5 = build_h5(raw)
        t0 = find_h5_t0(h5).set_index("case_id")

        self.assertEqual(float(t0.loc["A", "t0_min"]), 10.0)
        self.assertEqual(float(t0.loc["A", "t0_map"]), 61.0)
        self.assertEqual(float(t0.loc["B", "t0_min"]), 5.0)
        self.assertNotEqual(float(t0.loc["A", "t0_min"]), 35.0)

    def test_common18_feature_values_and_exact_contract(self) -> None:
        cases = pd.DataFrame(
            {"case_id": ["A"], "age_years": [70], "male": [1], "bmi": [25], "asa": [3]}
        )
        h5 = pd.concat(
            [
                canonical_series(
                    "A",
                    [5, 10, 15, 20, 25],
                    [80, 70, 60, 70, 200],
                )
            ],
            ignore_index=True,
        )
        t0 = pd.DataFrame({"case_id": ["A"], "t0_min": [15.0]})

        features, audit = build_lm5_common18(cases, h5, t0)
        self.assertEqual(features.columns.tolist(), ["case_id", *FEATURE_COLUMNS])
        self.assertEqual(len(FEATURE_COLUMNS), 18)
        row = features.iloc[0]
        self.assertEqual(float(row["t0_map"]), 60.0)
        self.assertEqual(float(row["t0_map_squared"]), 3600.0)
        self.assertEqual(float(row["t0_arterial_source"]), 1.0)
        self.assertEqual(float(row["pre10_map_record_count"]), 2.0)
        self.assertEqual(float(row["pre10_last_measurement_gap_min"]), 5.0)
        self.assertEqual(float(row["pre10_last_map"]), 70.0)
        self.assertEqual(float(row["pre10_mean_map"]), 75.0)
        self.assertEqual(float(row["pre10_map_ols_slope_per_min"]), -2.0)
        self.assertEqual(float(row["recovered_by_5min"]), 1.0)
        self.assertEqual(float(row["early_auc65_0_5_mmhg_min"]), 12.5)
        self.assertEqual(float(row["early_min_map_0_5"]), 60.0)
        self.assertEqual(float(row["early_mean_map_0_5"]), 65.0)
        self.assertEqual(float(row["early_map_record_count_0_5"]), 2.0)
        self.assertEqual(int(audit.iloc[0]["feature_evaluable"]), 1)

    def test_pre10_ols_slope_requires_two_actual_points(self) -> None:
        cases = pd.DataFrame(
            {"case_id": ["A"], "age_years": [60], "male": [0], "bmi": [22], "asa": [2]}
        )
        series = canonical_series("A", [5, 10, 15], [80, 60, 70])
        t0 = pd.DataFrame({"case_id": ["A"], "t0_min": [10.0]})
        features, _ = build_lm5_common18(cases, series, t0)
        self.assertTrue(np.isnan(features.iloc[0]["pre10_map_ols_slope_per_min"]))

    def test_features_do_not_read_values_after_landmark(self) -> None:
        cases = pd.DataFrame(
            {"case_id": ["A"], "age_years": [60], "male": [0], "bmi": [22], "asa": [2]}
        )
        base = canonical_series("A", [5, 10, 15], [70, 60, 70])
        with_future = pd.concat(
            [base, canonical_series("A", [20, 25, 30], [20, 200, 20])], ignore_index=True
        )
        t0 = pd.DataFrame({"case_id": ["A"], "t0_min": [10.0]})

        before, _ = build_lm5_common18(cases, base, t0)
        after, _ = build_lm5_common18(cases, with_future, t0)
        pd.testing.assert_frame_equal(before, after)

    def test_missing_landmark_is_not_locf_filled(self) -> None:
        cases = pd.DataFrame(
            {"case_id": ["A"], "age_years": [60], "male": [0], "bmi": [22], "asa": [2]}
        )
        h5 = canonical_series("A", [5, 10, 20], [70, 60, 80])
        t0 = pd.DataFrame({"case_id": ["A"], "t0_min": [10.0]})

        features, audit = build_lm5_common18(cases, h5, t0)
        self.assertTrue(features.empty)
        self.assertEqual(int(audit.iloc[0]["feature_evaluable"]), 0)
        self.assertIn("missing_t0_plus_5_observation", audit.iloc[0]["feature_exclusion_reason"])


class Stage1Tests(unittest.TestCase):
    def test_recovery_then_recurrence_triggers_stage1(self) -> None:
        r1 = canonical_series("A", [10, 11, 12, 13, 14, 15], [60, 66, 64, 70, 70, 70])
        t0 = pd.DataFrame({"case_id": ["A"], "t0_min": [10.0]})
        result = classify_stage1(r1, t0).iloc[0]

        self.assertEqual(result["recovered_by_5min"], 1)
        self.assertEqual(result["early_recurrence_after_recovery"], 1)
        self.assertEqual(result["stage1_high_alert"], 1)
        self.assertEqual(result["stage2_eligible"], 0)

    def test_auc_equal_to_50_triggers_stage1(self) -> None:
        h5 = canonical_series("A", [10, 15], [55, 55])
        t0 = pd.DataFrame({"case_id": ["A"], "t0_min": [10.0]})
        result = classify_stage1(h5, t0).iloc[0]

        self.assertEqual(float(result["auc_below_65_0_5"]), 50.0)
        self.assertEqual(result["stage1_high_alert"], 1)

    def test_missing_landmark_is_unevaluable_not_locf(self) -> None:
        h5 = canonical_series("A", [10, 20], [60, 70])
        t0 = pd.DataFrame({"case_id": ["A"], "t0_min": [10.0]})
        result = classify_stage1(h5, t0).iloc[0]
        self.assertEqual(result["stage1_evaluable"], 0)
        self.assertTrue(pd.isna(result["stage1_high_alert"]))


class FutureOutcomeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.t0 = pd.DataFrame({"case_id": ["A"], "t0_min": [10.0]})

    def test_h5_requires_boundaries_five_of_six_grids_and_gap_at_most_10(self) -> None:
        # Relative outcome grids are 5,10,15,20,25,30. One missing internal
        # point is allowed and creates exactly a 10-minute adjacent gap.
        h5 = canonical_series("A", [15, 20, 30, 35, 40], [60, 60, 66, 70, 70])
        result = classify_future_outcome(h5, self.t0, process="H5").iloc[0]
        self.assertEqual(result["n_outcome_grid_points"], 5)
        self.assertEqual(float(result["max_adjacent_gap_min"]), 10.0)
        self.assertEqual(result["outcome_evaluable"], 1)

        two_missing = h5[h5["time_min"].ne(35)].copy()
        failed = classify_future_outcome(two_missing, self.t0, process="H5").iloc[0]
        self.assertEqual(failed["n_outcome_grid_points"], 4)
        self.assertEqual(failed["outcome_evaluable"], 0)

    def test_h5_missing_either_anchor_is_unevaluable(self) -> None:
        full = canonical_series("A", [15, 20, 25, 30, 35, 40], [60, 60, 60, 70, 70, 70])
        for missing in (15, 40):
            with self.subTest(missing=missing):
                data = full[full["time_min"].ne(missing)]
                result = classify_future_outcome(data, self.t0, process="H5").iloc[0]
                self.assertEqual(result["outcome_evaluable"], 0)

    def test_future_endpoint_components_and_auc(self) -> None:
        non_event = canonical_series("A", [15, 20, 25, 30, 35, 40], [60, 60, 66, 70, 70, 70])
        result = classify_future_outcome(non_event, self.t0, process="H5").iloc[0]
        self.assertEqual(float(result["first_recovery_after_landmark_min"]), 10.0)
        self.assertEqual(result["future_persistent_recovery_gt10"], 0)
        self.assertAlmostEqual(float(result["future_auc_below_65_5_30"]), 37.5)
        self.assertEqual(result["future_high_burden_auc_ge75"], 0)
        self.assertEqual(result["primary_outcome"], 0)

        event = canonical_series("A", [15, 20, 25, 30, 35, 40], [60] * 6)
        event_result = classify_future_outcome(event, self.t0, process="H5").iloc[0]
        self.assertEqual(float(event_result["future_auc_below_65_5_30"]), 125.0)
        self.assertEqual(event_result["future_persistent_recovery_gt10"], 1)
        self.assertEqual(event_result["future_high_burden_auc_ge75"], 1)
        self.assertEqual(event_result["primary_outcome"], 1)

    def test_landmark_anchor_is_not_counted_as_future_recovery(self) -> None:
        h5 = canonical_series("A", [15, 20, 25, 30, 35, 40], [70, 60, 60, 60, 60, 60])
        result = classify_future_outcome(h5, self.t0, process="H5").iloc[0]
        self.assertTrue(np.isnan(result["first_recovery_after_landmark_min"]))
        self.assertEqual(result["future_persistent_recovery_gt10"], 1)

    def test_r1_uses_actual_points_and_maximum_gap_five(self) -> None:
        allowed = canonical_series("A", [15, 20, 25, 30, 35, 40], [60, 60, 66, 70, 70, 70])
        result = classify_future_outcome(allowed, self.t0, process="R1").iloc[0]
        self.assertEqual(result["outcome_evaluable"], 1)
        self.assertEqual(float(result["max_adjacent_gap_min"]), 5.0)

        excessive = canonical_series("A", [15, 21, 26, 31, 36, 40], [60, 60, 66, 70, 70, 70])
        failed = classify_future_outcome(excessive, self.t0, process="R1").iloc[0]
        self.assertEqual(float(failed["max_adjacent_gap_min"]), 6.0)
        self.assertEqual(failed["outcome_evaluable"], 0)

    def test_values_after_t0_plus_30_do_not_change_outcome(self) -> None:
        base = canonical_series("A", [15, 20, 25, 30, 35, 40], [60, 60, 66, 70, 70, 70])
        extra = pd.concat(
            [base, canonical_series("A", [41, 45, 100], [20, 20, 20])], ignore_index=True
        )
        before = classify_future_outcome(base, self.t0, process="H5")
        after = classify_future_outcome(extra, self.t0, process="H5")
        pd.testing.assert_frame_equal(before, after)


class ObservationTwoByTwoTests(unittest.TestCase):
    def test_all_four_cells_keep_identical_h5_patient_and_t0(self) -> None:
        cases = pd.DataFrame(
            {
                "case_id": ["A", "B"],
                "age_years": [60, 70],
                "male": [1, 0],
                "bmi": [24, 26],
                "asa": [2, 3],
            }
        )
        # A enters H5 stage 2 and has an evaluable H5 outcome. B triggers stage
        # 1 (AUC=50) and must not enter any of the four analysis cells.
        h5 = pd.concat(
            [
                canonical_series("A", [5, 10, 15, 20, 25, 30, 35, 40], [70, 60, 70, 70, 70, 70, 70, 70]),
                canonical_series("B", [5, 10, 15, 20, 25, 30, 35, 40], [70, 55, 55, 70, 70, 70, 70, 70]),
            ],
            ignore_index=True,
        )
        # R1 is deliberately sparse but respects the <=5-minute outcome gap.
        r1 = h5.copy()
        h5_t0 = pd.DataFrame({"case_id": ["A", "B"], "t0_min": [10.0, 10.0]})

        cells, audit = build_observation_2x2(cases, h5, r1, h5_t0)
        self.assertEqual(set(cells["cell"]), {
            "H5_features__H5_outcome",
            "H5_features__R1_outcome",
            "R1_features__H5_outcome",
            "R1_features__R1_outcome",
        })
        self.assertEqual(cells.groupby("cell")["case_id"].apply(list).to_dict(), {
            "H5_features__H5_outcome": ["A"],
            "H5_features__R1_outcome": ["A"],
            "R1_features__H5_outcome": ["A"],
            "R1_features__R1_outcome": ["A"],
        })
        self.assertTrue(cells["t0_min"].eq(10.0).all())
        self.assertEqual(len(audit), 4)

    def test_r1_missing_boundary_is_retained_and_flagged_not_carried_forward(self) -> None:
        cases = pd.DataFrame(
            {"case_id": ["A"], "age_years": [60], "male": [1], "bmi": [24], "asa": [2]}
        )
        h5 = canonical_series("A", [5, 10, 15, 20, 25, 30, 35, 40], [70, 60, 70, 70, 70, 70, 70, 70])
        # R1 lacks the exact landmark at 15 but has later observations. It must
        # remain in fixed-patient cells with feature/outcome eligibility false.
        r1 = canonical_series("A", [5, 10, 16, 20, 25, 30, 35, 40], [70, 60, 70, 70, 70, 70, 70, 70])
        t0 = pd.DataFrame({"case_id": ["A"], "t0_min": [10.0]})

        cells, _ = build_observation_2x2(cases, h5, r1, t0)
        r1_feature_rows = cells[cells["feature_process"].eq("R1")]
        r1_outcome_rows = cells[cells["outcome_process"].eq("R1")]
        self.assertEqual(len(r1_feature_rows), 2)
        self.assertTrue(r1_feature_rows["feature_evaluable"].eq(0).all())
        self.assertTrue(r1_feature_rows["t0_map"].eq(60.0).all())
        self.assertTrue(r1_feature_rows["early_auc65_0_5_mmhg_min"].isna().all())
        self.assertTrue(r1_outcome_rows["outcome_evaluable"].eq(0).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)

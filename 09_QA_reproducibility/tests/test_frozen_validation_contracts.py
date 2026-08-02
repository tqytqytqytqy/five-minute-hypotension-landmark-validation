"""Hard-failure tests for the frozen bundle and U0 validation primitives.

These tests use only synthetic data and temporary files.  In particular, they
must never open a MOVER cohort or generate a MOVER prediction.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd


DELIVERY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = DELIVERY_ROOT / "02_code_configs" / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from lm5_validation import frozen, statistics, validation  # noqa: E402


def minimal_bundle_payloads() -> tuple[dict, dict, dict]:
    model = {
        "study_id": "SYNTHETIC_LM5_TEST",
        "primary_model": "common18",
        "models": {
            "common18": {
                "feature_order": ["x"],
                "intercept_raw_scale": 0.0,
                "coefficients_raw_scale": {"x": 1.0},
            }
        },
    }
    preprocess = {
        "study_id": "SYNTHETIC_LM5_TEST",
        "models": {
            "common18": {
                "feature_order": ["x"],
                "imputation_medians": {"x": 0.0},
                "standardization_means": {"x": 0.0},
                "standardization_sds": {"x": 1.0},
            }
        },
    }
    thresholds = {"study_id": "SYNTHETIC_LM5_TEST"}
    return model, preprocess, thresholds


def write_minimal_bundle(
    directory: Path,
    vectors: pd.DataFrame,
    *,
    payloads: tuple[dict, dict, dict] | None = None,
) -> None:
    model, preprocess, thresholds = payloads or minimal_bundle_payloads()
    (directory / "model.json").write_text(json.dumps(model), encoding="utf-8")
    (directory / "preprocess.json").write_text(
        json.dumps(preprocess), encoding="utf-8"
    )
    (directory / "thresholds.json").write_text(
        json.dumps(thresholds), encoding="utf-8"
    )
    vectors.to_csv(directory / "test_vectors.csv", index=False)


class FrozenBundleFailureTests(unittest.TestCase):
    def test_empty_test_vector_table_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            write_minimal_bundle(
                bundle,
                pd.DataFrame(columns=["x", "expected_probability_common18"]),
            )
            with self.assertRaisesRegex(AssertionError, "empty"):
                frozen.verify_test_vectors(bundle)

    def test_nan_expected_probability_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            write_minimal_bundle(
                bundle,
                pd.DataFrame(
                    {"x": [0.0], "expected_probability_common18": [np.nan]}
                ),
            )
            with self.assertRaisesRegex(AssertionError, "non-finite"):
                frozen.verify_test_vectors(bundle)

    def test_nonfinite_model_parameter_is_rejected(self) -> None:
        model, preprocess, thresholds = minimal_bundle_payloads()
        model["models"]["common18"]["intercept_raw_scale"] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            frozen.validate_bundle_contract(model, preprocess, thresholds)

    def test_nonpositive_standardization_sd_is_rejected(self) -> None:
        model, preprocess, thresholds = minimal_bundle_payloads()
        preprocess["models"]["common18"]["standardization_sds"]["x"] = 0.0
        with self.assertRaisesRegex(ValueError, "SD is not positive"):
            frozen.validate_bundle_contract(model, preprocess, thresholds)

    def test_inconsistent_study_id_is_rejected(self) -> None:
        model, preprocess, thresholds = minimal_bundle_payloads()
        thresholds["study_id"] = "WRONG_STUDY"
        with self.assertRaisesRegex(ValueError, "study_id"):
            frozen.validate_bundle_contract(model, preprocess, thresholds)

    def test_sha256_tree_excludes_only_root_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            nested = root / "nested"
            nested.mkdir()
            (root / "data.txt").write_text("immutable input", encoding="utf-8")
            (root / "SHA256SUMS.csv").write_text("root manifest", encoding="utf-8")
            (nested / "SHA256SUMS.csv").write_text(
                "nested source manifest", encoding="utf-8"
            )
            paths = set(frozen.sha256_tree(root)["relative_path"])
            self.assertEqual(paths, {"data.txt", "nested/SHA256SUMS.csv"})

    def test_file_hash_changes_after_input_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.csv"
            path.write_bytes(b"patient_id,feature\nA,1\n")
            before = frozen.sha256_file(path)
            path.write_bytes(b"patient_id,feature\nA,2\n")
            after = frozen.sha256_file(path)
            self.assertNotEqual(before, after)


class StatisticalInputFailureTests(unittest.TestCase):
    def test_nan_probability_and_nonbinary_outcome_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-finite"):
            statistics.brier_score([0, 1], [0.2, np.nan])
        with self.assertRaisesRegex(ValueError, "only 0 and 1"):
            statistics.brier_score([0, 0.5, 1], [0.1, 0.5, 0.9])

    def test_calibration_slope_rejects_single_class_and_constant_risk(self) -> None:
        with self.assertRaisesRegex(ValueError, "both outcome classes"):
            statistics.calibration_intercept_slope(
                np.zeros(8, dtype=int), np.linspace(0.1, 0.8, 8)
            )
        with self.assertRaisesRegex(ValueError, "not identifiable"):
            statistics.calibration_intercept_slope(
                np.array([0, 1] * 4), np.full(8, 0.25)
            )

    def test_patient_bootstrap_enforces_minimum_valid_fraction(self) -> None:
        calls = 0

        def mostly_invalid_metric(y: np.ndarray, p: np.ndarray) -> float:
            nonlocal calls
            calls += 1
            return 0.0 if calls == 1 else np.nan

        with self.assertRaisesRegex(RuntimeError, "valid bootstrap replicates"):
            statistics.paired_patient_bootstrap(
                np.array([0, 1, 0, 1]),
                {"model": np.array([0.1, 0.8, 0.2, 0.7])},
                np.array(["A", "B", "C", "D"]),
                mostly_invalid_metric,
                n_boot=10,
                min_valid_fraction=0.95,
                random_state=3,
            )


class ValidationFailureAndUtilityTests(unittest.TestCase):
    def test_rcs_rejects_nan_and_constant_prediction(self) -> None:
        y = np.array([0, 1] * 10)
        with self.assertRaises(ValueError):
            validation.rcs_calibration_curve(y, np.r_[np.linspace(0.1, 0.9, 19), np.nan])
        with self.assertRaisesRegex(ValueError, "non-constant"):
            validation.rcs_calibration_curve(y, np.full(len(y), 0.3))

    def test_paired_validation_rejects_nonbinary_outcome_before_resampling(self) -> None:
        y = np.array([0.0, 0.5, 1.0, 0.0, 1.0, 0.0])
        p = np.array([0.1, 0.3, 0.8, 0.2, 0.7, 0.4])
        with self.assertRaisesRegex(ValueError, "binary"):
            validation.paired_bootstrap_validation(
                y,
                {"common18": p},
                np.array(["A", "B", "C", "D", "E", "F"]),
                [0.2],
                primary_model="common18",
                n_boot=2,
            )

    def test_two_stage_rejects_nonbinary_outcome_before_resampling(self) -> None:
        with self.assertRaisesRegex(ValueError, "binary"):
            validation.two_stage_strategy_bootstrap(
                np.array([0.0, 0.5, 1.0, 0.0]),
                np.array([True, False, False, False]),
                np.array([np.nan, 0.7, 0.6, 0.1]),
                np.array(["A", "B", "C", "D"]),
                [0.5],
                n_boot=2,
            )

    def test_two_stage_net_benefit_references_and_threshold_routing(self) -> None:
        summary, replicates = validation.two_stage_strategy_bootstrap(
            np.array([1, 0, 1, 0]),
            np.array([True, False, False, False]),
            np.array([np.nan, 0.8, 0.6, 0.1]),
            np.array(["A", "B", "C", "D"]),
            [0.2, 0.5],
            net_benefit_thresholds=[0.5],
            n_boot=8,
            random_state=17,
        )

        # Net-benefit estimands are clinical-threshold-only; workload remains at
        # both capacity/clinical thresholds.
        nb = summary[summary["metric"].str.contains("net_benefit|net_interventions")]
        self.assertEqual(set(nb["threshold"]), {0.5})
        self.assertEqual(set(summary["threshold"]), {0.2, 0.5})

        point = summary[summary["threshold"].eq(0.5)].set_index("metric")["estimate"]
        self.assertAlmostEqual(point["fixed_binary_strategy_net_benefit"], 0.25)
        self.assertAlmostEqual(point["net_benefit_all"], 0.0)
        self.assertAlmostEqual(point["net_benefit_none"], 0.0)
        self.assertAlmostEqual(
            point["net_interventions_avoided_per_100_vs_all"], 25.0
        )
        self.assertTrue(
            replicates.loc[
                replicates["threshold"].eq(0.2),
                "fixed_binary_strategy_net_benefit",
            ].isna().all()
        )

        # Every (threshold, metric) must identify one estimand, never duplicate
        # rows caused by a repeated metric name in the reporting list.
        self.assertFalse(summary.duplicated(["threshold", "metric"]).any())

    def test_paired_validation_stops_below_95_percent_primary_validity(self) -> None:
        # With one event, many non-stratified draws have no event.  The patched
        # RCS isolates the primary-metric valid-replicate gate under test.
        y = np.r_[1, np.zeros(19, dtype=int)]
        p = np.linspace(0.05, 0.95, len(y))
        stable_curve = pd.DataFrame(
            {
                "predicted_probability": [0.1, 0.9],
                "rcs_calibrated_observed_probability": [0.1, 0.9],
            }
        )
        with mock.patch.object(
            validation,
            "rcs_calibration_curve",
            return_value=(stable_curve, np.array([-2.0, -0.5, 0.5, 2.0])),
        ):
            with self.assertRaisesRegex(RuntimeError, "95% valid bootstrap"):
                validation.paired_bootstrap_validation(
                    y,
                    {"common18": p},
                    np.array([f"P{i}" for i in range(len(y))]),
                    [0.2],
                    primary_model="common18",
                    n_boot=40,
                    random_state=11,
                )

    def test_paired_validation_stops_below_95_percent_rcs_validity(self) -> None:
        y = np.array([0, 1] * 20)
        p = np.linspace(0.1, 0.9, len(y))
        stable_curve = pd.DataFrame(
            {
                "predicted_probability": [0.1, 0.9],
                "rcs_calibrated_observed_probability": [0.1, 0.9],
            }
        )
        calls = 0

        def mostly_failed_rcs(*args: object, **kwargs: object) -> tuple[pd.DataFrame, np.ndarray]:
            nonlocal calls
            calls += 1
            if calls == 1:  # full-cohort point curve
                return stable_curve.copy(), np.array([-2.0, -0.5, 0.5, 2.0])
            raise ValueError("synthetic nonidentifiable bootstrap RCS")

        with mock.patch.object(validation, "rcs_calibration_curve", side_effect=mostly_failed_rcs):
            with self.assertRaisesRegex(RuntimeError, "95% valid RCS"):
                validation.paired_bootstrap_validation(
                    y,
                    {"common18": p},
                    np.array([f"P{i}" for i in range(len(y))]),
                    [0.2],
                    primary_model="common18",
                    n_boot=5,
                    random_state=5,
                )


class U0PureFunctionTests(unittest.TestCase):
    @staticmethod
    def load_u0_module():
        path = DELIVERY_ROOT / "02_code_configs" / "scripts" / "06_run_mover_u0_once.py"
        spec = importlib.util.spec_from_file_location("lm5_u0_runner_for_tests", path)
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot load U0 runner")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_atomic_lock_create_is_exclusive_and_replace_is_valid_json(self) -> None:
        u0 = self.load_u0_module()
        with tempfile.TemporaryDirectory() as temporary:
            lock = Path(temporary) / "lock.json"
            u0.atomic_create_json(lock, {"status": "RUNNING", "attempt": 1})
            with self.assertRaises(FileExistsError):
                u0.atomic_create_json(lock, {"status": "RUNNING", "attempt": 2})
            u0.atomic_replace_json(lock, {"status": "FAILED_TECHNICAL", "attempt": 1})
            self.assertEqual(
                json.loads(lock.read_text(encoding="utf-8")),
                {"status": "FAILED_TECHNICAL", "attempt": 1},
            )
            self.assertEqual(list(Path(temporary).glob("*.tmp.*")), [])

    def test_process_liveness_pure_helper(self) -> None:
        u0 = self.load_u0_module()
        self.assertTrue(u0.process_is_running(os.getpid()))
        self.assertFalse(u0.process_is_running(None))
        self.assertFalse(u0.process_is_running(-1))

    def test_locked_thresholds_separate_workload_from_clinical_utility(self) -> None:
        u0 = self.load_u0_module()
        payload = {
            "study_id": u0.CONFIG["study_id"],
            "capacity_thresholds": {
                "top_05": {
                    "risk_threshold": 0.41,
                    "source": "INSPIRE_patient_level_holdout_top_05",
                }
            },
            "clinical_action_primary": 0.20,
            "clinical_action_sensitivity": [0.10, 0.30],
            "legacy_30_feature_thresholds_forbidden": [0.77],
        }
        workload, clinical, table = u0.locked_thresholds(payload)
        self.assertEqual(workload, [0.1, 0.2, 0.3, 0.41])
        self.assertEqual(clinical, [0.1, 0.2, 0.3])
        self.assertEqual(
            set(table.loc[table["threshold"].eq(0.41), "threshold_type"]),
            {"INSPIRE_capacity"},
        )

    def test_qa_contract_rehashes_every_locked_input_and_detects_mutation(self) -> None:
        u0 = self.load_u0_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            locked_input = root / "cohort.csv"
            qa_script = root / "qa.py"
            locked_input.write_bytes(b"patient_id,feature\nA,1\n")
            qa_script.write_text("# synthetic performance-blind QA\n", encoding="utf-8")
            paths = {"synthetic_input": locked_input}
            qa = {
                "study_id": u0.CONFIG["study_id"],
                "gate": "GREEN",
                "performance_blind": True,
                "model_bundle_loaded": False,
                "predictions_generated": False,
                "outcome_event_rate_calculated": False,
                "input_sha256": u0._hash_files(paths),
                "qa_code_sha256": frozen.sha256_file(qa_script),
                "checks": [{"check": "synthetic", "passed": True}],
            }
            with mock.patch.object(u0, "QA_INPUT_PATHS", paths), mock.patch.object(
                u0, "QA_SCRIPT", qa_script
            ):
                self.assertEqual(u0.verify_qa_contract(qa), qa["input_sha256"])
                locked_input.write_bytes(b"patient_id,feature\nA,2\n")
                with self.assertRaisesRegex(RuntimeError, "hashes no longer match"):
                    u0.verify_qa_contract(qa)

    def test_clean_output_gate_rejects_even_a_hidden_file(self) -> None:
        u0 = self.load_u0_module()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "H5_primary"
            output.mkdir()
            with mock.patch.object(u0, "OUTPUT", output):
                u0.require_clean_output()
                (output / ".DS_Store").write_bytes(b"unexpected")
                with self.assertRaisesRegex(RuntimeError, "not empty"):
                    u0.require_clean_output()

    def test_strict_identifiers_outcome_and_prediction_contracts(self) -> None:
        u0 = self.load_u0_module()
        valid = pd.DataFrame(
            {"patient_id": pd.Series(["P1", "P2"], dtype="string"),
             "case_id": pd.Series(["C1", "C2"], dtype="string")}
        )
        u0.strict_ids(valid, "synthetic")
        np.testing.assert_array_equal(
            u0.strict_binary(pd.Series([0, 1]), "outcome"), np.array([0, 1])
        )
        u0.validate_predictions(
            np.array([0, 1]), {"model": np.array([0.2, 0.8])}
        )

        padded = valid.copy()
        padded.loc[1, "patient_id"] = " P2"
        with self.assertRaisesRegex(ValueError, "blank or padded"):
            u0.strict_ids(padded, "synthetic")
        with self.assertRaisesRegex(ValueError, "strictly binary"):
            u0.strict_binary(pd.Series([0.0, 0.5]), "outcome")
        with self.assertRaisesRegex(ValueError, "constant"):
            u0.validate_predictions(
                np.array([0, 1]), {"model": np.array([0.4, 0.4])}
            )

    def test_result_manifest_detects_post_publish_tampering(self) -> None:
        u0 = self.load_u0_module()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            result = output / "result.csv"
            result.write_bytes(b"estimate\n0.25\n")
            pd.DataFrame(
                [
                    {
                        "file": result.name,
                        "size_bytes": result.stat().st_size,
                        "sha256": frozen.sha256_file(result),
                    }
                ]
            ).to_csv(output / u0.RESULT_MANIFEST_NAME, index=False)
            manifest_hash = u0.verify_result_manifest(output, [result.name])
            self.assertEqual(
                manifest_hash, frozen.sha256_file(output / u0.RESULT_MANIFEST_NAME)
            )

            result.write_bytes(b"estimate\n0.99\n")
            with self.assertRaisesRegex(RuntimeError, "hash verification failed"):
                u0.verify_result_manifest(output, [result.name])


if __name__ == "__main__":
    unittest.main(verbosity=2)

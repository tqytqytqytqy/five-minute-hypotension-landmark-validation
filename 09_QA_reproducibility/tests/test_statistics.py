"""Numerical and reproducibility tests for the dependency-light statistics engine."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


DELIVERY_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = (
    DELIVERY_ROOT
    / "02_code_configs"
    / "src"
    / "lm5_validation"
    / "statistics.py"
)
SPEC = importlib.util.spec_from_file_location("lm5_validation.statistics", MODULE_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import guard
    raise RuntimeError("cannot load statistics module")
statistics = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = statistics
SPEC.loader.exec_module(statistics)


class RidgeLogisticTests(unittest.TestCase):
    def test_intercept_is_unpenalized_and_predictions_are_stable(self) -> None:
        rng = np.random.default_rng(13)
        X = rng.normal(size=(600, 3))
        y = np.zeros(600, dtype=int)
        y[:180] = 1
        rng.shuffle(y)
        model = statistics.fit_ridge_logistic(X, y, l2=1e6, standardize=True)
        probability = model.predict_proba(X)

        self.assertTrue(model.converged_, model.message_)
        self.assertTrue(np.all(np.isfinite(probability)))
        self.assertTrue(np.all((probability > 0.0) & (probability < 1.0)))
        self.assertAlmostEqual(float(np.mean(probability)), float(np.mean(y)), places=8)
        self.assertLess(float(np.linalg.norm(model.standardized_coef_)), 1e-5)
        self.assertTrue(np.all(np.isfinite(model.covariance_)))

    def test_standardization_keeps_predictions_invariant_to_units(self) -> None:
        rng = np.random.default_rng(2026)
        X = rng.normal(size=(800, 2))
        probability = 1.0 / (1.0 + np.exp(-(-0.4 + X @ np.array([0.8, -0.5]))))
        y = rng.binomial(1, probability)
        model_a = statistics.fit_ridge_logistic(X, y, l2=0.03, standardize=True)
        model_b = statistics.fit_ridge_logistic(
            X * np.array([1000.0, 0.01]), y, l2=0.03, standardize=True
        )

        self.assertTrue(model_a.converged_)
        self.assertTrue(model_b.converged_)
        np.testing.assert_allclose(
            model_a.predict_proba(X),
            model_b.predict_proba(X * np.array([1000.0, 0.01])),
            rtol=1e-9,
            atol=1e-10,
        )

    def test_extreme_linear_predictors_remain_finite(self) -> None:
        X = np.array([[-1e6], [-1e3], [-1.0], [1.0], [1e3], [1e6]])
        y = np.array([0, 0, 0, 1, 1, 1])
        model = statistics.fit_ridge_logistic(X, y, l2=0.5, standardize=True)
        prediction = model.predict_proba(X)
        self.assertTrue(model.converged_, model.message_)
        self.assertTrue(np.all(np.isfinite(prediction)))
        self.assertTrue(np.all((prediction >= 0.0) & (prediction <= 1.0)))
        self.assertLess(prediction[0], prediction[-1])


class GroupedCrossValidationTests(unittest.TestCase):
    def test_grouped_folds_are_disjoint_and_exhaustive(self) -> None:
        groups = np.repeat(np.arange(23), np.arange(1, 24) % 5 + 1)
        folds = statistics.grouped_kfold_indices(
            groups, n_splits=5, random_state=77
        )
        validation_rows = []
        for training, validation in folds:
            self.assertEqual(
                set(groups[training]).intersection(set(groups[validation])), set()
            )
            validation_rows.extend(validation.tolist())
        self.assertEqual(sorted(validation_rows), list(range(groups.size)))

        folds_again = statistics.grouped_kfold_indices(
            groups, n_splits=5, random_state=77
        )
        for first, second in zip(folds, folds_again):
            np.testing.assert_array_equal(first[1], second[1])

    def test_patient_grouped_cv_returns_a_reproducible_final_model(self) -> None:
        rng = np.random.default_rng(44)
        groups = np.repeat(np.arange(60), 4)
        patient_signal = rng.normal(size=60)
        X = np.column_stack(
            [np.repeat(patient_signal, 4), rng.normal(size=groups.size)]
        )
        true_probability = 1.0 / (
            1.0 + np.exp(-(-0.3 + 0.9 * X[:, 0] - 0.4 * X[:, 1]))
        )
        y = rng.binomial(1, true_probability)

        result = statistics.fit_ridge_logistic_cv(
            X,
            y,
            groups,
            [0.0, 0.01, 0.1],
            n_splits=4,
            random_state=101,
        )
        result_again = statistics.fit_ridge_logistic_cv(
            X,
            y,
            groups,
            [0.0, 0.01, 0.1],
            n_splits=4,
            random_state=101,
        )
        self.assertIn(result.best_l2, {0.0, 0.01, 0.1})
        self.assertTrue(result.model.converged_, result.model.message_)
        self.assertTrue(result.patient_grouped)
        self.assertEqual(result.fold_results.shape[0], 12)
        self.assertEqual(result.best_l2, result_again.best_l2)
        pd.testing.assert_frame_equal(result.cv_results, result_again.cv_results)


class PredictionMetricTests(unittest.TestCase):
    def test_discrimination_and_proper_scores_match_hand_calculation(self) -> None:
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.4, 0.35, 0.8])
        self.assertAlmostEqual(statistics.auroc(y, p), 0.75)
        self.assertAlmostEqual(statistics.auprc(y, p), 5.0 / 6.0)
        self.assertAlmostEqual(
            statistics.brier_score(y, p), float(np.mean((y - p) ** 2))
        )
        expected_log_loss = -float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        self.assertAlmostEqual(statistics.binary_log_loss(y, p), expected_log_loss)
        expected_scaled = 1.0 - np.mean((y - p) ** 2) / 0.25
        self.assertAlmostEqual(statistics.scaled_brier_score(y, p), expected_scaled)

    def test_tied_predictions_are_processed_as_one_threshold(self) -> None:
        y = np.array([1, 0])
        p = np.array([0.5, 0.5])
        self.assertAlmostEqual(statistics.auroc(y, p), 0.5)
        self.assertAlmostEqual(statistics.auprc(y, p), 0.5)

    def test_metric_bundle_has_prespecified_columns(self) -> None:
        y = np.array([0, 1, 0, 1])
        p = np.array([0.1, 0.8, 0.3, 0.7])
        metrics = statistics.evaluate_binary_predictions(y, p)
        self.assertEqual(
            list(metrics.index),
            ["auroc", "auprc", "brier", "scaled_brier", "log_loss"],
        )
        self.assertTrue(np.all(np.isfinite(metrics.to_numpy())))


class CalibrationTests(unittest.TestCase):
    @staticmethod
    def perfectly_calibrated_two_level_data() -> tuple[np.ndarray, np.ndarray]:
        low_y = np.r_[np.ones(20), np.zeros(80)]
        high_y = np.r_[np.ones(80), np.zeros(20)]
        y = np.r_[low_y, high_y]
        p = np.r_[np.full(100, 0.2), np.full(100, 0.8)]
        return y, p

    def test_citl_and_calibration_slope_recover_ideal_values(self) -> None:
        y, p = self.perfectly_calibrated_two_level_data()
        citl = statistics.calibration_in_the_large(y, p)
        slope = statistics.calibration_intercept_slope(y, p)

        self.assertTrue(citl.converged)
        self.assertAlmostEqual(citl.estimate, 0.0, places=8)
        self.assertTrue(slope.converged, slope.message)
        self.assertAlmostEqual(slope.intercept, 0.0, places=7)
        self.assertAlmostEqual(slope.slope, 1.0, places=7)
        self.assertLess(slope.intercept_ci_lower, 0.0)
        self.assertGreater(slope.intercept_ci_upper, 0.0)
        self.assertLess(slope.slope_ci_lower, 1.0)
        self.assertGreater(slope.slope_ci_upper, 1.0)

    def test_oe_log_ci_and_equal_frequency_ici(self) -> None:
        y, p = self.perfectly_calibrated_two_level_data()
        oe = statistics.oe_ratio_log_ci(y, p)
        ici = statistics.ici_equal_frequency(y, p, n_bins=2)

        self.assertAlmostEqual(oe.ratio, 1.0)
        self.assertAlmostEqual(oe.log_ratio, 0.0)
        self.assertLess(oe.ci_lower, 1.0)
        self.assertGreater(oe.ci_upper, 1.0)
        self.assertAlmostEqual(ici.ici, 0.0, places=12)
        self.assertEqual(ici.effective_bins, 2)
        self.assertIn("equal_frequency", ici.method)

    def test_zero_event_oe_has_finite_one_sided_upper_bound(self) -> None:
        result = statistics.oe_ratio_log_ci(
            np.zeros(20), np.full(20, 0.1)
        )
        self.assertEqual(result.ratio, 0.0)
        self.assertTrue(np.isneginf(result.log_ratio))
        self.assertEqual(result.ci_lower, 0.0)
        self.assertTrue(np.isfinite(result.ci_upper))
        self.assertGreater(result.ci_upper, 0.0)


class ClinicalUtilityTests(unittest.TestCase):
    def test_decision_curve_matches_manual_net_benefit(self) -> None:
        y = np.array([1, 0, 1, 0])
        p = np.array([0.9, 0.8, 0.2, 0.1])
        curve = statistics.decision_curve(y, p, [0.5])
        row = curve.iloc[0]
        self.assertAlmostEqual(row["net_benefit_model"], 0.0)
        self.assertAlmostEqual(row["net_benefit_all"], 0.0)
        self.assertAlmostEqual(row["net_benefit_none"], 0.0)
        self.assertAlmostEqual(row["true_positive_weight"], 1.0)
        self.assertAlmostEqual(row["false_positive_weight"], 1.0)

    def test_fixed_threshold_workload_reports_alerts_per_hit(self) -> None:
        y = np.array([1, 0, 1, 0])
        p = np.array([0.9, 0.8, 0.2, 0.1])
        patients = np.array(["A", "B", "C", "D"])
        workload = statistics.fixed_threshold_workload(
            y, p, [0.5], patient_ids=patients
        )
        row = workload.iloc[0]
        self.assertEqual(row["n_alerts"], 2)
        self.assertAlmostEqual(row["alerts_per_true_positive"], 2.0)
        self.assertAlmostEqual(row["false_alerts_per_true_positive"], 1.0)
        self.assertEqual(row["alerted_patients"], 2)
        self.assertEqual(row["captured_event_patients"], 1)


class PatientBootstrapTests(unittest.TestCase):
    def test_bootstrap_is_patient_clustered_paired_and_reproducible(self) -> None:
        patients = np.repeat(np.arange(12), 3)
        y = np.tile(np.array([0, 1, 0]), 12)
        p_a = np.clip(0.15 + 0.65 * y, 0.01, 0.99)
        p_b = p_a.copy()

        result = statistics.paired_patient_bootstrap(
            y,
            {"H5": p_a, "R1": p_b},
            patients,
            statistics.brier_score,
            n_boot=120,
            random_state=919,
            comparisons=[("R1", "H5")],
        )
        result_again = statistics.paired_patient_bootstrap(
            y,
            {"H5": p_a, "R1": p_b},
            patients,
            statistics.brier_score,
            n_boot=120,
            random_state=919,
            comparisons=[("R1", "H5")],
        )

        self.assertEqual(result.n_patients, 12)
        self.assertFalse(result.stratified)
        self.assertEqual(result.sampling_unit, "patient_cluster")
        np.testing.assert_allclose(
            result.replicates["R1-minus-H5"].to_numpy(), 0.0
        )
        pd.testing.assert_frame_equal(result.replicates, result_again.replicates)
        pd.testing.assert_frame_equal(result.summary, result_again.summary)

    def test_invalid_bootstrap_metric_is_rejected(self) -> None:
        patients = np.repeat(np.arange(4), 2)
        y = np.tile([0, 1], 4)
        p = np.full(y.size, 0.5)
        with self.assertRaises(TypeError):
            statistics.paired_patient_bootstrap(
                y,
                {"model": p},
                patients,
                lambda outcome, prediction: np.array([0.0, 1.0]),
                n_boot=3,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)

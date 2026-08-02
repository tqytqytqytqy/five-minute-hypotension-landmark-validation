"""Reproducible statistical primitives for the LM5 external validation study.

The module deliberately depends only on NumPy and pandas.  Definitions that can
vary between packages are made explicit:

* ridge logistic regression minimises weighted *mean* log loss plus
  ``0.5 * l2 * ||beta||^2``; the intercept is never penalised;
* AUPRC is the non-interpolated, stepwise average precision, with tied scores
  processed as one threshold;
* CITL is the intercept from a logistic recalibration model with the original
  prediction logit entered as an offset;
* ICI uses an equal-frequency, piecewise-constant calibration smoother;
* bootstrap resampling is non-stratified and performed on whole patients.

All public routines are deterministic when supplied the same ``random_state``.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


_EPS = np.finfo(float).eps
_PROB_EPS = 1e-12


@dataclass(frozen=True)
class RidgeLogisticResult:
    """Fitted ridge logistic model.

    ``coef_`` and ``intercept_`` are on the original feature scale.  Ridge is
    applied after weighted standardisation when ``standardize=True``; the
    corresponding penalised coefficients are retained in ``standardized_coef_``.
    ``covariance_`` follows the order ``[intercept, coefficients...]`` (or only
    coefficients when ``fit_intercept=False``) and is also on the original scale.
    """

    coef_: np.ndarray
    intercept_: float
    standardized_coef_: np.ndarray
    standardized_intercept_: float
    feature_mean_: np.ndarray
    feature_scale_: np.ndarray
    l2: float
    fit_intercept: bool
    standardize: bool
    converged_: bool
    n_iter_: int
    objective_: float
    gradient_norm_: float
    covariance_: np.ndarray
    solver_: str
    message_: str

    def decision_function(self, X: Any) -> np.ndarray:
        X_arr = _as_2d_float(X, "X")
        if X_arr.shape[1] != self.coef_.size:
            raise ValueError(
                "X has %d columns; fitted model expects %d"
                % (X_arr.shape[1], self.coef_.size)
            )
        return self.intercept_ + X_arr @ self.coef_

    def predict_proba(self, X: Any) -> np.ndarray:
        return _sigmoid(self.decision_function(X))


@dataclass(frozen=True)
class RidgeLogisticCVResult:
    model: RidgeLogisticResult
    best_l2: float
    scoring: str
    cv_results: pd.DataFrame
    fold_results: pd.DataFrame
    n_splits: int
    random_state: int
    patient_grouped: bool = True


@dataclass(frozen=True)
class CalibrationInLargeResult:
    estimate: float
    standard_error: float
    ci_lower: float
    ci_upper: float
    converged: bool
    n_iter: int
    observed_rate: float
    mean_predicted: float
    method: str = "logistic_offset_intercept"


@dataclass(frozen=True)
class CalibrationSlopeResult:
    intercept: float
    slope: float
    intercept_se: float
    slope_se: float
    intercept_ci_lower: float
    intercept_ci_upper: float
    slope_ci_lower: float
    slope_ci_upper: float
    covariance: np.ndarray
    converged: bool
    n_iter: int
    message: str
    method: str = "logistic_recalibration_on_prediction_logit"


@dataclass(frozen=True)
class OERatioResult:
    ratio: float
    log_ratio: float
    standard_error_log: float
    ci_lower: float
    ci_upper: float
    observed: float
    expected: float
    method: str


@dataclass(frozen=True)
class ICIResult:
    ici: float
    method: str
    requested_bins: int
    effective_bins: int
    bins: pd.DataFrame


@dataclass(frozen=True)
class PatientBootstrapResult:
    """Paired patient-level bootstrap output.

    ``summary`` contains model statistics and requested paired contrasts.
    ``replicates`` uses exactly the same patient draw for every prediction
    column, which is what makes contrasts paired.
    """

    summary: pd.DataFrame
    replicates: pd.DataFrame
    point_estimates: pd.Series
    n_boot: int
    n_patients: int
    alpha: float
    random_state: int
    sampling_unit: str = "patient_cluster"
    stratified: bool = False


def _as_1d_float(values: Any, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        arr = np.ravel(arr)
    if arr.size == 0:
        raise ValueError("%s must not be empty" % name)
    if not np.all(np.isfinite(arr)):
        raise ValueError("%s contains missing or non-finite values" % name)
    return arr


def _as_2d_float(values: Any, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError("%s must be a non-empty two-dimensional array" % name)
    if not np.all(np.isfinite(arr)):
        raise ValueError("%s contains missing or non-finite values" % name)
    return arr


def _validate_y(y: Any, n: Optional[int] = None) -> np.ndarray:
    arr = _as_1d_float(y, "y")
    if n is not None and arr.size != n:
        raise ValueError("y and predictions/features must have equal length")
    if not np.all((arr == 0.0) | (arr == 1.0)):
        raise ValueError("y must contain only 0 and 1")
    return arr


def _validate_weights(sample_weight: Any, n: int) -> np.ndarray:
    if sample_weight is None:
        return np.ones(n, dtype=float)
    weight = _as_1d_float(sample_weight, "sample_weight")
    if weight.size != n:
        raise ValueError("sample_weight has the wrong length")
    if np.any(weight < 0.0) or not np.any(weight > 0.0):
        raise ValueError("sample_weight must be non-negative with positive total")
    return weight


def _validate_probabilities(probability: Any, n: Optional[int] = None) -> np.ndarray:
    p = _as_1d_float(probability, "probability")
    if n is not None and p.size != n:
        raise ValueError("y and probability must have equal length")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("probabilities must lie in [0, 1]")
    return p


def _safe_logit(probability: np.ndarray, eps: float = _PROB_EPS) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=float), eps, 1.0 - eps)
    return np.log(p) - np.log1p(-p)


def _sigmoid(linear_predictor: Any) -> np.ndarray:
    z = np.asarray(linear_predictor, dtype=float)
    out = np.empty_like(z, dtype=float)
    nonnegative = z >= 0.0
    out[nonnegative] = 1.0 / (1.0 + np.exp(-z[nonnegative]))
    exp_z = np.exp(z[~nonnegative])
    out[~nonnegative] = exp_z / (1.0 + exp_z)
    return out


def _z_value(alpha: float) -> float:
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be between 0 and 1")
    return float(NormalDist().inv_cdf(1.0 - alpha / 2.0))


def fit_ridge_logistic(
    X: Any,
    y: Any,
    l2: float = 0.0,
    sample_weight: Any = None,
    *,
    fit_intercept: bool = True,
    standardize: bool = True,
    max_iter: int = 100,
    tol: float = 1e-8,
    initial: Any = None,
    min_step: float = 2.0 ** -24,
) -> RidgeLogisticResult:
    """Fit ridge logistic regression by damped Newton/IRLS.

    The optimised objective is weighted mean negative log likelihood plus
    ``0.5*l2*sum(beta**2)``.  The intercept has a zero entry in the penalty
    matrix.  Each Newton step is protected by an Armijo backtracking line
    search; singular Hessians fall back to a Moore-Penrose solution.
    """

    X_raw = _as_2d_float(X, "X")
    y_arr = _validate_y(y, X_raw.shape[0])
    weight = _validate_weights(sample_weight, X_raw.shape[0])
    if l2 < 0.0 or not np.isfinite(l2):
        raise ValueError("l2 must be finite and non-negative")
    if max_iter < 1 or tol <= 0.0 or min_step <= 0.0:
        raise ValueError("max_iter, tol, and min_step must be positive")

    total_weight = float(np.sum(weight))
    if standardize:
        mean = np.sum(X_raw * weight[:, None], axis=0) / total_weight
        centered = X_raw - mean
        variance = np.sum(weight[:, None] * centered * centered, axis=0) / total_weight
        scale = np.sqrt(np.maximum(variance, 0.0))
        scale = np.where(scale > 1e-12, scale, 1.0)
        X_work = centered / scale
    else:
        mean = np.zeros(X_raw.shape[1], dtype=float)
        scale = np.ones(X_raw.shape[1], dtype=float)
        X_work = X_raw.copy()

    if fit_intercept:
        design = np.column_stack([np.ones(X_work.shape[0]), X_work])
        penalty_mask = np.r_[0.0, np.ones(X_work.shape[1])]
    else:
        design = X_work
        penalty_mask = np.ones(X_work.shape[1])

    if initial is None:
        theta = np.zeros(design.shape[1], dtype=float)
        if fit_intercept:
            prevalence = float(np.sum(weight * y_arr) / total_weight)
            theta[0] = float(_safe_logit(np.array([prevalence]))[0])
    else:
        theta = _as_1d_float(initial, "initial").copy()
        if theta.size != design.shape[1]:
            raise ValueError("initial has the wrong length")

    def state(current: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        linear = design @ current
        probability = _sigmoid(linear)
        data_loss = np.sum(weight * (np.logaddexp(0.0, linear) - y_arr * linear)) / total_weight
        objective = float(data_loss + 0.5 * l2 * np.sum(penalty_mask * current * current))
        residual = weight * (probability - y_arr) / total_weight
        gradient = design.T @ residual + l2 * penalty_mask * current
        curvature = weight * probability * (1.0 - probability) / total_weight
        hessian = design.T @ (design * curvature[:, None])
        hessian.flat[:: hessian.shape[0] + 1] += l2 * penalty_mask
        return objective, gradient, hessian, probability

    converged = False
    message = "maximum iterations reached"
    solver_used = "newton_irls_solve"
    objective, gradient, hessian, _ = state(theta)
    n_iter = 0
    for iteration in range(1, max_iter + 1):
        n_iter = iteration
        gradient_norm = float(np.max(np.abs(gradient)))
        if gradient_norm <= tol:
            converged = True
            message = "gradient tolerance reached"
            break

        # The tiny diagonal is numerical damping, not statistical penalisation.
        damping = max(1e-12, 1e-10 * float(np.max(np.diag(hessian))))
        try:
            direction = np.linalg.solve(hessian + damping * np.eye(hessian.shape[0]), gradient)
        except np.linalg.LinAlgError:
            direction = np.linalg.pinv(hessian, rcond=1e-12) @ gradient
            solver_used = "newton_irls_pinv"
        directional_derivative = float(gradient @ direction)
        if (not np.all(np.isfinite(direction))) or directional_derivative <= 0.0:
            direction = gradient.copy()
            directional_derivative = float(gradient @ gradient)
            solver_used = "gradient_fallback"

        step = 1.0
        accepted = False
        while step >= min_step:
            candidate = theta - step * direction
            candidate_state = state(candidate)
            if np.isfinite(candidate_state[0]) and candidate_state[0] <= (
                objective - 1e-4 * step * directional_derivative
            ):
                accepted = True
                break
            step *= 0.5
        if not accepted:
            message = "line search failed to find a decreasing finite step"
            break

        parameter_change = float(np.max(np.abs(candidate - theta)))
        theta = candidate
        objective, gradient, hessian, _ = candidate_state
        if parameter_change <= tol * (1.0 + float(np.max(np.abs(theta)))):
            converged = True
            message = "parameter tolerance reached"
            break

    objective, gradient, hessian, probability = state(theta)
    gradient_norm = float(np.max(np.abs(gradient)))
    if not converged and gradient_norm <= tol:
        converged = True
        message = "gradient tolerance reached"

    # Model-based covariance of the penalised estimate, on the working scale.
    curvature_sum = weight * probability * (1.0 - probability)
    information = design.T @ (design * curvature_sum[:, None])
    information.flat[:: information.shape[0] + 1] += l2 * total_weight * penalty_mask
    covariance_work = np.linalg.pinv(information, rcond=1e-12)

    if fit_intercept:
        intercept_work = float(theta[0])
        coef_work = theta[1:].copy()
        coef_original = coef_work / scale
        intercept_original = float(intercept_work - mean @ coef_original)
        transform = np.zeros((theta.size, theta.size), dtype=float)
        transform[0, 0] = 1.0
        transform[0, 1:] = -mean / scale
        transform[1:, 1:] = np.diag(1.0 / scale)
        covariance_original = transform @ covariance_work @ transform.T
    else:
        intercept_work = 0.0
        coef_work = theta.copy()
        coef_original = coef_work / scale
        intercept_original = 0.0
        transform = np.diag(1.0 / scale)
        covariance_original = transform @ covariance_work @ transform.T

    return RidgeLogisticResult(
        coef_=coef_original,
        intercept_=intercept_original,
        standardized_coef_=coef_work,
        standardized_intercept_=intercept_work,
        feature_mean_=mean,
        feature_scale_=scale,
        l2=float(l2),
        fit_intercept=fit_intercept,
        standardize=standardize,
        converged_=converged,
        n_iter_=n_iter,
        objective_=float(objective),
        gradient_norm_=gradient_norm,
        covariance_=covariance_original,
        solver_=solver_used,
        message_=message,
    )


def predict_logit(model: RidgeLogisticResult, X: Any) -> np.ndarray:
    return model.decision_function(X)


def predict_proba(model: RidgeLogisticResult, X: Any) -> np.ndarray:
    return model.predict_proba(X)


def grouped_kfold_indices(
    groups: Any,
    n_splits: int = 5,
    *,
    shuffle: bool = True,
    random_state: int = 20260712,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create leakage-safe folds while balancing row counts across folds.

    A group (patient) is assigned to exactly one validation fold.  Assignment
    is not outcome-stratified; large groups are greedily placed in the fold
    having the fewest rows, with seeded random tie-breaking.
    """

    group_arr = np.asarray(groups)
    if group_arr.ndim != 1:
        group_arr = np.ravel(group_arr)
    if group_arr.size == 0 or pd.isna(group_arr).any():
        raise ValueError("groups must be non-empty and contain no missing values")
    codes, unique_groups = pd.factorize(group_arr, sort=False)
    n_groups = len(unique_groups)
    if not (2 <= n_splits <= n_groups):
        raise ValueError("n_splits must be between 2 and the number of groups")
    counts = np.bincount(codes, minlength=n_groups)
    rng = np.random.default_rng(random_state)
    tie_breaker = rng.random(n_groups) if shuffle else np.arange(n_groups, dtype=float)
    order = np.lexsort((tie_breaker, -counts))
    fold_load = np.zeros(n_splits, dtype=int)
    fold_group_count = np.zeros(n_splits, dtype=int)
    assignment = np.empty(n_groups, dtype=int)
    for group_code in order:
        minimum = np.min(fold_load)
        candidates = np.flatnonzero(fold_load == minimum)
        if candidates.size > 1:
            min_groups = np.min(fold_group_count[candidates])
            candidates = candidates[fold_group_count[candidates] == min_groups]
        fold = int(rng.choice(candidates)) if shuffle and candidates.size > 1 else int(candidates[0])
        assignment[group_code] = fold
        fold_load[fold] += int(counts[group_code])
        fold_group_count[fold] += 1

    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    all_indices = np.arange(group_arr.size)
    for fold in range(n_splits):
        validation = all_indices[assignment[codes] == fold]
        training = all_indices[assignment[codes] != fold]
        folds.append((training, validation))
    return folds


def fit_ridge_logistic_cv(
    X: Any,
    y: Any,
    groups: Any,
    l2_grid: Sequence[float],
    *,
    n_splits: int = 5,
    scoring: str = "log_loss",
    sample_weight: Any = None,
    fit_intercept: bool = True,
    standardize: bool = True,
    max_iter: int = 100,
    tol: float = 1e-8,
    random_state: int = 20260712,
) -> RidgeLogisticCVResult:
    """Select ridge strength using patient-grouped cross-validation."""

    X_arr = _as_2d_float(X, "X")
    y_arr = _validate_y(y, X_arr.shape[0])
    weight = _validate_weights(sample_weight, X_arr.shape[0])
    group_arr = np.asarray(groups)
    if group_arr.ndim != 1:
        group_arr = np.ravel(group_arr)
    if group_arr.size != X_arr.shape[0]:
        raise ValueError("groups has the wrong length")
    grid = np.asarray(list(l2_grid), dtype=float)
    if grid.size == 0 or not np.all(np.isfinite(grid)) or np.any(grid < 0.0):
        raise ValueError("l2_grid must contain finite non-negative values")
    grid = np.unique(grid)
    metric_functions: Dict[str, Tuple[Callable[..., float], bool]] = {
        "log_loss": (binary_log_loss, False),
        "brier": (brier_score, False),
        "auroc": (auroc, True),
        "auprc": (auprc, True),
    }
    if scoring not in metric_functions:
        raise ValueError("scoring must be one of %s" % sorted(metric_functions))
    metric, higher_is_better = metric_functions[scoring]
    folds = grouped_kfold_indices(
        group_arr, n_splits=n_splits, shuffle=True, random_state=random_state
    )

    fold_rows: List[Dict[str, Any]] = []
    for l2 in grid:
        for fold_number, (training, validation) in enumerate(folds):
            fitted = fit_ridge_logistic(
                X_arr[training],
                y_arr[training],
                l2=float(l2),
                sample_weight=weight[training],
                fit_intercept=fit_intercept,
                standardize=standardize,
                max_iter=max_iter,
                tol=tol,
            )
            probability = fitted.predict_proba(X_arr[validation])
            value = metric(y_arr[validation], probability, sample_weight=weight[validation])
            fold_rows.append(
                {
                    "l2": float(l2),
                    "fold": int(fold_number),
                    "score": float(value),
                    "converged": bool(fitted.converged_),
                    "n_iter": int(fitted.n_iter_),
                    "n_training_rows": int(training.size),
                    "n_validation_rows": int(validation.size),
                    "n_training_patients": int(pd.unique(group_arr[training]).size),
                    "n_validation_patients": int(pd.unique(group_arr[validation]).size),
                }
            )
    fold_results = pd.DataFrame(fold_rows)
    aggregate_rows = []
    for l2, frame in fold_results.groupby("l2", sort=True):
        finite = np.isfinite(frame["score"].to_numpy(dtype=float))
        finite_scores = frame.loc[finite, "score"].to_numpy(dtype=float)
        aggregate_rows.append(
            {
                "l2": float(l2),
                "mean_score": float(np.mean(finite_scores)) if finite_scores.size else np.nan,
                "sd_score": float(np.std(finite_scores, ddof=1)) if finite_scores.size > 1 else 0.0,
                "valid_folds": int(finite_scores.size),
                "converged_folds": int(frame["converged"].sum()),
            }
        )
    cv_results = pd.DataFrame(aggregate_rows).sort_values("l2").reset_index(drop=True)
    valid_rows = cv_results[np.isfinite(cv_results["mean_score"])]
    if valid_rows.empty:
        raise RuntimeError("no cross-validation fold produced a finite score")
    target = valid_rows["mean_score"].max() if higher_is_better else valid_rows["mean_score"].min()
    tied = valid_rows[np.isclose(valid_rows["mean_score"], target, rtol=1e-12, atol=1e-15)]
    best_l2 = float(tied["l2"].min())
    cv_results["selected"] = cv_results["l2"] == best_l2

    final_model = fit_ridge_logistic(
        X_arr,
        y_arr,
        l2=best_l2,
        sample_weight=weight,
        fit_intercept=fit_intercept,
        standardize=standardize,
        max_iter=max_iter,
        tol=tol,
    )
    return RidgeLogisticCVResult(
        model=final_model,
        best_l2=best_l2,
        scoring=scoring,
        cv_results=cv_results,
        fold_results=fold_results,
        n_splits=n_splits,
        random_state=random_state,
    )


def auroc(y: Any, probability: Any, sample_weight: Any = None) -> float:
    """Weighted empirical AUROC with exact handling of tied scores."""

    y_arr = _validate_y(y)
    p = _validate_probabilities(probability, y_arr.size)
    weight = _validate_weights(sample_weight, y_arr.size)
    positive_weight = float(np.sum(weight[y_arr == 1.0]))
    negative_weight = float(np.sum(weight[y_arr == 0.0]))
    if positive_weight <= 0.0 or negative_weight <= 0.0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    p_sorted, y_sorted, w_sorted = p[order], y_arr[order], weight[order]
    concordance = 0.0
    negative_below = 0.0
    start = 0
    while start < p_sorted.size:
        end = start + 1
        while end < p_sorted.size and p_sorted[end] == p_sorted[start]:
            end += 1
        group_positive = float(np.sum(w_sorted[start:end] * y_sorted[start:end]))
        group_negative = float(np.sum(w_sorted[start:end] * (1.0 - y_sorted[start:end])))
        concordance += group_positive * (negative_below + 0.5 * group_negative)
        negative_below += group_negative
        start = end
    return float(concordance / (positive_weight * negative_weight))


def auprc(y: Any, probability: Any, sample_weight: Any = None) -> float:
    """Stepwise AUPRC (average precision), grouping tied scores together."""

    y_arr = _validate_y(y)
    p = _validate_probabilities(probability, y_arr.size)
    weight = _validate_weights(sample_weight, y_arr.size)
    total_positive = float(np.sum(weight * y_arr))
    if total_positive <= 0.0:
        return float("nan")
    order = np.argsort(-p, kind="mergesort")
    p_sorted, y_sorted, w_sorted = p[order], y_arr[order], weight[order]
    true_positive = 0.0
    false_positive = 0.0
    previous_recall = 0.0
    area = 0.0
    start = 0
    while start < p_sorted.size:
        end = start + 1
        while end < p_sorted.size and p_sorted[end] == p_sorted[start]:
            end += 1
        true_positive += float(np.sum(w_sorted[start:end] * y_sorted[start:end]))
        false_positive += float(np.sum(w_sorted[start:end] * (1.0 - y_sorted[start:end])))
        recall = true_positive / total_positive
        precision = true_positive / (true_positive + false_positive)
        area += (recall - previous_recall) * precision
        previous_recall = recall
        start = end
    return float(area)


def brier_score(y: Any, probability: Any, sample_weight: Any = None) -> float:
    y_arr = _validate_y(y)
    p = _validate_probabilities(probability, y_arr.size)
    weight = _validate_weights(sample_weight, y_arr.size)
    return float(np.average((y_arr - p) ** 2, weights=weight))


def scaled_brier_score(y: Any, probability: Any, sample_weight: Any = None) -> float:
    """Brier skill score relative to a constant observed-prevalence model."""

    y_arr = _validate_y(y)
    p = _validate_probabilities(probability, y_arr.size)
    weight = _validate_weights(sample_weight, y_arr.size)
    prevalence = float(np.average(y_arr, weights=weight))
    null_brier = prevalence * (1.0 - prevalence)
    if null_brier <= 0.0:
        return float("nan")
    return float(1.0 - brier_score(y_arr, p, weight) / null_brier)


def binary_log_loss(y: Any, probability: Any, sample_weight: Any = None) -> float:
    y_arr = _validate_y(y)
    p = _validate_probabilities(probability, y_arr.size)
    weight = _validate_weights(sample_weight, y_arr.size)
    p_safe = np.clip(p, _PROB_EPS, 1.0 - _PROB_EPS)
    loss = -(y_arr * np.log(p_safe) + (1.0 - y_arr) * np.log1p(-p_safe))
    return float(np.average(loss, weights=weight))


def evaluate_binary_predictions(
    y: Any, probability: Any, sample_weight: Any = None
) -> pd.Series:
    """Return the prespecified global predictive-performance metrics."""

    return pd.Series(
        {
            "auroc": auroc(y, probability, sample_weight),
            "auprc": auprc(y, probability, sample_weight),
            "brier": brier_score(y, probability, sample_weight),
            "scaled_brier": scaled_brier_score(y, probability, sample_weight),
            "log_loss": binary_log_loss(y, probability, sample_weight),
        },
        dtype=float,
    )


def calibration_in_the_large(
    y: Any,
    probability: Any,
    sample_weight: Any = None,
    *,
    alpha: float = 0.05,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> CalibrationInLargeResult:
    """Estimate CITL using the original prediction logit as a fixed offset."""

    y_arr = _validate_y(y)
    p = _validate_probabilities(probability, y_arr.size)
    weight = _validate_weights(sample_weight, y_arr.size)
    total_weight = float(np.sum(weight))
    observed_rate = float(np.sum(weight * y_arr) / total_weight)
    mean_predicted = float(np.sum(weight * p) / total_weight)
    z_value = _z_value(alpha)
    if observed_rate == 0.0:
        return CalibrationInLargeResult(
            -np.inf, np.inf, -np.inf, np.inf, False, 0, observed_rate, mean_predicted
        )
    if observed_rate == 1.0:
        return CalibrationInLargeResult(
            np.inf, np.inf, -np.inf, np.inf, False, 0, observed_rate, mean_predicted
        )

    offset = _safe_logit(p)

    def score(intercept: float) -> float:
        return float(np.sum(weight * (_sigmoid(offset + intercept) - y_arr)) / total_weight)

    lower, upper = -60.0, 60.0
    if score(lower) > 0.0 or score(upper) < 0.0:
        raise RuntimeError("failed to bracket the CITL root")
    estimate = 0.0
    converged = False
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        estimate = 0.5 * (lower + upper)
        value = score(estimate)
        if abs(value) <= tol or (upper - lower) <= tol:
            converged = True
            break
        if value > 0.0:
            upper = estimate
        else:
            lower = estimate
    calibrated = _sigmoid(offset + estimate)
    information = float(np.sum(weight * calibrated * (1.0 - calibrated)))
    standard_error = float(np.sqrt(1.0 / information)) if information > 0.0 else np.inf
    return CalibrationInLargeResult(
        estimate=float(estimate),
        standard_error=standard_error,
        ci_lower=float(estimate - z_value * standard_error),
        ci_upper=float(estimate + z_value * standard_error),
        converged=converged,
        n_iter=n_iter,
        observed_rate=observed_rate,
        mean_predicted=mean_predicted,
    )


def calibration_intercept_slope(
    y: Any,
    probability: Any,
    sample_weight: Any = None,
    *,
    alpha: float = 0.05,
    max_iter: int = 200,
    tol: float = 1e-9,
) -> CalibrationSlopeResult:
    """Fit ``logit(P(Y=1)) = intercept + slope*logit(prediction)``."""

    y_arr = _validate_y(y)
    p = _validate_probabilities(probability, y_arr.size)
    weight = _validate_weights(sample_weight, y_arr.size)
    if np.unique(y_arr).size != 2:
        raise ValueError("calibration intercept/slope require both outcome classes")
    prediction_logit = _safe_logit(p)
    weighted_mean = float(np.average(prediction_logit, weights=weight))
    weighted_variance = float(
        np.average((prediction_logit - weighted_mean) ** 2, weights=weight)
    )
    if not np.isfinite(weighted_variance) or weighted_variance <= 1e-12:
        raise ValueError("calibration slope is not identifiable for constant prediction logit")
    fitted = fit_ridge_logistic(
        prediction_logit.reshape(-1, 1),
        y_arr,
        l2=0.0,
        sample_weight=weight,
        fit_intercept=True,
        standardize=False,
        max_iter=max_iter,
        tol=tol,
        initial=np.array([0.0, 1.0]),
    )
    if not fitted.converged_:
        raise RuntimeError(f"calibration model did not converge: {fitted.message_}")
    covariance = fitted.covariance_
    if covariance.shape != (2, 2) or not np.isfinite(covariance).all():
        raise RuntimeError("calibration covariance is non-finite or has invalid rank")
    intercept_se = float(np.sqrt(max(covariance[0, 0], 0.0)))
    slope_se = float(np.sqrt(max(covariance[1, 1], 0.0)))
    if not np.isfinite([fitted.intercept_, fitted.coef_[0], intercept_se, slope_se]).all():
        raise RuntimeError("calibration estimates are non-finite")
    z_value = _z_value(alpha)
    return CalibrationSlopeResult(
        intercept=float(fitted.intercept_),
        slope=float(fitted.coef_[0]),
        intercept_se=intercept_se,
        slope_se=slope_se,
        intercept_ci_lower=float(fitted.intercept_ - z_value * intercept_se),
        intercept_ci_upper=float(fitted.intercept_ + z_value * intercept_se),
        slope_ci_lower=float(fitted.coef_[0] - z_value * slope_se),
        slope_ci_upper=float(fitted.coef_[0] + z_value * slope_se),
        covariance=covariance,
        converged=fitted.converged_,
        n_iter=fitted.n_iter_,
        message=fitted.message_,
    )


def oe_ratio_log_ci(
    y: Any,
    probability: Any,
    sample_weight: Any = None,
    *,
    alpha: float = 0.05,
) -> OERatioResult:
    """Observed/expected ratio with a Poisson log-scale confidence interval.

    Predictions are treated as fixed.  For at least one observed event, the
    conventional approximation ``SE(log(O/E)) = 1/sqrt(O)`` is used.  With
    zero events, the upper confidence limit is the exact zero-count Poisson
    bound divided by E; the log estimate is minus infinity.
    """

    y_arr = _validate_y(y)
    p = _validate_probabilities(probability, y_arr.size)
    weight = _validate_weights(sample_weight, y_arr.size)
    _z_value(alpha)  # validates alpha
    observed = float(np.sum(weight * y_arr))
    expected = float(np.sum(weight * p))
    if expected <= 0.0:
        return OERatioResult(
            ratio=np.nan,
            log_ratio=np.nan,
            standard_error_log=np.inf,
            ci_lower=np.nan,
            ci_upper=np.nan,
            observed=observed,
            expected=expected,
            method="undefined_expected_zero",
        )
    if observed <= 0.0:
        upper_count = -np.log(alpha / 2.0)
        return OERatioResult(
            ratio=0.0,
            log_ratio=-np.inf,
            standard_error_log=np.inf,
            ci_lower=0.0,
            ci_upper=float(upper_count / expected),
            observed=observed,
            expected=expected,
            method="poisson_exact_zero_event_upper",
        )
    ratio = observed / expected
    log_ratio = float(np.log(ratio))
    se_log = float(1.0 / np.sqrt(observed))
    z_value = _z_value(alpha)
    return OERatioResult(
        ratio=float(ratio),
        log_ratio=log_ratio,
        standard_error_log=se_log,
        ci_lower=float(np.exp(log_ratio - z_value * se_log)),
        ci_upper=float(np.exp(log_ratio + z_value * se_log)),
        observed=observed,
        expected=expected,
        method="poisson_log_normal_fixed_expected",
    )


def ici_equal_frequency(
    y: Any,
    probability: Any,
    sample_weight: Any = None,
    *,
    n_bins: int = 10,
) -> ICIResult:
    """Estimate ICI with an equal-frequency piecewise-constant smoother.

    Unweighted prediction quantiles define bins (duplicate cut points collapse),
    while sample weights are honoured in each bin's event rate and in the final
    mean absolute calibration error.  This scalable smoother is deliberately
    labelled and should not be described as LOWESS or kernel ICI.
    """

    y_arr = _validate_y(y)
    p = _validate_probabilities(probability, y_arr.size)
    weight = _validate_weights(sample_weight, y_arr.size)
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(p, quantiles))
    if edges.size == 1:
        bin_id = np.zeros(p.size, dtype=int)
        edges = np.array([edges[0], edges[0]], dtype=float)
    else:
        bin_id = np.searchsorted(edges[1:-1], p, side="right")
    smoothed = np.empty(p.size, dtype=float)
    rows: List[Dict[str, Any]] = []
    for current_bin in np.unique(bin_id):
        mask = bin_id == current_bin
        bin_weight = float(np.sum(weight[mask]))
        observed_rate = float(np.sum(weight[mask] * y_arr[mask]) / bin_weight)
        mean_prediction = float(np.sum(weight[mask] * p[mask]) / bin_weight)
        smoothed[mask] = observed_rate
        rows.append(
            {
                "bin": int(current_bin + 1),
                "lower_probability": float(np.min(p[mask])),
                "upper_probability": float(np.max(p[mask])),
                "n": int(np.sum(mask)),
                "weight": bin_weight,
                "mean_predicted": mean_prediction,
                "observed_rate": observed_rate,
                "absolute_bin_gap": abs(observed_rate - mean_prediction),
            }
        )
    ici = float(np.average(np.abs(smoothed - p), weights=weight))
    bins = pd.DataFrame(rows)
    return ICIResult(
        ici=ici,
        method="equal_frequency_piecewise_constant_unweighted_quantiles",
        requested_bins=int(n_bins),
        effective_bins=int(bins.shape[0]),
        bins=bins,
    )


def decision_curve(
    y: Any,
    probability: Any,
    thresholds: Iterable[float],
    sample_weight: Any = None,
) -> pd.DataFrame:
    """Decision-curve net benefit for model, treat-all, and treat-none."""

    y_arr = _validate_y(y)
    p = _validate_probabilities(probability, y_arr.size)
    weight = _validate_weights(sample_weight, y_arr.size)
    threshold_arr = _as_1d_float(list(thresholds), "thresholds")
    if np.any((threshold_arr <= 0.0) | (threshold_arr >= 1.0)):
        raise ValueError("decision thresholds must be strictly between 0 and 1")
    total_weight = float(np.sum(weight))
    prevalence = float(np.sum(weight * y_arr) / total_weight)
    rows = []
    for threshold in threshold_arr:
        alert = p >= threshold
        true_positive = float(np.sum(weight * alert * y_arr))
        false_positive = float(np.sum(weight * alert * (1.0 - y_arr)))
        odds = float(threshold / (1.0 - threshold))
        net_benefit = true_positive / total_weight - false_positive / total_weight * odds
        net_benefit_all = prevalence - (1.0 - prevalence) * odds
        rows.append(
            {
                "threshold": float(threshold),
                "true_positive_weight": true_positive,
                "false_positive_weight": false_positive,
                "net_benefit_model": float(net_benefit),
                "net_benefit_all": float(net_benefit_all),
                "net_benefit_none": 0.0,
                "standardized_net_benefit_model": (
                    float(net_benefit / prevalence) if prevalence > 0.0 else np.nan
                ),
                "net_interventions_avoided_per_100_vs_all": float(
                    (net_benefit - net_benefit_all) / odds * 100.0
                ),
            }
        )
    return pd.DataFrame(rows)


def fixed_threshold_workload(
    y: Any,
    probability: Any,
    thresholds: Iterable[float],
    *,
    patient_ids: Any = None,
    sample_weight: Any = None,
) -> pd.DataFrame:
    """Clinical alert burden and yield at fixed frozen thresholds.

    Every probability at or above threshold is one landmark alert; no cool-down
    or alert suppression is applied.  ``alerts_per_true_positive`` is therefore
    the prespecified "alerts needed per hit" measure.
    """

    y_arr = _validate_y(y)
    p = _validate_probabilities(probability, y_arr.size)
    weight = _validate_weights(sample_weight, y_arr.size)
    threshold_arr = _as_1d_float(list(thresholds), "thresholds")
    if np.any((threshold_arr < 0.0) | (threshold_arr > 1.0)):
        raise ValueError("thresholds must lie in [0, 1]")
    if patient_ids is not None:
        patient_arr = np.asarray(patient_ids)
        if patient_arr.ndim != 1:
            patient_arr = np.ravel(patient_arr)
        if patient_arr.size != y_arr.size or pd.isna(patient_arr).any():
            raise ValueError("patient_ids must match y and contain no missing values")
        patient_codes, unique_patients = pd.factorize(patient_arr, sort=False)
        n_patients = int(len(unique_patients))
    else:
        patient_codes = None
        n_patients = 0

    total_weight = float(np.sum(weight))
    rows: List[Dict[str, Any]] = []
    for threshold in threshold_arr:
        alert = p >= threshold
        tp = float(np.sum(weight * alert * y_arr))
        fp = float(np.sum(weight * alert * (1.0 - y_arr)))
        fn = float(np.sum(weight * (~alert) * y_arr))
        tn = float(np.sum(weight * (~alert) * (1.0 - y_arr)))
        n_alerts_weighted = tp + fp
        raw_alerts = int(np.sum(alert))
        row: Dict[str, Any] = {
            "threshold": float(threshold),
            "n_landmarks": int(y_arr.size),
            "landmark_weight": total_weight,
            "n_alerts": raw_alerts,
            "weighted_alerts": n_alerts_weighted,
            "alert_rate": n_alerts_weighted / total_weight,
            "alerts_per_100_landmarks": n_alerts_weighted / total_weight * 100.0,
            "true_positive_weight": tp,
            "false_positive_weight": fp,
            "false_negative_weight": fn,
            "true_negative_weight": tn,
            "sensitivity": tp / (tp + fn) if (tp + fn) > 0.0 else np.nan,
            "specificity": tn / (tn + fp) if (tn + fp) > 0.0 else np.nan,
            "positive_predictive_value": tp / (tp + fp) if (tp + fp) > 0.0 else np.nan,
            "negative_predictive_value": tn / (tn + fn) if (tn + fn) > 0.0 else np.nan,
            "alerts_per_true_positive": n_alerts_weighted / tp if tp > 0.0 else np.inf,
            "false_alerts_per_true_positive": fp / tp if tp > 0.0 else np.inf,
        }
        if patient_codes is not None:
            alerted_patients = np.unique(patient_codes[alert])
            event_patient_codes = np.unique(patient_codes[y_arr == 1.0])
            captured_codes = np.unique(patient_codes[alert & (y_arr == 1.0)])
            row.update(
                {
                    "n_patients": n_patients,
                    "alerted_patients": int(alerted_patients.size),
                    "patient_alert_rate": alerted_patients.size / n_patients,
                    "event_patients": int(event_patient_codes.size),
                    "captured_event_patients": int(captured_codes.size),
                    "event_patient_capture_rate": (
                        captured_codes.size / event_patient_codes.size
                        if event_patient_codes.size
                        else np.nan
                    ),
                    "raw_alerts_per_captured_event_patient": (
                        raw_alerts / captured_codes.size if captured_codes.size else np.inf
                    ),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def paired_patient_bootstrap(
    y: Any,
    predictions: Mapping[str, Any],
    patient_ids: Any,
    metric: Callable[..., float],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    random_state: int = 20260712,
    comparisons: Optional[Sequence[Tuple[str, str]]] = None,
    metric_kwargs: Optional[Mapping[str, Any]] = None,
    min_valid_fraction: float = 0.80,
    return_replicates: bool = True,
) -> PatientBootstrapResult:
    """Non-stratified paired bootstrap that resamples whole patients.

    ``metric`` must accept ``metric(y, probability, **metric_kwargs)`` and return
    one scalar.  For a comparison ``("R1", "H5")``, the stored contrast is
    ``metric(R1) - metric(H5)``.  Patients are sampled uniformly with replacement
    without using their outcome, and repeated selection duplicates all of that
    patient's landmarks.  The same draw is used for every model.
    """

    y_arr = _validate_y(y)
    if not isinstance(predictions, Mapping) or len(predictions) == 0:
        raise ValueError("predictions must be a non-empty mapping")
    prediction_arrays: Dict[str, np.ndarray] = {}
    for name, values in predictions.items():
        if not isinstance(name, str) or not name:
            raise ValueError("prediction names must be non-empty strings")
        prediction_arrays[name] = _validate_probabilities(values, y_arr.size)
    patient_arr = np.asarray(patient_ids)
    if patient_arr.ndim != 1:
        patient_arr = np.ravel(patient_arr)
    if patient_arr.size != y_arr.size or pd.isna(patient_arr).any():
        raise ValueError("patient_ids must match y and contain no missing values")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    _z_value(alpha)  # validates alpha
    if not (0.0 < min_valid_fraction <= 1.0):
        raise ValueError("min_valid_fraction must lie in (0, 1]")
    kwargs = dict(metric_kwargs or {})
    codes, unique_patients = pd.factorize(patient_arr, sort=False)
    n_patients = int(len(unique_patients))
    if n_patients < 2:
        raise ValueError("at least two patients are required for patient bootstrap")
    cluster_rows = [np.flatnonzero(codes == code) for code in range(n_patients)]

    if comparisons is None:
        comparisons = []
    for first, second in comparisons:
        if first not in prediction_arrays or second not in prediction_arrays:
            raise ValueError("every comparison name must exist in predictions")

    def evaluate(indices: np.ndarray, name: str, *, allow_estimand_failure: bool = False) -> float:
        try:
            value = metric(y_arr[indices], prediction_arrays[name][indices], **kwargs)
        except (ValueError, RuntimeError, OverflowError, FloatingPointError, np.linalg.LinAlgError):
            if allow_estimand_failure:
                return np.nan
            raise
        arr = np.asarray(value)
        if arr.size != 1:
            raise TypeError("metric must return one scalar")
        return float(arr.reshape(-1)[0])

    all_rows = np.arange(y_arr.size)
    point = {name: evaluate(all_rows, name) for name in prediction_arrays}
    rng = np.random.default_rng(random_state)
    replicate_array = np.full((n_boot, len(prediction_arrays)), np.nan, dtype=float)
    names = list(prediction_arrays)
    for bootstrap_index in range(n_boot):
        selected_clusters = rng.integers(0, n_patients, size=n_patients)
        sampled_rows = np.concatenate([cluster_rows[code] for code in selected_clusters])
        for model_index, name in enumerate(names):
            replicate_array[bootstrap_index, model_index] = evaluate(
                sampled_rows, name, allow_estimand_failure=True
            )
    replicates = pd.DataFrame(replicate_array, columns=names)

    comparison_names: List[str] = []
    for first, second in comparisons:
        contrast_name = "%s-minus-%s" % (first, second)
        if contrast_name in replicates.columns:
            raise ValueError("duplicate model/contrast name: %s" % contrast_name)
        replicates[contrast_name] = replicates[first] - replicates[second]
        point[contrast_name] = point[first] - point[second]
        comparison_names.append(contrast_name)

    summary_rows: List[Dict[str, Any]] = []
    for name in names + comparison_names:
        values = replicates[name].to_numpy(dtype=float)
        valid = values[np.isfinite(values)]
        valid_fraction = valid.size / n_boot
        if valid_fraction < min_valid_fraction:
            raise RuntimeError(
                "%s has only %.1f%% valid bootstrap replicates" % (name, 100.0 * valid_fraction)
            )
        summary_rows.append(
            {
                "estimand": name,
                "kind": "contrast" if name in comparison_names else "model",
                "estimate": float(point[name]),
                "bootstrap_standard_error": (
                    float(np.std(valid, ddof=1)) if valid.size > 1 else np.nan
                ),
                "ci_lower": float(np.quantile(valid, alpha / 2.0)),
                "ci_upper": float(np.quantile(valid, 1.0 - alpha / 2.0)),
                "n_valid": int(valid.size),
                "valid_fraction": float(valid_fraction),
            }
        )
    summary = pd.DataFrame(summary_rows)
    retained_replicates = replicates if return_replicates else pd.DataFrame(index=replicates.index)
    return PatientBootstrapResult(
        summary=summary,
        replicates=retained_replicates,
        point_estimates=pd.Series(point, dtype=float),
        n_boot=int(n_boot),
        n_patients=n_patients,
        alpha=float(alpha),
        random_state=int(random_state),
    )


# Explicit aliases matching common reporting terminology.
average_precision = auprc
log_loss = binary_log_loss
citl_offset = calibration_in_the_large
calibration_slope = calibration_intercept_slope
observed_expected_ratio = oe_ratio_log_ci


__all__ = [
    "RidgeLogisticResult",
    "RidgeLogisticCVResult",
    "CalibrationInLargeResult",
    "CalibrationSlopeResult",
    "OERatioResult",
    "ICIResult",
    "PatientBootstrapResult",
    "fit_ridge_logistic",
    "fit_ridge_logistic_cv",
    "grouped_kfold_indices",
    "predict_logit",
    "predict_proba",
    "auroc",
    "auprc",
    "average_precision",
    "brier_score",
    "scaled_brier_score",
    "binary_log_loss",
    "log_loss",
    "evaluate_binary_predictions",
    "calibration_in_the_large",
    "citl_offset",
    "calibration_intercept_slope",
    "calibration_slope",
    "oe_ratio_log_ci",
    "observed_expected_ratio",
    "ici_equal_frequency",
    "decision_curve",
    "fixed_threshold_workload",
    "paired_patient_bootstrap",
]

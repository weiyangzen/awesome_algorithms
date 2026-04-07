"""Linear regression with regularization (Ridge + Lasso) MVP.

This script is deterministic and requires no interactive input.
It demonstrates why regularization helps under multicollinearity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


HistoryItem = Tuple[int, float, float, int]


@dataclass
class FitResult:
    name: str
    alpha: float
    coef: np.ndarray
    intercept: float
    converged: bool
    epochs_used: int
    history: List[HistoryItem]


def validate_dataset(x: np.ndarray, y: np.ndarray) -> None:
    if x.ndim != 2:
        raise ValueError(f"X must be 2D, got shape={x.shape}.")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D, got shape={y.shape}.")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Sample mismatch: X has {x.shape[0]} rows while y has {y.shape[0]}.")
    if x.shape[0] < 4:
        raise ValueError("Need at least 4 samples.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("X and y must contain only finite numbers.")


def train_test_split(
    x: np.ndarray,
    y: np.ndarray,
    test_ratio: float = 0.25,
    seed: int = 2026,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not (0.0 < test_ratio < 1.0):
        raise ValueError("test_ratio must be in (0, 1).")

    n_samples = x.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)
    n_test = int(round(n_samples * test_ratio))
    n_test = min(max(n_test, 1), n_samples - 1)

    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


def standardize_from_train(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    if np.any(std < 1e-12):
        bad_col = int(np.argmin(std))
        raise ValueError(f"Feature column {bad_col} has near-zero std; cannot standardize reliably.")
    return (x_train - mean) / std, (x_test - mean) / std, mean, std


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def predict(x_std: np.ndarray, coef: np.ndarray, intercept: float) -> np.ndarray:
    return x_std @ coef + intercept


def fit_ols_closed_form(x_std: np.ndarray, y: np.ndarray) -> FitResult:
    y_mean = float(np.mean(y))
    y_centered = y - y_mean

    coef, *_ = np.linalg.lstsq(x_std, y_centered, rcond=None)
    history = [(1, 0.5 * float(np.mean((y_centered - x_std @ coef) ** 2)), 0.0, int(np.count_nonzero(np.abs(coef) > 1e-12)))]

    return FitResult(
        name="OLS",
        alpha=0.0,
        coef=coef,
        intercept=y_mean,
        converged=True,
        epochs_used=1,
        history=history,
    )


def fit_ridge_closed_form(x_std: np.ndarray, y: np.ndarray, alpha: float) -> FitResult:
    if alpha <= 0.0:
        raise ValueError("Ridge alpha must be > 0.")

    y_mean = float(np.mean(y))
    y_centered = y - y_mean
    n_features = x_std.shape[1]

    xtx = x_std.T @ x_std
    rhs = x_std.T @ y_centered
    coef = np.linalg.solve(xtx + alpha * np.eye(n_features), rhs)

    obj = 0.5 * float(np.mean((y_centered - x_std @ coef) ** 2)) + 0.5 * alpha * float(np.dot(coef, coef))
    history = [(1, obj, 0.0, int(np.count_nonzero(np.abs(coef) > 1e-12)))]

    return FitResult(
        name="Ridge",
        alpha=alpha,
        coef=coef,
        intercept=y_mean,
        converged=True,
        epochs_used=1,
        history=history,
    )


def soft_threshold(value: float, lam: float) -> float:
    if value > lam:
        return value - lam
    if value < -lam:
        return value + lam
    return 0.0


def lasso_objective(x_std: np.ndarray, y_centered: np.ndarray, coef: np.ndarray, alpha: float) -> float:
    residual = y_centered - x_std @ coef
    return 0.5 * float(np.mean(residual * residual)) + alpha * float(np.sum(np.abs(coef)))


def fit_lasso_coordinate_descent(
    x_std: np.ndarray,
    y: np.ndarray,
    alpha: float,
    tol: float = 1e-8,
    max_epochs: int = 3000,
) -> FitResult:
    if alpha <= 0.0:
        raise ValueError("Lasso alpha must be > 0.")
    if tol <= 0.0:
        raise ValueError("tol must be > 0.")
    if max_epochs <= 0:
        raise ValueError("max_epochs must be > 0.")

    y_mean = float(np.mean(y))
    y_centered = y - y_mean

    n_samples, n_features = x_std.shape
    z = np.sum(x_std * x_std, axis=0) / n_samples
    if np.any(z <= 1e-15):
        bad_col = int(np.argmin(z))
        raise ValueError(f"Feature column {bad_col} is near-constant after standardization.")

    coef = np.zeros(n_features, dtype=float)
    residual = y_centered.copy()
    history: List[HistoryItem] = []
    converged = False

    for epoch in range(1, max_epochs + 1):
        max_delta = 0.0

        for j in range(n_features):
            xj = x_std[:, j]
            old = coef[j]

            rho_j = float(np.dot(xj, residual + xj * old)) / n_samples
            new = soft_threshold(rho_j, alpha) / z[j]
            delta = new - old

            if delta != 0.0:
                coef[j] = new
                residual -= xj * delta
                abs_delta = abs(delta)
                if abs_delta > max_delta:
                    max_delta = abs_delta

        obj = lasso_objective(x_std=x_std, y_centered=y_centered, coef=coef, alpha=alpha)
        if not np.isfinite(obj):
            raise RuntimeError("Encountered non-finite objective in Lasso optimization.")

        nnz = int(np.count_nonzero(np.abs(coef) > 1e-10))
        history.append((epoch, obj, max_delta, nnz))

        if max_delta < tol:
            converged = True
            break

    return FitResult(
        name="Lasso",
        alpha=alpha,
        coef=coef,
        intercept=y_mean,
        converged=converged,
        epochs_used=len(history),
        history=history,
    )


def objective_monotone_check(history: Sequence[HistoryItem], tol: float = 1e-10) -> Tuple[bool, int]:
    violations = 0
    for i in range(1, len(history)):
        if history[i][1] > history[i - 1][1] + tol:
            violations += 1
    return violations == 0, violations


def summarize_result(
    result: FitResult,
    x_train_std: np.ndarray,
    y_train: np.ndarray,
    x_test_std: np.ndarray,
    y_test: np.ndarray,
    true_coef_std: np.ndarray,
) -> Dict[str, float]:
    train_pred = predict(x_train_std, result.coef, result.intercept)
    test_pred = predict(x_test_std, result.coef, result.intercept)

    train_mse = mse(y_train, train_pred)
    test_mse = mse(y_test, test_pred)
    coef_l2_norm = float(np.linalg.norm(result.coef))
    coef_l2_error = float(np.linalg.norm(result.coef - true_coef_std))
    nnz = float(np.count_nonzero(np.abs(result.coef) > 1e-3))

    monotone_ok = True
    violations = 0
    if len(result.history) >= 2:
        monotone_ok, violations = objective_monotone_check(result.history)

    print(f"\n=== {result.name} (alpha={result.alpha:.4f}) ===")
    print(f"converged: {result.converged}")
    print(f"epochs_used: {result.epochs_used}")
    print(f"train_mse: {train_mse:.8f}")
    print(f"test_mse:  {test_mse:.8f}")
    print(f"coef_l2_norm: {coef_l2_norm:.8f}")
    print(f"coef_l2_error_vs_true: {coef_l2_error:.8f}")
    print(f"nnz(|coef|>1e-3): {int(nnz)}")
    if len(result.history) >= 2:
        print(f"objective_monotone: {monotone_ok} (violations={violations})")

    return {
        "train_mse": train_mse,
        "test_mse": test_mse,
        "coef_l2_norm": coef_l2_norm,
        "coef_l2_error": coef_l2_error,
        "nnz": nnz,
        "converged": float(result.converged),
        "monotone_ok": float(monotone_ok),
    }


def make_collinear_regression(
    seed: int = 2026,
    n_samples: int = 280,
    noise_std: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    latent = rng.normal(size=(n_samples, 4))
    x0 = latent[:, 0] + 0.02 * rng.normal(size=n_samples)
    x1 = latent[:, 0] + 0.02 * rng.normal(size=n_samples)
    x2 = latent[:, 1] + 0.02 * rng.normal(size=n_samples)
    x3 = latent[:, 1] + 0.02 * rng.normal(size=n_samples)

    extra = rng.normal(size=(n_samples, 8))
    x = np.column_stack([x0, x1, x2, x3, extra])

    true_coef = np.array([2.8, -2.2, 1.9, 0.0, 1.2, 0.0, 0.0, -1.0, 0.0, 0.0, 0.7, 0.0], dtype=float)
    y = x @ true_coef + noise_std * rng.normal(size=n_samples)

    return x, y, true_coef


def main() -> None:
    x, y, true_coef_raw = make_collinear_regression(seed=2026)
    validate_dataset(x, y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_ratio=0.25, seed=2026)
    x_train_std, x_test_std, _, x_std = standardize_from_train(x_train, x_test)

    # True coefficients represented in standardized feature space.
    true_coef_std = true_coef_raw * x_std

    condition_before = float(np.linalg.cond(x_train_std.T @ x_train_std))
    condition_after_ridge = float(np.linalg.cond(x_train_std.T @ x_train_std + 5.0 * np.eye(x_train_std.shape[1])))

    print("Linear Regression with Regularization demo")
    print(f"dataset: X={x.shape}, y={y.shape}, train={x_train.shape[0]}, test={x_test.shape[0]}")
    print(f"condition_number(X^T X): {condition_before:.4e}")
    print(f"condition_number(X^T X + 5I): {condition_after_ridge:.4e}")

    ols_result = fit_ols_closed_form(x_train_std, y_train)
    ridge_result = fit_ridge_closed_form(x_train_std, y_train, alpha=5.0)
    lasso_result = fit_lasso_coordinate_descent(x_train_std, y_train, alpha=0.08, tol=1e-8, max_epochs=3000)

    ols_summary = summarize_result(ols_result, x_train_std, y_train, x_test_std, y_test, true_coef_std)
    ridge_summary = summarize_result(ridge_result, x_train_std, y_train, x_test_std, y_test, true_coef_std)
    lasso_summary = summarize_result(lasso_result, x_train_std, y_train, x_test_std, y_test, true_coef_std)

    print("\n=== Coefficients (standardized feature space) ===")
    print("true:", np.array2string(true_coef_std, precision=4, suppress_small=True))
    print("ols: ", np.array2string(ols_result.coef, precision=4, suppress_small=True))
    print("ridge:", np.array2string(ridge_result.coef, precision=4, suppress_small=True))
    print("lasso:", np.array2string(lasso_result.coef, precision=4, suppress_small=True))

    ridge_generalization_gain = ols_summary["test_mse"] - ridge_summary["test_mse"]
    lasso_generalization_gain = ols_summary["test_mse"] - lasso_summary["test_mse"]
    lasso_sparsity_gain = ols_summary["nnz"] - lasso_summary["nnz"]
    all_checks = (
        ridge_summary["coef_l2_norm"] < ols_summary["coef_l2_norm"]
        and lasso_generalization_gain > 0.0
        and lasso_sparsity_gain > 0.0
        and lasso_result.converged
        and lasso_summary["monotone_ok"] > 0.5
    )

    print("\n=== Final Checks ===")
    print(f"ridge_test_mse_improvement_vs_ols: {ridge_generalization_gain:.8f}")
    print(f"lasso_test_mse_improvement_vs_ols: {lasso_generalization_gain:.8f}")
    print(f"lasso_nonzero_reduction_vs_ols: {lasso_sparsity_gain:.0f}")
    print(f"global_checks_pass: {all_checks}")


if __name__ == "__main__":
    main()

"""Polynomial regression MVP (closed-form least squares with degree selection)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

Array = np.ndarray


@dataclass
class PolynomialRegressionModel:
    degree: int
    coefficients: Array


def validate_1d_array(name: str, arr: Array) -> None:
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape={arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")


def polynomial_design_matrix(x: Array, degree: int) -> Array:
    validate_1d_array("x", x)
    if degree < 1:
        raise ValueError(f"degree must be >= 1, got {degree}.")
    cols = [np.ones_like(x, dtype=float)]
    for p in range(1, degree + 1):
        cols.append(np.power(x, p))
    return np.column_stack(cols)


def fit_polynomial_regression(
    x: Array,
    y: Array,
    degree: int,
    l2_reg: float = 1e-8,
) -> PolynomialRegressionModel:
    validate_1d_array("x", x)
    validate_1d_array("y", y)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")
    if x.size < 2:
        raise ValueError("Need at least 2 samples.")
    if l2_reg < 0.0:
        raise ValueError("l2_reg must be >= 0.")

    design = polynomial_design_matrix(x, degree)
    gram = design.T @ design
    reg = l2_reg * np.eye(gram.shape[0], dtype=float)
    reg[0, 0] = 0.0  # Keep bias term unregularized.
    rhs = design.T @ y

    try:
        coef = np.linalg.solve(gram + reg, rhs)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(gram + reg) @ rhs

    return PolynomialRegressionModel(degree=degree, coefficients=coef)


def predict_polynomial(model: PolynomialRegressionModel, x: Array) -> Array:
    design = polynomial_design_matrix(x, model.degree)
    return design @ model.coefficients


def rmse(y_true: Array, y_pred: Array) -> float:
    validate_1d_array("y_true", y_true)
    validate_1d_array("y_pred", y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2_score(y_true: Array, y_pred: Array) -> float:
    validate_1d_array("y_true", y_true)
    validate_1d_array("y_pred", y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    y_mean = float(np.mean(y_true))
    ss_tot = float(np.sum((y_true - y_mean) ** 2))
    if ss_tot <= 1e-15:
        return 1.0 if ss_res <= 1e-15 else 0.0
    return float(1.0 - ss_res / ss_tot)


def split_train_valid_test(
    x: Array,
    y: Array,
    train_ratio: float = 0.6,
    valid_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[Array, Array, Array, Array, Array, Array]:
    validate_1d_array("x", x)
    validate_1d_array("y", y)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")
    if not (0.0 < train_ratio < 1.0 and 0.0 < valid_ratio < 1.0):
        raise ValueError("train_ratio and valid_ratio must be in (0, 1).")
    if train_ratio + valid_ratio >= 1.0:
        raise ValueError("train_ratio + valid_ratio must be < 1.")

    n = x.size
    if n < 10:
        raise ValueError("Need at least 10 samples for train/valid/test split.")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    train_end = int(np.floor(n * train_ratio))
    valid_end = train_end + int(np.floor(n * valid_ratio))
    if train_end < 2 or valid_end <= train_end or valid_end >= n:
        raise ValueError("Invalid split sizes; adjust sample size or ratios.")

    train_idx = idx[:train_end]
    valid_idx = idx[train_end:valid_end]
    test_idx = idx[valid_end:]
    return (
        x[train_idx],
        y[train_idx],
        x[valid_idx],
        y[valid_idx],
        x[test_idx],
        y[test_idx],
    )


def generate_synthetic_data(
    n_samples: int = 240,
    seed: int = 7,
    noise_std: float = 0.35,
) -> Tuple[Array, Array, Array]:
    if n_samples < 20:
        raise ValueError("n_samples must be >= 20.")
    if noise_std < 0.0:
        raise ValueError("noise_std must be >= 0.")

    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.2, 2.2, size=n_samples)
    true_coef = np.array([1.2, -1.8, 0.6, 0.9], dtype=float)  # degree-3 ground truth
    y_clean = polynomial_design_matrix(x, degree=3) @ true_coef
    y = y_clean + rng.normal(0.0, noise_std, size=n_samples)
    return x, y, true_coef


def select_best_degree(
    x_train: Array,
    y_train: Array,
    x_valid: Array,
    y_valid: Array,
    candidate_degrees: Sequence[int],
    l2_reg: float = 1e-8,
) -> Tuple[int, List[Tuple[int, float, float]]]:
    if not candidate_degrees:
        raise ValueError("candidate_degrees cannot be empty.")

    records: List[Tuple[int, float, float]] = []
    best_degree = -1
    best_valid_rmse = float("inf")

    for degree in candidate_degrees:
        model = fit_polynomial_regression(x_train, y_train, degree=degree, l2_reg=l2_reg)
        train_pred = predict_polynomial(model, x_train)
        valid_pred = predict_polynomial(model, x_valid)
        train_rmse = rmse(y_train, train_pred)
        valid_rmse = rmse(y_valid, valid_pred)
        records.append((degree, train_rmse, valid_rmse))

        is_better = valid_rmse < best_valid_rmse - 1e-12
        tie_break = abs(valid_rmse - best_valid_rmse) <= 1e-12 and degree < best_degree
        if is_better or tie_break:
            best_degree = degree
            best_valid_rmse = valid_rmse

    if best_degree <= 0:
        raise RuntimeError("Failed to select a valid polynomial degree.")
    return best_degree, records


def print_degree_search(records: Sequence[Tuple[int, float, float]]) -> None:
    print("degree | train_rmse      | valid_rmse")
    print("-" * 40)
    for degree, train_rmse, valid_rmse in records:
        print(f"{degree:6d} | {train_rmse:14.8f} | {valid_rmse:10.8f}")


def print_prediction_samples(x_test: Array, y_test: Array, y_pred: Array, top_k: int = 8) -> None:
    validate_1d_array("x_test", x_test)
    validate_1d_array("y_test", y_test)
    validate_1d_array("y_pred", y_pred)
    if not (x_test.shape == y_test.shape == y_pred.shape):
        raise ValueError("x_test, y_test, y_pred must have same shape.")
    top_k = int(max(1, min(top_k, x_test.size)))
    order = np.argsort(x_test)
    print("\nSample predictions (sorted by x):")
    print("x            | y_true       | y_pred       | abs_error")
    print("-" * 60)
    for idx in order[:top_k]:
        abs_err = abs(y_test[idx] - y_pred[idx])
        print(
            f"{x_test[idx]:11.6f} | {y_test[idx]:11.6f} | "
            f"{y_pred[idx]:11.6f} | {abs_err:9.6f}"
        )


def main() -> None:
    x, y, true_coef = generate_synthetic_data(n_samples=240, seed=7, noise_std=0.35)
    x_train, y_train, x_valid, y_valid, x_test, y_test = split_train_valid_test(
        x, y, train_ratio=0.6, valid_ratio=0.2, seed=42
    )
    print(
        "Dataset split: "
        f"train={x_train.size}, valid={x_valid.size}, test={x_test.size}"
    )

    candidate_degrees = list(range(1, 9))
    best_degree, records = select_best_degree(
        x_train,
        y_train,
        x_valid,
        y_valid,
        candidate_degrees=candidate_degrees,
        l2_reg=1e-8,
    )
    print_degree_search(records)
    print(f"\nSelected degree by validation RMSE: {best_degree}")

    x_fit = np.concatenate([x_train, x_valid])
    y_fit = np.concatenate([y_train, y_valid])
    final_model = fit_polynomial_regression(x_fit, y_fit, degree=best_degree, l2_reg=1e-8)

    train_pred = predict_polynomial(final_model, x_fit)
    test_pred = predict_polynomial(final_model, x_test)
    train_rmse = rmse(y_fit, train_pred)
    test_rmse = rmse(y_test, test_pred)
    train_r2 = r2_score(y_fit, train_pred)
    test_r2 = r2_score(y_test, test_pred)

    print("\nFinal model coefficients (from x^0 to x^d):")
    print(np.array2string(final_model.coefficients, precision=6, floatmode="fixed"))
    print("Ground-truth coefficients used to generate data (degree 3):")
    print(np.array2string(true_coef, precision=6, floatmode="fixed"))

    print("\nMetrics:")
    print(f"train+valid RMSE: {train_rmse:.8f}")
    print(f"test RMSE:        {test_rmse:.8f}")
    print(f"train+valid R^2:  {train_r2:.8f}")
    print(f"test R^2:         {test_r2:.8f}")

    print_prediction_samples(x_test, y_test, test_pred, top_k=8)

    pass_flag = test_r2 > 0.90 and test_rmse < 0.80
    print("\nSummary:")
    print(f"selected_degree={best_degree}, pass={pass_flag}")


if __name__ == "__main__":
    main()

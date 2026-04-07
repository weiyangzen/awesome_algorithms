"""Minimal runnable MVP for linear regression via least squares."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RegressionResult:
    beta: np.ndarray
    y_pred: np.ndarray
    mse: float
    r2: float
    residual_norm_l2: float


def generate_synthetic_data(
    n_samples: int = 120,
    seed: int = 42,
    noise_std: float = 0.30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a deterministic two-feature linear dataset."""
    if n_samples < 3:
        raise ValueError("n_samples must be >= 3")
    if noise_std < 0.0:
        raise ValueError("noise_std must be >= 0")

    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-3.0, 3.0, size=n_samples)
    x2 = rng.normal(loc=0.5, scale=1.2, size=n_samples)
    x3 = rng.uniform(-1.0, 1.0, size=n_samples)

    x = np.column_stack((x1, x2, x3))

    # Ground-truth model: y = beta0 + beta1*x1 + beta2*x2 + beta3*x3 + noise
    beta_true = np.array([2.0, 1.5, -0.8, 0.6], dtype=float)
    noise = rng.normal(loc=0.0, scale=noise_std, size=n_samples)
    y = beta_true[0] + x @ beta_true[1:] + noise

    return x, y, beta_true


def validate_xy(x: np.ndarray, y: np.ndarray) -> None:
    if x.ndim != 2:
        raise ValueError("x must be 2D: shape (n_samples, n_features)")
    if y.ndim != 1:
        raise ValueError("y must be 1D: shape (n_samples,)")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of samples")
    if x.shape[0] < x.shape[1] + 1:
        raise ValueError("need at least n_features+1 samples for stable fitting")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("x and y must contain only finite values")


def add_intercept_column(x: np.ndarray) -> np.ndarray:
    ones = np.ones((x.shape[0], 1), dtype=float)
    return np.hstack((ones, x.astype(float, copy=False)))


def solve_least_squares_normal_equation(x: np.ndarray, y: np.ndarray) -> RegressionResult:
    """Solve min ||Xb - y||^2 using normal equation with a pseudo-inverse fallback."""
    validate_xy(x, y)
    x_aug = add_intercept_column(x)

    xtx = x_aug.T @ x_aug
    xty = x_aug.T @ y

    rank = np.linalg.matrix_rank(xtx)
    if rank == xtx.shape[0]:
        beta = np.linalg.solve(xtx, xty)
    else:
        beta = np.linalg.pinv(xtx) @ xty

    y_pred = x_aug @ beta
    residual = y - y_pred

    sse = float(np.sum(residual**2))
    mse = float(np.mean(residual**2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - sse / sst if sst > 0.0 else 1.0

    return RegressionResult(
        beta=beta,
        y_pred=y_pred,
        mse=mse,
        r2=r2,
        residual_norm_l2=float(np.linalg.norm(residual, ord=2)),
    )


def print_dataset_preview(x: np.ndarray, y: np.ndarray, rows: int = 5) -> None:
    rows = max(1, min(rows, x.shape[0]))
    print("dataset preview (first rows):")
    print(f"{'idx':>4} {'x1':>10} {'x2':>10} {'x3':>10} {'y':>12}")
    for i in range(rows):
        print(
            f"{i:4d} "
            f"{x[i, 0]:10.4f} {x[i, 1]:10.4f} {x[i, 2]:10.4f} {y[i]:12.6f}"
        )


def print_result_summary(beta_true: np.ndarray, result: RegressionResult, x: np.ndarray, y: np.ndarray) -> None:
    cond = float(np.linalg.cond(add_intercept_column(x).T @ add_intercept_column(x)))
    beta_error = result.beta - beta_true

    print("\nleast-squares fit summary:")
    print(f"samples={x.shape[0]}, features={x.shape[1]}")
    print("true beta      :", np.array2string(beta_true, precision=6, suppress_small=True))
    print("estimated beta :", np.array2string(result.beta, precision=6, suppress_small=True))
    print("beta error     :", np.array2string(beta_error, precision=6, suppress_small=True))
    print(f"MSE={result.mse:.8f}")
    print(f"R2={result.r2:.8f}")
    print(f"||residual||2={result.residual_norm_l2:.8f}")
    print(f"cond(X^T X)={cond:.6e}")

    print("\nprediction preview (first 5 rows):")
    print(f"{'idx':>4} {'y_true':>12} {'y_pred':>12} {'residual':>12}")
    n_rows = min(5, y.shape[0])
    for i in range(n_rows):
        residual = y[i] - result.y_pred[i]
        print(f"{i:4d} {y[i]:12.6f} {result.y_pred[i]:12.6f} {residual:12.6f}")


def main() -> None:
    x, y, beta_true = generate_synthetic_data(n_samples=120, seed=42, noise_std=0.30)
    print("Linear Regression - Least Squares MVP")
    print("=" * 46)

    print_dataset_preview(x, y, rows=5)

    result = solve_least_squares_normal_equation(x, y)
    print_result_summary(beta_true, result, x, y)


if __name__ == "__main__":
    main()

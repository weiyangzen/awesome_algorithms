"""Linear Regression MVP (scratch normal equation + sklearn baseline).

Run:
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass
class RegressionMetrics:
    """Container for standard regression metrics."""

    mse: float
    rmse: float
    mae: float
    r2: float


def make_synthetic_linear_data(
    n_samples: int = 640,
    noise_std: float = 0.60,
    seed: int = 202,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Create a reproducible linear dataset with known ground-truth parameters."""
    rng = np.random.default_rng(seed)
    x = rng.normal(loc=0.0, scale=1.0, size=(n_samples, 4)).astype(np.float64)

    w_true = np.array([2.5, -1.8, 0.7, 3.2], dtype=np.float64)
    b_true = -0.4
    noise = rng.normal(loc=0.0, scale=noise_std, size=n_samples).astype(np.float64)
    y = x @ w_true + b_true + noise
    return x, y, w_true, b_true


def train_test_split_numpy(
    x: np.ndarray,
    y: np.ndarray,
    test_ratio: float = 0.20,
    seed: int = 202,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic train/test split without external dependencies."""
    if x.ndim != 2:
        raise ValueError("x must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same sample count")
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be in (0, 1)")

    n_samples = x.shape[0]
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    n_test = max(1, int(round(n_samples * test_ratio)))
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


class LinearRegressionNormalEq:
    """Compact OLS solver using normal equation with tiny L2 stabilization."""

    def __init__(self, l2: float = 1e-10) -> None:
        if l2 < 0:
            raise ValueError("l2 must be non-negative")
        self.l2 = float(l2)
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LinearRegressionNormalEq":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if x.ndim != 2:
            raise ValueError("x must be 2D")
        if y.ndim != 1:
            raise ValueError("y must be 1D")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y shape mismatch")

        n_samples, n_features = x.shape
        x_bar = np.concatenate([np.ones((n_samples, 1), dtype=np.float64), x], axis=1)

        reg = self.l2 * np.eye(n_features + 1, dtype=np.float64)
        reg[0, 0] = 0.0  # Do not regularize intercept.

        lhs = x_bar.T @ x_bar + reg
        rhs = x_bar.T @ y
        theta = np.linalg.solve(lhs, rhs)

        self.intercept_ = float(theta[0])
        self.coef_ = theta[1:]
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("model is not fitted")
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("x must be 2D")
        return x @ self.coef_ + self.intercept_


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    """Compute MSE/RMSE/MAE/R^2."""
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return RegressionMetrics(mse=mse, rmse=rmse, mae=mae, r2=r2)


def main() -> None:
    print("Linear Regression MVP (CS-0091)")
    print("=" * 72)

    x, y, w_true, b_true = make_synthetic_linear_data(n_samples=640, noise_std=0.60, seed=202)
    x_train, x_test, y_train, y_test = train_test_split_numpy(x, y, test_ratio=0.20, seed=202)

    scratch = LinearRegressionNormalEq(l2=1e-10).fit(x_train, y_train)
    sklearn_model = LinearRegression(fit_intercept=True).fit(x_train, y_train)

    y_pred_train_scratch = scratch.predict(x_train)
    y_pred_test_scratch = scratch.predict(x_test)
    y_pred_train_sklearn = sklearn_model.predict(x_train)
    y_pred_test_sklearn = sklearn_model.predict(x_test)

    train_metrics_scratch = evaluate(y_train, y_pred_train_scratch)
    test_metrics_scratch = evaluate(y_test, y_pred_test_scratch)
    train_metrics_sklearn = evaluate(y_train, y_pred_train_sklearn)
    test_metrics_sklearn = evaluate(y_test, y_pred_test_sklearn)

    report = pd.DataFrame(
        [
            {
                "model": "scratch-normal-equation",
                "split": "train",
                "mse": train_metrics_scratch.mse,
                "rmse": train_metrics_scratch.rmse,
                "mae": train_metrics_scratch.mae,
                "r2": train_metrics_scratch.r2,
            },
            {
                "model": "scratch-normal-equation",
                "split": "test",
                "mse": test_metrics_scratch.mse,
                "rmse": test_metrics_scratch.rmse,
                "mae": test_metrics_scratch.mae,
                "r2": test_metrics_scratch.r2,
            },
            {
                "model": "sklearn-linear-regression",
                "split": "train",
                "mse": train_metrics_sklearn.mse,
                "rmse": train_metrics_sklearn.rmse,
                "mae": train_metrics_sklearn.mae,
                "r2": train_metrics_sklearn.r2,
            },
            {
                "model": "sklearn-linear-regression",
                "split": "test",
                "mse": test_metrics_sklearn.mse,
                "rmse": test_metrics_sklearn.rmse,
                "mae": test_metrics_sklearn.mae,
                "r2": test_metrics_sklearn.r2,
            },
        ]
    )

    coef_l2_gap = float(np.linalg.norm(scratch.coef_ - sklearn_model.coef_))
    intercept_gap = float(abs(scratch.intercept_ - sklearn_model.intercept_))

    coef_true_gap = float(np.linalg.norm(scratch.coef_ - w_true))
    intercept_true_gap = float(abs(scratch.intercept_ - b_true))

    prediction_gap_test = float(np.mean(np.abs(y_pred_test_scratch - y_pred_test_sklearn)))

    params_df = pd.DataFrame(
        {
            "feature": ["intercept", "x1", "x2", "x3", "x4"],
            "ground_truth": [b_true, *w_true.tolist()],
            "scratch": [scratch.intercept_, *scratch.coef_.tolist()],
            "sklearn": [float(sklearn_model.intercept_), *sklearn_model.coef_.tolist()],
        }
    )

    preview = pd.DataFrame(
        {
            "y_true": y_test[:10],
            "y_pred_scratch": y_pred_test_scratch[:10],
            "y_pred_sklearn": y_pred_test_sklearn[:10],
        }
    )

    print(f"dataset: n_samples={x.shape[0]}, n_features={x.shape[1]}")
    print(f"split: train={x_train.shape[0]}, test={x_test.shape[0]}")
    print()
    print("[Metrics]")
    print(report.to_string(index=False, float_format=lambda v: f"{v: .6f}"))
    print()
    print("[Parameters]")
    print(params_df.to_string(index=False, float_format=lambda v: f"{v: .6f}"))
    print()
    print(f"coef_l2_gap(scratch vs sklearn)      = {coef_l2_gap:.10f}")
    print(f"intercept_abs_gap(scratch vs sklearn)= {intercept_gap:.10f}")
    print(f"coef_l2_gap(scratch vs truth)        = {coef_true_gap:.6f}")
    print(f"intercept_abs_gap(scratch vs truth)  = {intercept_true_gap:.6f}")
    print(f"mean_abs_prediction_gap_on_test      = {prediction_gap_test:.12f}")
    print()
    print("[Prediction Preview]")
    print(preview.to_string(index=False, float_format=lambda v: f"{v: .6f}"))

    # Deterministic quality guards for this synthetic problem.
    if test_metrics_scratch.r2 < 0.95:
        raise RuntimeError(f"scratch test R^2 too low: {test_metrics_scratch.r2:.6f}")
    if coef_l2_gap > 1e-6:
        raise RuntimeError(f"coef mismatch vs sklearn too large: {coef_l2_gap:.6e}")
    if intercept_gap > 1e-6:
        raise RuntimeError(f"intercept mismatch vs sklearn too large: {intercept_gap:.6e}")
    if coef_true_gap > 0.20:
        raise RuntimeError(f"scratch coefficients deviate too much from truth: {coef_true_gap:.6f}")

    print("=" * 72)
    print("Run completed successfully.")


if __name__ == "__main__":
    main()

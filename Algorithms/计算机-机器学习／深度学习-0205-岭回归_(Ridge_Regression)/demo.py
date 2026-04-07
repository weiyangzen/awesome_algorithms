"""Ridge Regression MVP (closed-form + sklearn baseline).

Run:
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class RidgeClosedFormModel:
    """Closed-form Ridge regression with unregularized intercept."""

    alpha: float
    coef_: np.ndarray | None = None
    intercept_: float | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgeClosedFormModel":
        if self.alpha <= 0:
            raise ValueError("alpha must be > 0 for ridge regression")
        if x.ndim != 2:
            raise ValueError("x must be a 2D array")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same number of rows")

        x_mean = x.mean(axis=0)
        y_mean = float(y.mean())
        x_centered = x - x_mean
        y_centered = y - y_mean

        n_features = x.shape[1]
        gram = x_centered.T @ x_centered
        rhs = x_centered.T @ y_centered
        reg = self.alpha * np.eye(n_features)

        coef = np.linalg.solve(gram + reg, rhs)
        intercept = y_mean - float(x_mean @ coef)

        self.coef_ = coef
        self.intercept_ = intercept
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("model is not fitted")
        return x @ self.coef_ + self.intercept_


def make_correlated_regression_data(
    n_samples: int = 360,
    n_features: int = 12,
    latent_dim: int = 4,
    noise_std: float = 1.4,
    seed: int = 2026,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Generate a reproducible regression dataset with multicollinearity."""
    if latent_dim > n_features:
        raise ValueError("latent_dim should be <= n_features")

    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n_samples, latent_dim))
    mixing = rng.normal(size=(latent_dim, n_features))
    x = latent @ mixing + 0.08 * rng.normal(size=(n_samples, n_features))

    true_coef = np.array([3.0, -2.6, 0.0, 1.8, 0.0, -1.2, 0.7, 0.0, 0.0, 2.2, -0.9, 0.4])
    if n_features != true_coef.shape[0]:
        true_coef = rng.normal(scale=1.5, size=n_features)

    true_intercept = 5.0
    y = x @ true_coef + true_intercept + rng.normal(scale=noise_std, size=n_samples)
    return x, y, true_coef, true_intercept


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    return {"mse": float(mse), "rmse": rmse, "r2": float(r2)}


def condition_number(mat: np.ndarray) -> float:
    return float(np.linalg.cond(mat))


def main() -> None:
    alpha = 4.0
    x, y, true_coef, true_intercept = make_correlated_regression_data()

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    custom_ridge = RidgeClosedFormModel(alpha=alpha).fit(x_train_scaled, y_train)
    sklearn_ridge = Ridge(alpha=alpha, fit_intercept=True, solver="cholesky")
    sklearn_ridge.fit(x_train_scaled, y_train)

    ols = LinearRegression()
    ols.fit(x_train_scaled, y_train)

    pred_custom = custom_ridge.predict(x_test_scaled)
    pred_sklearn = sklearn_ridge.predict(x_test_scaled)
    pred_ols = ols.predict(x_test_scaled)

    metrics = pd.DataFrame(
        [
            {"model": "OLS", **evaluate_regression(y_test, pred_ols)},
            {"model": "Ridge(custom closed-form)", **evaluate_regression(y_test, pred_custom)},
            {"model": "Ridge(sklearn)", **evaluate_regression(y_test, pred_sklearn)},
        ]
    )

    gram_train = x_train_scaled.T @ x_train_scaled
    cond_plain = condition_number(gram_train)
    cond_ridge = condition_number(gram_train + alpha * np.eye(gram_train.shape[0]))

    feature_names = [f"x{i:02d}" for i in range(x_train_scaled.shape[1])]
    coef_frame = pd.DataFrame(
        {
            "feature": feature_names,
            "true_coef": true_coef,
            "ols_coef": ols.coef_,
            "custom_ridge_coef": custom_ridge.coef_,
            "sklearn_ridge_coef": sklearn_ridge.coef_,
        }
    )
    coef_frame["|ols| - |ridge| shrink"] = np.abs(coef_frame["ols_coef"]) - np.abs(
        coef_frame["custom_ridge_coef"]
    )

    preview = pd.DataFrame(
        {
            "y_true": y_test[:8],
            "y_pred_custom": pred_custom[:8],
            "y_pred_sklearn": pred_sklearn[:8],
            "abs_err_custom": np.abs(y_test[:8] - pred_custom[:8]),
        }
    )

    coef_l2_gap = float(np.linalg.norm(custom_ridge.coef_ - sklearn_ridge.coef_))
    intercept_gap = float(abs(custom_ridge.intercept_ - sklearn_ridge.intercept_))

    print("=== Ridge Regression MVP ===")
    print(f"n_train={x_train_scaled.shape[0]}, n_test={x_test_scaled.shape[0]}, n_features={x_train_scaled.shape[1]}")
    print(f"alpha={alpha}")
    print(f"true_intercept={true_intercept:.4f}")
    print()

    print("[Condition Number]")
    print(f"cond(X^T X)                 = {cond_plain:.4e}")
    print(f"cond(X^T X + alpha * I)     = {cond_ridge:.4e}")
    print()

    print("[Test Metrics]")
    print(metrics.to_string(index=False, float_format=lambda v: f"{v: .6f}"))
    print()

    print("[Custom vs sklearn Consistency]")
    print(f"L2(coef_custom - coef_sklearn) = {coef_l2_gap:.6e}")
    print(f"|intercept_custom - intercept_sklearn| = {intercept_gap:.6e}")
    print()

    print("[Coefficient Snapshot]")
    print(coef_frame.head(8).to_string(index=False, float_format=lambda v: f"{v: .6f}"))
    print()

    print("[Prediction Preview]")
    print(preview.to_string(index=False, float_format=lambda v: f"{v: .6f}"))


if __name__ == "__main__":
    main()

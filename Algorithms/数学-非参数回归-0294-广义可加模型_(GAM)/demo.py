"""Generalized Additive Model (GAM) MVP with spline backfitting.

This demo implements a transparent GAM for regression:
    y = beta0 + sum_j f_j(x_j) + epsilon

Each component function f_j is represented by a 1D B-spline basis.
Fitting uses coordinate-wise backfitting with quadratic smoothness penalty.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import SplineTransformer


def _second_order_difference_penalty(n_basis: int) -> np.ndarray:
    """Return D^T D where D is 2nd-order finite-difference matrix."""
    if n_basis < 3:
        return np.eye(n_basis, dtype=float)
    d2 = np.diff(np.eye(n_basis, dtype=float), n=2, axis=0)
    return d2.T @ d2


@dataclass
class AdditiveSplineGAM:
    """Minimal GAM for Gaussian response with identity link.

    Model:
        y = intercept + sum_j B_j(x_j) @ theta_j + noise

    Training:
        Coordinate-wise backfitting:
            theta_j <- argmin ||r_j - B_j theta_j||^2 + alpha * ||D theta_j||^2
    """

    n_knots: int = 9
    degree: int = 3
    alpha: float = 0.1
    max_iter: int = 200
    tol: float = 1e-5

    def __post_init__(self) -> None:
        if self.n_knots < 4:
            raise ValueError("n_knots must be >= 4")
        if self.degree < 1:
            raise ValueError("degree must be >= 1")
        if self.alpha < 0:
            raise ValueError("alpha must be >= 0")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if self.tol <= 0:
            raise ValueError("tol must be positive")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdditiveSplineGAM":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features)")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array of shape (n_samples,)")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples")
        if X.shape[0] < 4:
            raise ValueError("need at least 4 samples")
        if np.any(~np.isfinite(X)) or np.any(~np.isfinite(y)):
            raise ValueError("X and y must be finite")

        n_samples, n_features = X.shape
        self.n_features_ = n_features

        self.transformers_: list[SplineTransformer] = []
        design_mats: list[np.ndarray] = []
        penalties: list[np.ndarray] = []

        for j in range(n_features):
            transformer = SplineTransformer(
                n_knots=self.n_knots,
                degree=self.degree,
                include_bias=False,
                extrapolation="constant",
            )
            bj = transformer.fit_transform(X[:, [j]])
            self.transformers_.append(transformer)
            design_mats.append(bj)
            penalties.append(_second_order_difference_penalty(bj.shape[1]))

        intercept = float(np.mean(y))
        components = np.zeros((n_features, n_samples), dtype=float)
        coefs = [np.zeros(bj.shape[1], dtype=float) for bj in design_mats]
        offsets = np.zeros(n_features, dtype=float)

        self.training_history_ = []
        eye_cache = [np.eye(bj.shape[1], dtype=float) for bj in design_mats]

        for it in range(1, self.max_iter + 1):
            prev_components = components.copy()

            for j in range(n_features):
                # Partial residual for component j.
                residual = y - (intercept + np.sum(components, axis=0) - components[j])

                bj = design_mats[j]
                lhs = bj.T @ bj + self.alpha * penalties[j] + 1e-8 * eye_cache[j]
                rhs = bj.T @ residual
                theta = np.linalg.solve(lhs, rhs)

                fj = bj @ theta
                mean_fj = float(np.mean(fj))

                # Enforce identifiability: each component has mean zero on train set.
                components[j] = fj - mean_fj
                intercept += mean_fj

                coefs[j] = theta
                offsets[j] = mean_fj

            max_component_change = float(np.max(np.abs(components - prev_components)))
            train_rmse = float(np.sqrt(np.mean((y - (intercept + np.sum(components, axis=0))) ** 2)))

            self.training_history_.append(
                {
                    "iter": it,
                    "rmse": train_rmse,
                    "max_component_change": max_component_change,
                }
            )

            if max_component_change < self.tol:
                break

        self.intercept_ = intercept
        self.coefs_ = coefs
        self.component_offsets_ = offsets
        self.n_iter_ = len(self.training_history_)
        return self

    def _check_predict_input(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if not hasattr(self, "coefs_"):
            raise RuntimeError("model must be fitted before predict")
        if X.ndim != 2:
            raise ValueError("X must be 2D for predict")
        if X.shape[1] != self.n_features_:
            raise ValueError(f"X feature dimension mismatch: expected {self.n_features_}, got {X.shape[1]}")
        if np.any(~np.isfinite(X)):
            raise ValueError("X must be finite")
        return X

    def component_contributions(self, X: np.ndarray) -> np.ndarray:
        X = self._check_predict_input(X)
        n_samples = X.shape[0]
        components = np.zeros((n_samples, self.n_features_), dtype=float)

        for j in range(self.n_features_):
            bj = self.transformers_[j].transform(X[:, [j]])
            components[:, j] = bj @ self.coefs_[j] - self.component_offsets_[j]
        return components

    def predict(self, X: np.ndarray) -> np.ndarray:
        components = self.component_contributions(X)
        return self.intercept_ + np.sum(components, axis=1)


def make_synthetic_additive_data(
    n_samples: int = 420,
    noise_std: float = 0.22,
    seed: int = 2026,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Generate additive ground-truth data with 3 features."""
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0.0, 1.0, size=n_samples)
    x2 = rng.uniform(-1.0, 1.0, size=n_samples)
    x3 = rng.uniform(0.0, 1.0, size=n_samples)
    X = np.column_stack([x1, x2, x3])

    f1 = np.sin(2.0 * np.pi * x1)
    f2 = 0.8 * (x2**2 - 1.0 / 3.0)
    exp_term = np.exp(-3.0 * x3)
    f3 = exp_term - np.mean(exp_term)
    true_components = np.column_stack([f1, f2, f3])

    intercept = 1.35
    y_clean = intercept + np.sum(true_components, axis=1)
    y = y_clean + rng.normal(0.0, noise_std, size=n_samples)
    return X, y, true_components, intercept


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def search_alpha(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    alpha_grid: list[float],
    n_knots: int,
    degree: int,
) -> tuple[float, pd.DataFrame]:
    rows = []
    for alpha in alpha_grid:
        model = AdditiveSplineGAM(
            n_knots=n_knots,
            degree=degree,
            alpha=alpha,
            max_iter=250,
            tol=1e-5,
        ).fit(X_train, y_train)
        pred_val = model.predict(X_val)
        rows.append(
            {
                "alpha": alpha,
                "val_mse": mean_squared_error(y_val, pred_val),
                "val_mae": mean_absolute_error(y_val, pred_val),
                "n_iter": model.n_iter_,
            }
        )

    table = pd.DataFrame(rows).sort_values(by="val_mse", ascending=True).reset_index(drop=True)
    best_alpha = float(table.loc[0, "alpha"])
    return best_alpha, table


def main() -> None:
    X, y, true_components, true_intercept = make_synthetic_additive_data()

    X_train_full, X_test, y_train_full, y_test, comp_train_full, comp_test = train_test_split(
        X,
        y,
        true_components,
        test_size=0.25,
        random_state=7,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.25,
        random_state=11,
    )

    alpha_grid = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    best_alpha, alpha_table = search_alpha(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        alpha_grid=alpha_grid,
        n_knots=9,
        degree=3,
    )

    gam = AdditiveSplineGAM(n_knots=9, degree=3, alpha=best_alpha, max_iter=300, tol=1e-5)
    gam.fit(X_train_full, y_train_full)
    y_pred_gam = gam.predict(X_test)

    lin = LinearRegression().fit(X_train_full, y_train_full)
    y_pred_lin = lin.predict(X_test)

    gam_mse = mean_squared_error(y_test, y_pred_gam)
    gam_mae = mean_absolute_error(y_test, y_pred_gam)
    gam_r2 = r2_score(y_test, y_pred_gam)

    lin_mse = mean_squared_error(y_test, y_pred_lin)
    lin_mae = mean_absolute_error(y_test, y_pred_lin)
    lin_r2 = r2_score(y_test, y_pred_lin)

    component_pred = gam.component_contributions(X_test)
    component_corr = [_safe_corr(component_pred[:, j], comp_test[:, j]) for j in range(component_pred.shape[1])]

    print("=== Generalized Additive Model (GAM) Demo ===")
    print("Model: Gaussian family + identity link + spline backfitting")
    print(f"True intercept used in data generation: {true_intercept:.3f}")
    print("\nAlpha search results (validation split):")
    print(alpha_table.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nSelected alpha: {best_alpha:.6f}")
    print(f"Backfitting iterations (final fit): {gam.n_iter_}")

    print("\nTest metrics:")
    print(f"  GAM    -> MSE={gam_mse:.6f}, MAE={gam_mae:.6f}, R2={gam_r2:.6f}")
    print(f"  Linear -> MSE={lin_mse:.6f}, MAE={lin_mae:.6f}, R2={lin_r2:.6f}")

    print("\nRecovered component correlation with ground truth (test set):")
    print(f"  corr(f1_hat, f1_true) = {component_corr[0]:.4f}")
    print(f"  corr(f2_hat, f2_true) = {component_corr[1]:.4f}")
    print(f"  corr(f3_hat, f3_true) = {component_corr[2]:.4f}")

    print("\nSample predictions:")
    print("  idx    y_true     y_gam      y_linear")
    for idx in range(6):
        print(
            f"  {idx:>3d}  "
            f"{y_test[idx]:+9.4f}  "
            f"{y_pred_gam[idx]:+9.4f}  "
            f"{y_pred_lin[idx]:+9.4f}"
        )


if __name__ == "__main__":
    main()

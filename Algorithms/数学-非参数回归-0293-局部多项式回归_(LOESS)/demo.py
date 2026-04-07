"""LOESS (local polynomial regression) runnable MVP.

This script implements a minimal, explicit 1D LOESS regressor with:
- tricube distance weights
- local weighted polynomial fitting (degree 1 or 2)
- optional robust reweighting iterations (Tukey bisquare)

Run:
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


@dataclass
class LoessRegressor:
    """A minimal 1D LOESS regressor.

    Parameters
    ----------
    span:
        Fraction of training points used in each local neighborhood, in (0, 1].
    degree:
        Local polynomial degree. Supported: 1 (local linear) or 2 (local quadratic).
    robust_iters:
        Number of robust reweighting iterations. 0 means standard LOESS.
    regularization:
        Tiny L2 stabilization used in weighted least squares.
    eps:
        Numerical epsilon for stability checks.
    """

    span: float = 0.3
    degree: int = 2
    robust_iters: int = 2
    regularization: float = 1e-8
    eps: float = 1e-12

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LoessRegressor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, 1).")
        if X.shape[1] != 1:
            raise ValueError("This MVP supports exactly one feature (1D LOESS).")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        if not (0.0 < self.span <= 1.0):
            raise ValueError("span must be in (0, 1].")
        if self.degree not in (1, 2):
            raise ValueError("degree must be 1 or 2 in this MVP.")
        if self.robust_iters < 0:
            raise ValueError("robust_iters must be >= 0.")

        self.x_train_ = X[:, 0].copy()
        self.y_train_ = y.copy()
        self.n_samples_ = self.x_train_.shape[0]
        self.k_neighbors_ = max(
            self.degree + 1, int(np.ceil(self.span * self.n_samples_))
        )
        self.y_mean_ = float(np.mean(self.y_train_))

        robust_weights = np.ones(self.n_samples_, dtype=float)
        for _ in range(self.robust_iters):
            fitted = self._predict_internal(self.x_train_, robust_weights)
            residual = np.abs(self.y_train_ - fitted)
            scale = np.median(residual)
            if scale <= self.eps:
                break
            robust_weights = self._tukey_bisquare_weight(residual / (6.0 * scale))

        self.robust_weights_ = robust_weights
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "x_train_"):
            raise RuntimeError("Call fit before predict.")

        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != 1:
            raise ValueError("X must be a 2D array of shape (n_samples, 1).")

        return self._predict_internal(X[:, 0], self.robust_weights_)

    def _predict_internal(
        self, x_query: np.ndarray, robust_weights: np.ndarray
    ) -> np.ndarray:
        preds = np.empty_like(x_query, dtype=float)
        for i, x0 in enumerate(x_query):
            preds[i] = self._predict_one(float(x0), robust_weights)
        return preds

    def _predict_one(self, x0: float, robust_weights: np.ndarray) -> float:
        distances = np.abs(self.x_train_ - x0)

        # Adaptive local scale h(x0): distance to the k-th nearest neighbor.
        h = float(np.partition(distances, self.k_neighbors_ - 1)[self.k_neighbors_ - 1])
        if h <= self.eps:
            h = max(float(np.max(distances)), self.eps)

        u = distances / h
        local_w = self._tricube_weight(u)
        weights = local_w * robust_weights

        if float(np.sum(weights)) <= self.eps:
            return self.y_mean_

        x_centered = self.x_train_ - x0
        pred = self._weighted_least_squares(x_centered, self.y_train_, weights)
        if np.isfinite(pred):
            return float(pred)

        # Fallback: weighted average if local polynomial solve becomes unstable.
        weight_sum = float(np.sum(weights))
        if weight_sum <= self.eps:
            return self.y_mean_
        return float(np.dot(weights, self.y_train_) / weight_sum)

    def _weighted_least_squares(
        self, x_centered: np.ndarray, y: np.ndarray, w: np.ndarray
    ) -> float:
        design_cols = [np.ones_like(x_centered), x_centered]
        if self.degree == 2:
            design_cols.append(x_centered**2)
        Xloc = np.column_stack(design_cols)

        sqrt_w = np.sqrt(np.clip(w, 0.0, None))
        Xw = Xloc * sqrt_w[:, None]
        yw = y * sqrt_w

        xtx = Xw.T @ Xw
        xtx += self.regularization * np.eye(xtx.shape[0])
        xty = Xw.T @ yw

        try:
            beta = np.linalg.solve(xtx, xty)
        except np.linalg.LinAlgError:
            beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)

        return float(beta[0])

    @staticmethod
    def _tricube_weight(u: np.ndarray) -> np.ndarray:
        w = np.zeros_like(u, dtype=float)
        mask = u < 1.0
        um = u[mask]
        w[mask] = (1.0 - um**3) ** 3
        return w

    @staticmethod
    def _tukey_bisquare_weight(u: np.ndarray) -> np.ndarray:
        w = np.zeros_like(u, dtype=float)
        mask = u < 1.0
        um = u[mask]
        w[mask] = (1.0 - um**2) ** 2
        return w


def true_function(x: np.ndarray) -> np.ndarray:
    return np.sin(1.7 * x) + 0.25 * x**2


def make_synthetic_data(n: int = 280, seed: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(-3.0, 3.0, size=n))
    y_clean = true_function(x)

    noise = rng.normal(loc=0.0, scale=0.25, size=n)
    y = y_clean + noise

    # Inject a few large outliers to demonstrate robust LOESS behavior.
    outlier_idx = rng.choice(n, size=10, replace=False)
    y[outlier_idx] += rng.normal(loc=0.0, scale=2.0, size=outlier_idx.shape[0])

    return x.reshape(-1, 1), y, y_clean


def main() -> None:
    X, y, _ = make_synthetic_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

    span_candidates = [0.15, 0.22, 0.30, 0.40, 0.55]

    print("=== LOESS span search ===")
    best_span = None
    best_mse = float("inf")

    for span in span_candidates:
        model = LoessRegressor(span=span, degree=2, robust_iters=2)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mse = mean_squared_error(y_test, pred)
        print(f"span={span:.2f}, test_mse={mse:.6f}")
        if mse < best_mse:
            best_mse = mse
            best_span = span

    assert best_span is not None

    final_model = LoessRegressor(span=best_span, degree=2, robust_iters=2)
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n=== Best model ===")
    print(f"best_span={best_span:.2f}")
    print(f"MSE={mse:.6f}")
    print(f"MAE={mae:.6f}")
    print(f"R2={r2:.6f}")

    order = np.argsort(X_test[:, 0])
    sample_idx = order[:5]
    print("\n=== Prediction samples (sorted by x) ===")
    for idx in sample_idx:
        x_val = X_test[idx, 0]
        print(
            f"x={x_val:+.4f}, y_true={y_test[idx]:+.4f}, y_pred={y_pred[idx]:+.4f}"
        )


if __name__ == "__main__":
    main()

"""Spline regression MVP (runnable demo)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import UnivariateSpline


@dataclass
class SplineRegressor:
    """Minimal 1D smoothing-spline regressor based on SciPy FITPACK."""

    degree: int = 3
    smoothing_factor: float | None = None
    ext: int = 0

    def __post_init__(self) -> None:
        if not (1 <= self.degree <= 5):
            raise ValueError("degree must be in [1, 5]")
        if self.smoothing_factor is not None and self.smoothing_factor < 0.0:
            raise ValueError("smoothing_factor must be >= 0")
        if self.ext not in {0, 1, 2, 3}:
            raise ValueError("ext must be one of {0, 1, 2, 3}")

    @staticmethod
    def _aggregate_duplicate_x(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Merge duplicate x values to satisfy spline monotonic-x requirements."""
        unique_x, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
        y_sum = np.bincount(inverse, weights=y)
        y_mean = y_sum / counts

        # Use sqrt(count) as an uncertainty proxy: duplicated x gains higher confidence.
        weights = np.sqrt(counts.astype(float))
        return unique_x, y_mean, weights

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SplineRegressor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features)")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array of shape (n_samples,)")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        if X.shape[1] != 1:
            raise ValueError("this MVP supports only 1D input (n_features == 1)")
        if X.shape[0] < self.degree + 1:
            raise ValueError("not enough samples for requested spline degree")

        x = X[:, 0]
        order = np.argsort(x)
        x_sorted = x[order]
        y_sorted = y[order]

        x_unique, y_unique, weights = self._aggregate_duplicate_x(x_sorted, y_sorted)
        if x_unique.size < self.degree + 1:
            raise ValueError("unique x values are insufficient for requested spline degree")

        s_value = float(x_unique.size) if self.smoothing_factor is None else float(self.smoothing_factor)
        self.spline_ = UnivariateSpline(
            x_unique,
            y_unique,
            w=weights,
            k=self.degree,
            s=s_value,
            ext=self.ext,
        )

        self.x_train_ = x_unique
        self.y_train_ = y_unique
        self.weights_ = weights
        self.s_used_ = s_value
        self.train_residual_ = float(self.spline_.get_residual())
        return self

    def predict(self, X_query: np.ndarray) -> np.ndarray:
        if not hasattr(self, "spline_"):
            raise RuntimeError("model must be fitted before calling predict")

        X_query = np.asarray(X_query, dtype=float)
        if X_query.ndim != 2:
            raise ValueError("X_query must be a 2D array of shape (n_query, n_features)")
        if X_query.shape[1] != 1:
            raise ValueError("this MVP supports only 1D input (n_features == 1)")

        return np.asarray(self.spline_(X_query[:, 0]), dtype=float)


def make_synthetic_data(n_samples: int = 280, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0.0, 1.0, size=n_samples))
    y_true = np.sin(2.0 * np.pi * x) + 0.35 * np.cos(5.0 * np.pi * x)
    y = y_true + rng.normal(0.0, 0.15, size=n_samples)
    return x[:, None], y, y_true


def train_test_split_numpy(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.30,
    random_state: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be in (0, 1)")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")

    rng = np.random.default_rng(random_state)
    indices = rng.permutation(X.shape[0])
    test_count = int(round(X.shape[0] * test_size))
    test_count = min(max(test_count, 1), X.shape[0] - 1)

    test_idx = indices[:test_count]
    train_idx = indices[test_count:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def mean_squared_error_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.mean(diff**2))


def mean_absolute_error_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(diff)))


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    sse = np.sum((y_true - y_pred) ** 2)
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    if sst <= 1e-15:
        return 0.0
    return float(1.0 - sse / sst)


def evaluate_smoothing_factors(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    smoothing_factors: list[float],
    degree: int,
) -> tuple[float, dict[float, float]]:
    mse_by_s: dict[float, float] = {}
    for s_val in smoothing_factors:
        model = SplineRegressor(degree=degree, smoothing_factor=s_val)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse_by_s[s_val] = mean_squared_error_np(y_test, y_pred)

    best_s = min(mse_by_s, key=mse_by_s.get)
    return best_s, mse_by_s


def main() -> None:
    X, y, _ = make_synthetic_data()
    X_train, X_test, y_train, y_test = train_test_split_numpy(
        X,
        y,
        test_size=0.30,
        random_state=7,
    )

    degree = 3
    candidate_s = [0.0, 0.5, 1.5, 4.0, 10.0, 25.0]

    best_s, mse_table = evaluate_smoothing_factors(
        X_train,
        y_train,
        X_test,
        y_test,
        smoothing_factors=candidate_s,
        degree=degree,
    )

    best_model = SplineRegressor(degree=degree, smoothing_factor=best_s).fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    mse = mean_squared_error_np(y_test, y_pred)
    mae = mean_absolute_error_np(y_test, y_pred)
    r2 = r2_score_np(y_test, y_pred)

    print("=== Spline Regression Demo ===")
    print(f"Degree: {degree}")
    print("\nSmoothing-factor search (lower MSE is better):")
    for s_val in candidate_s:
        print(f"  s={s_val:>6.2f} -> MSE={mse_table[s_val]:.6f}")

    print(f"\nBest smoothing factor: s={best_s:.2f}")
    print(f"Weighted train residual (best model): {best_model.train_residual_:.6f}")
    print(f"Test MSE: {mse:.6f}")
    print(f"Test MAE: {mae:.6f}")
    print(f"Test R2 : {r2:.6f}")

    order = np.argsort(X_test[:, 0])
    print("\nSample predictions (sorted by x):")
    print("  x        y_true     y_pred")
    for idx in order[:8]:
        print(f"  {X_test[idx, 0]:.4f}   {y_test[idx]:+.4f}   {y_pred[idx]:+.4f}")


if __name__ == "__main__":
    main()

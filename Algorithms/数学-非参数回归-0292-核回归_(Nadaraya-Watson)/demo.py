"""Nadaraya-Watson kernel regression MVP (runnable demo)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NadarayaWatsonRegressor:
    """Minimal nonparametric kernel regressor."""

    bandwidth: float = 0.12
    kernel: str = "gaussian"
    eps: float = 1e-12

    def __post_init__(self) -> None:
        if self.bandwidth <= 0:
            raise ValueError("bandwidth must be positive")
        if self.kernel not in {"gaussian", "epanechnikov", "uniform"}:
            raise ValueError("kernel must be one of: gaussian, epanechnikov, uniform")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NadarayaWatsonRegressor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features)")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array of shape (n_samples,)")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        if X.shape[0] == 0:
            raise ValueError("X and y must not be empty")

        self.X_train_ = X
        self.y_train_ = y
        self.y_mean_ = float(np.mean(y))
        return self

    def _pairwise_scaled_distance(self, X_query: np.ndarray) -> np.ndarray:
        diff = (X_query[:, None, :] - self.X_train_[None, :, :]) / self.bandwidth
        return np.linalg.norm(diff, axis=2)

    def _kernel_weights(self, scaled_distance: np.ndarray) -> np.ndarray:
        u = scaled_distance
        if self.kernel == "gaussian":
            return np.exp(-0.5 * (u**2))
        if self.kernel == "epanechnikov":
            inside = np.abs(u) <= 1.0
            return 0.75 * (1.0 - u**2) * inside
        inside = np.abs(u) <= 1.0
        return 0.5 * inside

    def predict(self, X_query: np.ndarray) -> np.ndarray:
        if not hasattr(self, "X_train_"):
            raise RuntimeError("model must be fitted before calling predict")

        X_query = np.asarray(X_query, dtype=float)
        if X_query.ndim != 2:
            raise ValueError("X_query must be a 2D array of shape (n_query, n_features)")
        if X_query.shape[1] != self.X_train_.shape[1]:
            raise ValueError("X_query feature dimension must match training data")

        scaled_distance = self._pairwise_scaled_distance(X_query)
        weights = self._kernel_weights(scaled_distance)

        numerator = np.sum(weights * self.y_train_[None, :], axis=1)
        denominator = np.sum(weights, axis=1)

        fallback = np.full_like(numerator, fill_value=self.y_mean_, dtype=float)
        return np.divide(numerator, denominator, out=fallback, where=denominator > self.eps)


def make_synthetic_data(n_samples: int = 240, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, n_samples)
    y_true = np.sin(2.0 * np.pi * x) + 0.3 * np.cos(6.0 * np.pi * x)
    y = y_true + rng.normal(0.0, 0.18, size=n_samples)
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


def evaluate_bandwidths(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    bandwidths: list[float],
    kernel: str,
) -> tuple[float, dict[float, float]]:
    mse_by_bandwidth: dict[float, float] = {}
    for bw in bandwidths:
        model = NadarayaWatsonRegressor(bandwidth=bw, kernel=kernel)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse_by_bandwidth[bw] = mean_squared_error_np(y_test, y_pred)

    best_bandwidth = min(mse_by_bandwidth, key=mse_by_bandwidth.get)
    return best_bandwidth, mse_by_bandwidth


def main() -> None:
    X, y, _ = make_synthetic_data()
    X_train, X_test, y_train, y_test = train_test_split_numpy(
        X,
        y,
        test_size=0.30,
        random_state=7,
    )

    candidate_bandwidths = [0.03, 0.06, 0.10, 0.15, 0.24, 0.36]
    kernel = "gaussian"

    best_bw, mse_table = evaluate_bandwidths(
        X_train,
        y_train,
        X_test,
        y_test,
        bandwidths=candidate_bandwidths,
        kernel=kernel,
    )

    best_model = NadarayaWatsonRegressor(bandwidth=best_bw, kernel=kernel).fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    mse = mean_squared_error_np(y_test, y_pred)
    mae = mean_absolute_error_np(y_test, y_pred)
    r2 = r2_score_np(y_test, y_pred)

    print("=== Nadaraya-Watson Kernel Regression Demo ===")
    print(f"Kernel: {kernel}")
    print("\nBandwidth search (lower MSE is better):")
    for bw in candidate_bandwidths:
        print(f"  h={bw:>5.2f} -> MSE={mse_table[bw]:.6f}")

    print(f"\nBest bandwidth: h={best_bw:.2f}")
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

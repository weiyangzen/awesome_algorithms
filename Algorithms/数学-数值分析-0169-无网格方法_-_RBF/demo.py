"""RBF meshfree method MVP.

This script builds a minimal radial-basis-function regressor from scratch
(using NumPy linear algebra) and evaluates it on a 1D smooth function.
Run:
    python3 demo.py
"""

from __future__ import annotations

import numpy as np


class RBFRegressor:
    """Simple Gaussian-RBF regressor with Tikhonov regularization."""

    def __init__(self, epsilon: float = 4.0, lam: float = 1e-6) -> None:
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if lam < 0:
            raise ValueError("lam must be non-negative")
        self.epsilon = float(epsilon)
        self.lam = float(lam)
        self.centers_: np.ndarray | None = None
        self.weights_: np.ndarray | None = None

    @staticmethod
    def _pairwise_l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Return Euclidean distance matrix between rows of a and b."""
        a2 = np.sum(a * a, axis=1, keepdims=True)
        b2 = np.sum(b * b, axis=1, keepdims=True).T
        cross = np.einsum("ik,jk->ij", a, b, optimize=True)
        sq = np.maximum(a2 + b2 - 2.0 * cross, 0.0)
        return np.sqrt(sq)

    def _kernel(self, r: np.ndarray) -> np.ndarray:
        """Gaussian radial basis: phi(r) = exp(-(epsilon*r)^2)."""
        er = self.epsilon * r
        return np.exp(-(er * er))

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RBFRegressor":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if x.ndim == 1:
            x = x[:, None]
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same number of samples")

        k = self._kernel(self._pairwise_l2(x, x))
        system = k + self.lam * np.eye(x.shape[0], dtype=float)
        w = np.linalg.solve(system, y)

        self.centers_ = x
        self.weights_ = w
        return self

    def predict(self, xq: np.ndarray) -> np.ndarray:
        if self.centers_ is None or self.weights_ is None:
            raise RuntimeError("model is not fitted")

        xq = np.asarray(xq, dtype=float)
        if xq.ndim == 1:
            xq = xq[:, None]

        phi = self._kernel(self._pairwise_l2(xq, self.centers_))
        return np.einsum("ij,j->i", phi, self.weights_, optimize=True)


def target_function(x: np.ndarray) -> np.ndarray:
    """Ground-truth function used for synthetic benchmark."""
    x = np.asarray(x)
    return np.sin(2.0 * np.pi * x) + 0.3 * np.cos(5.0 * np.pi * x)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def main() -> None:
    rng = np.random.default_rng(7)

    # 1) build noisy training samples
    n_train = 28
    x_train = np.linspace(0.0, 1.0, n_train)
    y_clean = target_function(x_train)
    y_train = y_clean + 0.03 * rng.standard_normal(n_train)

    # 2) fit RBF model
    model = RBFRegressor(epsilon=4.5, lam=1e-5)
    model.fit(x_train[:, None], y_train)

    # 3) evaluate on dense test grid
    x_test = np.linspace(0.0, 1.0, 500)
    y_test = target_function(x_test)
    y_pred = model.predict(x_test[:, None])

    # 4) compare with piecewise-linear interpolation baseline
    y_lin = np.interp(x_test, x_train, y_train)

    rbf_rmse = rmse(y_test, y_pred)
    lin_rmse = rmse(y_test, y_lin)
    max_abs_err = float(np.max(np.abs(y_test - y_pred)))

    print("RBF Meshfree MVP (Gaussian kernel)")
    print(f"train_samples={n_train}, epsilon={model.epsilon}, lambda={model.lam}")
    print(f"RMSE (RBF)      : {rbf_rmse:.6f}")
    print(f"RMSE (Linear)   : {lin_rmse:.6f}")
    print(f"MaxAbsErr (RBF) : {max_abs_err:.6f}")
    print(f"RBF better than linear baseline: {rbf_rmse < lin_rmse}")


if __name__ == "__main__":
    main()

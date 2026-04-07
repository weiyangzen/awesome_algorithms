"""Minimal runnable MVP for Expectation Propagation (EP).

This demo implements EP for Gaussian Process binary classification
with probit likelihood:
    p(y_i=1|f_i) = Phi(f_i), y_i in {0,1}

Run:
    python3 demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    PANDAS_AVAILABLE = False


RANDOM_SEED = 7


def gaussian_pdf(x: np.ndarray | float) -> np.ndarray | float:
    """Standard normal PDF."""
    return np.exp(-0.5 * np.asarray(x) ** 2) / math.sqrt(2.0 * math.pi)


def gaussian_cdf(x: np.ndarray | float) -> np.ndarray | float:
    """Standard normal CDF implemented via erf for portability."""
    x_arr = np.asarray(x, dtype=float)
    erf_vec = np.vectorize(math.erf)
    out = 0.5 * (1.0 + erf_vec(x_arr / math.sqrt(2.0)))
    if np.isscalar(x):
        return float(out)
    return out


def rbf_kernel(
    xa: np.ndarray,
    xb: np.ndarray,
    lengthscale: float,
    variance: float,
) -> np.ndarray:
    """RBF kernel matrix."""
    xa = np.asarray(xa, dtype=float)
    xb = np.asarray(xb, dtype=float)
    if not (np.isfinite(xa).all() and np.isfinite(xb).all()):
        raise ValueError("Kernel input contains non-finite values.")

    # For this MVP we prefer the explicit broadcast form for numerical robustness.
    diff = xa[:, None, :] - xb[None, :, :]
    sqdist = np.sum(diff * diff, axis=2)
    return variance * np.exp(-0.5 * sqdist / (lengthscale**2))


def inverse_psd(matrix: np.ndarray, jitter: float = 1e-6) -> np.ndarray:
    """Numerically stable inverse for PSD matrices via eigenvalue flooring."""
    sym = 0.5 * (matrix + matrix.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    floor = max(float(jitter), 1e-10)
    eigvals = np.clip(eigvals, floor, None)
    inv = np.dot(eigvecs / eigvals, eigvecs.T)
    return 0.5 * (inv + inv.T)


def stratified_train_test_split(
    x: np.ndarray,
    y: np.ndarray,
    test_size: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simple stratified split without sklearn dependency."""
    rng = np.random.default_rng(random_seed)
    train_idx_parts: list[np.ndarray] = []
    test_idx_parts: list[np.ndarray] = []

    for label in np.unique(y):
        idx = np.where(y == label)[0]
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_size)))
        test_idx_parts.append(idx[:n_test])
        train_idx_parts.append(idx[n_test:])

    train_idx = np.concatenate(train_idx_parts)
    test_idx = np.concatenate(test_idx_parts)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


def make_two_moons_like(n_samples: int = 260, noise: float = 0.14, seed: int = RANDOM_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Generate a deterministic two-moons-like binary dataset with numpy only."""
    rng = np.random.default_rng(seed)
    n1 = n_samples // 2
    n2 = n_samples - n1

    t1 = rng.uniform(0.0, math.pi, size=n1)
    moon1 = np.column_stack([np.cos(t1), np.sin(t1)])

    t2 = rng.uniform(0.0, math.pi, size=n2)
    moon2 = np.column_stack([1.0 - np.cos(t2), -np.sin(t2) - 0.45])

    moon1 += rng.normal(0.0, noise, size=moon1.shape)
    moon2 += rng.normal(0.0, noise, size=moon2.shape)

    x = np.vstack([moon1, moon2])
    y = np.concatenate([np.ones(n1, dtype=int), np.zeros(n2, dtype=int)])

    perm = rng.permutation(n_samples)
    return x[perm], y[perm]


@dataclass
class EPMetrics:
    accuracy: float
    nll: float
    brier: float


class EPBinaryGP:
    """Expectation Propagation for GP binary classification (probit likelihood)."""

    def __init__(
        self,
        lengthscale: float = 1.0,
        kernel_variance: float = 1.0,
        damping: float = 0.7,
        max_sweeps: int = 20,
        tol: float = 1e-4,
        jitter: float = 1e-4,
    ) -> None:
        self.lengthscale = float(lengthscale)
        self.kernel_variance = float(kernel_variance)
        self.damping = float(damping)
        self.max_sweeps = int(max_sweeps)
        self.tol = float(tol)
        self.jitter = float(jitter)

        self.x_train: Optional[np.ndarray] = None
        self.y_pm1: Optional[np.ndarray] = None
        self.k_inv: Optional[np.ndarray] = None
        self.sigma: Optional[np.ndarray] = None
        self.mean: Optional[np.ndarray] = None
        self.tau: Optional[np.ndarray] = None
        self.nu: Optional[np.ndarray] = None
        self.history: list[float] = []

    def _check_ready(self) -> None:
        if any(v is None for v in [self.x_train, self.k_inv, self.sigma, self.mean]):
            raise RuntimeError("Model is not fitted. Call fit() first.")

    def fit(self, x: np.ndarray, y: np.ndarray, verbose: bool = False) -> "EPBinaryGP":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=int)

        uniq = set(np.unique(y).tolist())
        if uniq == {0, 1}:
            y_pm1 = 2 * y - 1
        elif uniq == {-1, 1}:
            y_pm1 = y
        else:
            raise ValueError("y must be binary labels in {0,1} or {-1,1}.")

        n = x.shape[0]
        k = rbf_kernel(x, x, lengthscale=self.lengthscale, variance=self.kernel_variance)
        k = k + self.jitter * np.eye(n)
        k_inv = inverse_psd(k, jitter=self.jitter)

        tau = np.zeros(n, dtype=float)
        nu = np.zeros(n, dtype=float)

        sigma = k.copy()
        mean = np.zeros(n, dtype=float)

        self.history = []

        for sweep in range(self.max_sweeps):
            max_delta = 0.0

            for i in range(n):
                sigma_ii = float(max(sigma[i, i], 1e-12))

                tau_cav = 1.0 / sigma_ii - tau[i]
                tau_cav = max(tau_cav, 1e-12)
                sigma2_cav = 1.0 / tau_cav

                nu_cav = mean[i] / sigma_ii - nu[i]
                mu_cav = sigma2_cav * nu_cav

                yi = float(y_pm1[i])
                z = yi * mu_cav / math.sqrt(1.0 + sigma2_cav)

                cdf_z = float(np.clip(gaussian_cdf(z), 1e-12, 1.0))
                ratio = float(gaussian_pdf(z) / cdf_z)

                mu_hat = mu_cav + yi * sigma2_cav / math.sqrt(1.0 + sigma2_cav) * ratio
                sigma2_hat = sigma2_cav - (sigma2_cav**2 / (1.0 + sigma2_cav)) * ratio * (ratio + z)
                sigma2_hat = max(float(sigma2_hat), 1e-12)

                tau_new = max(1.0 / sigma2_hat - tau_cav, 1e-12)
                nu_new = mu_hat / sigma2_hat - nu_cav

                tau_updated = (1.0 - self.damping) * tau[i] + self.damping * tau_new
                nu_updated = (1.0 - self.damping) * nu[i] + self.damping * nu_new

                max_delta = max(max_delta, abs(tau_updated - tau[i]), abs(nu_updated - nu[i]))
                tau[i] = tau_updated
                nu[i] = nu_updated

            sigma_inv = k_inv + np.diag(tau)
            sigma = inverse_psd(sigma_inv, jitter=self.jitter)
            mean = np.dot(sigma, nu)

            self.history.append(max_delta)
            if verbose:
                print(f"[EP] sweep={sweep + 1:02d}, max_site_delta={max_delta:.3e}")

            if max_delta < self.tol:
                break

        self.x_train = x
        self.y_pm1 = y_pm1
        self.k_inv = k_inv
        self.sigma = sigma
        self.mean = mean
        self.tau = tau
        self.nu = nu
        return self

    def predict_latent(self, x_new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._check_ready()
        assert self.x_train is not None
        assert self.k_inv is not None
        assert self.sigma is not None
        assert self.mean is not None

        x_new = np.asarray(x_new, dtype=float)

        k_s = rbf_kernel(self.x_train, x_new, self.lengthscale, self.kernel_variance)
        alpha = np.dot(self.k_inv, self.mean)
        mean_star = np.dot(k_s.T, alpha)

        b_mat = self.k_inv - np.linalg.multi_dot([self.k_inv, self.sigma, self.k_inv])
        var_star = self.kernel_variance - np.sum(k_s * np.dot(b_mat, k_s), axis=0)
        var_star = np.maximum(var_star, 1e-10)

        return mean_star, var_star

    def predict_proba(self, x_new: np.ndarray) -> np.ndarray:
        mean_star, var_star = self.predict_latent(x_new)
        z = mean_star / np.sqrt(1.0 + var_star)
        p1 = gaussian_cdf(z)
        return np.asarray(p1, dtype=float)

    def predict(self, x_new: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x_new) >= 0.5).astype(int)


def evaluate_binary(y_true: np.ndarray, p1: np.ndarray) -> EPMetrics:
    y_true = np.asarray(y_true, dtype=int)
    p1 = np.asarray(p1, dtype=float)
    p1 = np.clip(p1, 1e-9, 1.0 - 1e-9)

    pred = (p1 >= 0.5).astype(int)
    acc = float(np.mean(pred == y_true))
    nll = float(-np.mean(y_true * np.log(p1) + (1 - y_true) * np.log(1 - p1)))
    brier = float(np.mean((p1 - y_true) ** 2))
    return EPMetrics(accuracy=acc, nll=nll, brier=brier)


def main() -> None:
    x, y = make_two_moons_like(n_samples=260, noise=0.14, seed=RANDOM_SEED)
    x_train, x_test, y_train, y_test = stratified_train_test_split(
        x,
        y,
        test_size=0.25,
        random_seed=RANDOM_SEED,
    )

    model = EPBinaryGP(
        lengthscale=0.9,
        kernel_variance=1.25,
        damping=0.65,
        max_sweeps=25,
        tol=5e-4,
        jitter=1e-4,
    )
    model.fit(x_train, y_train, verbose=False)

    p_test = model.predict_proba(x_test)
    y_pred = (p_test >= 0.5).astype(int)
    metrics = evaluate_binary(y_test, p_test)

    print("=== Expectation Propagation (EP) MVP ===")
    print(f"Train size: {len(y_train)}, Test size: {len(y_test)}")
    print(f"EP sweeps executed: {len(model.history)}")
    print(f"Final max site delta: {model.history[-1]:.6e}")
    print(
        "Site precision tau stats: "
        f"min={float(np.min(model.tau)):.4e}, "
        f"mean={float(np.mean(model.tau)):.4e}, "
        f"max={float(np.max(model.tau)):.4e}"
    )

    print("\nTest metrics")
    print(f"  Accuracy : {metrics.accuracy:.4f}")
    print(f"  NLL      : {metrics.nll:.4f}")
    print(f"  Brier    : {metrics.brier:.4f}")

    margins = np.abs(p_test - 0.5)
    uncertain_idx = np.argsort(margins)[:8]

    if PANDAS_AVAILABLE:
        assert model.history is not None
        history_df = pd.DataFrame(
            {
                "sweep": np.arange(1, len(model.history) + 1),
                "max_site_delta": np.array(model.history),
            }
        )
        print("\nConvergence trace (last 8 sweeps)")
        print(history_df.tail(8).to_string(index=False))

        uncertain_df = pd.DataFrame(
            {
                "idx": uncertain_idx,
                "prob_y1": p_test[uncertain_idx],
                "pred": y_pred[uncertain_idx],
                "true": y_test[uncertain_idx],
            }
        )
        print("\nMost uncertain test samples (closest to p=0.5)")
        print(uncertain_df.to_string(index=False))
    else:
        print("\nConvergence trace unavailable as table because pandas is not installed.")
        print("Most uncertain indices:", uncertain_idx.tolist())


if __name__ == "__main__":
    main()

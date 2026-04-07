"""Nyström method MVP for kernel matrix approximation and kernel regression.

This script demonstrates:
1) constructing a low-rank Nyström approximation of an RBF kernel matrix,
2) using Nyström features for ridge regression,
3) comparing against full kernel ridge regression as a reference.

No interactive input is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class NystromModel:
    landmark_indices: np.ndarray
    landmarks: np.ndarray
    gamma: float
    eigvecs: np.ndarray
    inv_sqrt_eigvals: np.ndarray


def safe_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matmul with local warning suppression for BLAS-level floating-point noise."""
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return a @ b


def rbf_kernel(x: np.ndarray, y: np.ndarray, gamma: float) -> np.ndarray:
    """Compute the RBF kernel matrix exp(-gamma * ||xi-yj||^2)."""
    x_sq = np.sum(x * x, axis=1, keepdims=True)
    y_sq = np.sum(y * y, axis=1, keepdims=True).T
    sq_dist = x_sq + y_sq - 2.0 * safe_matmul(x, y.T)
    sq_dist = np.maximum(sq_dist, 0.0)
    return np.exp(-gamma * sq_dist)


def build_nystrom(
    x_train: np.ndarray,
    m: int,
    rank: int,
    gamma: float,
    rng: np.random.Generator,
    eps: float = 1e-12,
) -> Tuple[NystromModel, np.ndarray]:
    """Build Nyström model and return train features Z (shape n x rank_eff)."""
    n = x_train.shape[0]
    if not (1 <= m <= n):
        raise ValueError(f"m must satisfy 1 <= m <= n_train, got m={m}, n_train={n}.")
    if rank < 1:
        raise ValueError(f"rank must be >= 1, got {rank}.")

    landmark_indices = np.sort(rng.choice(n, size=m, replace=False))
    landmarks = x_train[landmark_indices]

    c = rbf_kernel(x_train, landmarks, gamma)  # n x m
    w = c[landmark_indices, :]  # m x m, equals K(landmarks, landmarks)

    eigvals, eigvecs = np.linalg.eigh(w)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    positive = eigvals > eps
    if not np.any(positive):
        raise ValueError("No positive eigenvalue in W; cannot form Nyström features.")

    max_rank = int(np.sum(positive))
    rank_eff = min(rank, max_rank)
    eigvals_r = eigvals[:rank_eff]
    eigvecs_r = eigvecs[:, :rank_eff]
    inv_sqrt = 1.0 / np.sqrt(eigvals_r)

    z_train = safe_matmul(c, eigvecs_r) * inv_sqrt[np.newaxis, :]

    model = NystromModel(
        landmark_indices=landmark_indices,
        landmarks=landmarks,
        gamma=gamma,
        eigvecs=eigvecs_r,
        inv_sqrt_eigvals=inv_sqrt,
    )
    return model, z_train


def transform_nystrom(model: NystromModel, x: np.ndarray) -> np.ndarray:
    """Map new samples x to Nyström feature space."""
    k_xm = rbf_kernel(x, model.landmarks, model.gamma)
    return safe_matmul(k_xm, model.eigvecs) * model.inv_sqrt_eigvals[np.newaxis, :]


def fit_ridge_primal(z: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Solve ridge regression in feature space: (Z^T Z + lam I) w = Z^T y."""
    if lam <= 0.0:
        raise ValueError(f"lam must be positive, got {lam}.")
    d = z.shape[1]
    a = safe_matmul(z.T, z) + lam * np.eye(d)
    b = safe_matmul(z.T, y)
    return np.linalg.solve(a, b)


def fit_kernel_ridge_dual(
    x_train: np.ndarray, y_train: np.ndarray, gamma: float, lam: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Reference full kernel ridge regression in dual form."""
    if lam <= 0.0:
        raise ValueError(f"lam must be positive, got {lam}.")
    k_train = rbf_kernel(x_train, x_train, gamma)
    n = k_train.shape[0]
    alpha = np.linalg.solve(k_train + lam * np.eye(n), y_train)
    return alpha, k_train


def predict_kernel_ridge(
    x_test: np.ndarray, x_train: np.ndarray, alpha: np.ndarray, gamma: float
) -> np.ndarray:
    k_test_train = rbf_kernel(x_test, x_train, gamma)
    return safe_matmul(k_test_train, alpha)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def relative_fro_error(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a, ord="fro")
    if denom == 0.0:
        return 0.0
    return float(np.linalg.norm(a - b, ord="fro") / denom)


def make_regression_data(
    n_train: int = 320,
    n_test: int = 200,
    noise_std: float = 0.08,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a smooth 1D nonlinear regression dataset."""
    rng = np.random.default_rng(seed)

    def f(x_col: np.ndarray) -> np.ndarray:
        return np.sin(1.5 * x_col) + 0.3 * np.cos(4.0 * x_col)

    x_train = rng.uniform(-3.0, 3.0, size=(n_train, 1))
    y_train = f(x_train[:, 0]) + noise_std * rng.normal(size=n_train)

    x_test = np.linspace(-3.0, 3.0, n_test, dtype=float).reshape(-1, 1)
    y_test = f(x_test[:, 0])
    return x_train, y_train, x_test, y_test


def run_single_case(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    gamma: float,
    lam: float,
    m: int,
    rank: int,
    seed: int,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)

    alpha, k_full = fit_kernel_ridge_dual(x_train, y_train, gamma=gamma, lam=lam)
    y_pred_full = predict_kernel_ridge(x_test, x_train, alpha, gamma=gamma)

    nys_model, z_train = build_nystrom(
        x_train=x_train,
        m=m,
        rank=rank,
        gamma=gamma,
        rng=rng,
    )
    z_test = transform_nystrom(nys_model, x_test)

    w = fit_ridge_primal(z_train, y_train, lam=lam)
    y_pred_nys = safe_matmul(z_test, w)

    k_hat = safe_matmul(z_train, z_train.T)

    return {
        "m": float(m),
        "rank_eff": float(z_train.shape[1]),
        "kernel_rel_fro_err": relative_fro_error(k_full, k_hat),
        "rmse_full_krr": rmse(y_test, y_pred_full),
        "rmse_nystrom_krr": rmse(y_test, y_pred_nys),
    }


def main() -> None:
    x_train, y_train, x_test, y_test = make_regression_data()

    gamma = 0.8
    lam = 1e-2

    # Three budgets to show approximation-quality trade-off.
    cases = [
        {"m": 24, "rank": 16, "seed": 7},
        {"m": 48, "rank": 24, "seed": 7},
        {"m": 96, "rank": 48, "seed": 7},
    ]

    print("Nyström MVP: kernel approximation + regression")
    print("=" * 92)
    print(
        f"{'m':>6} {'rank_eff':>10} {'relF(K,ZZ^T)':>16} "
        f"{'RMSE full KRR':>16} {'RMSE Nyström':>16} {'gap':>10}"
    )
    print("-" * 92)

    for cfg in cases:
        out = run_single_case(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            gamma=gamma,
            lam=lam,
            m=cfg["m"],
            rank=cfg["rank"],
            seed=cfg["seed"],
        )
        gap = out["rmse_nystrom_krr"] - out["rmse_full_krr"]
        print(
            f"{int(out['m']):6d} {int(out['rank_eff']):10d} "
            f"{out['kernel_rel_fro_err']:16.6e} {out['rmse_full_krr']:16.6e} "
            f"{out['rmse_nystrom_krr']:16.6e} {gap:10.3e}"
        )

    print("-" * 92)
    print(
        "Interpretation: larger m/rank usually improves kernel approximation "
        "(smaller relF) and narrows RMSE gap to full KRR."
    )


if __name__ == "__main__":
    main()

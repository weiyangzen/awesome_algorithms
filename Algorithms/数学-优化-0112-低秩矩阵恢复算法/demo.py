"""低秩矩阵恢复算法 MVP: matrix completion via SVT/proximal gradient.

运行:
    python3 demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class RecoveryResult:
    recovered: np.ndarray
    history: Dict[str, List[float]]
    iterations: int


def stable_matmul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Matrix multiply with finite-value guard to avoid silent NaN/Inf propagation."""
    with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
        out = left @ right
    if not np.all(np.isfinite(out)):
        raise FloatingPointError("Non-finite values produced in matrix multiplication.")
    return out


def make_low_rank_matrix(
    m: int,
    n: int,
    rank: int,
    seed: int,
    noise_std: float = 0.0,
) -> np.ndarray:
    """Construct a synthetic low-rank matrix M = U V^T (+ optional Gaussian noise)."""
    if rank <= 0 or rank > min(m, n):
        raise ValueError("rank must satisfy 1 <= rank <= min(m, n)")

    rng = np.random.default_rng(seed)
    u = rng.standard_normal((m, rank))
    v = rng.standard_normal((n, rank))
    matrix = stable_matmul(u, v.T) / np.sqrt(float(rank))

    if noise_std > 0.0:
        matrix = matrix + noise_std * rng.standard_normal((m, n))

    return matrix


def make_observation_mask(shape: Tuple[int, int], observe_ratio: float, seed: int) -> np.ndarray:
    """Sample observation mask Omega and ensure each row/column has at least one observation."""
    if not (0.0 < observe_ratio < 1.0):
        raise ValueError("observe_ratio must be in (0, 1)")

    m, n = shape
    rng = np.random.default_rng(seed)
    mask = rng.random((m, n)) < observe_ratio

    for i in range(m):
        if not mask[i].any():
            mask[i, rng.integers(0, n)] = True

    for j in range(n):
        if not mask[:, j].any():
            mask[rng.integers(0, m), j] = True

    return mask.astype(float)


def singular_value_thresholding(matrix: np.ndarray, tau: float) -> Tuple[np.ndarray, int, float]:
    """Apply soft-thresholding to singular values: prox_{tau ||.||_*}(matrix)."""
    u, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
    shrunk = np.maximum(singular_values - tau, 0.0)
    recovered = stable_matmul(u * shrunk, vt)
    est_rank = int(np.sum(shrunk > 1e-8))
    nuclear_norm = float(np.sum(shrunk))
    return recovered, est_rank, nuclear_norm


def recover_low_rank_matrix_svt(
    observed: np.ndarray,
    mask: np.ndarray,
    lam: float,
    step: float,
    max_iter: int,
    tol: float,
) -> RecoveryResult:
    """Solve min_X 0.5||P_Omega(X-observed)||_F^2 + lam||X||_* via proximal gradient."""
    if observed.shape != mask.shape:
        raise ValueError("observed and mask must share the same shape")
    if lam <= 0.0:
        raise ValueError("lam must be positive")
    if not (0.0 < step <= 1.0):
        raise ValueError("step must be in (0, 1]")

    x = np.zeros_like(observed)
    obs_count = max(float(np.sum(mask)), 1.0)
    eps = 1e-12

    history: Dict[str, List[float]] = {
        "objective": [],
        "obs_rmse": [],
        "rel_change": [],
        "rank": [],
    }

    for it in range(1, max_iter + 1):
        grad = mask * (x - observed)
        y = x - step * grad
        x_next, rank_est, nuclear_norm = singular_value_thresholding(y, tau=step * lam)

        residual_obs = mask * (x_next - observed)
        obj = 0.5 * float(np.sum(residual_obs**2)) + lam * nuclear_norm
        obs_rmse = float(np.sqrt(np.sum(residual_obs**2) / obs_count))

        rel_change = float(
            np.linalg.norm(x_next - x, ord="fro") / max(np.linalg.norm(x, ord="fro"), eps)
        )

        history["objective"].append(obj)
        history["obs_rmse"].append(obs_rmse)
        history["rel_change"].append(rel_change)
        history["rank"].append(float(rank_est))

        x = x_next

        if it >= 5 and rel_change < tol:
            return RecoveryResult(recovered=x, history=history, iterations=it)

    return RecoveryResult(recovered=x, history=history, iterations=max_iter)


def rmse(matrix_a: np.ndarray, matrix_b: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Compute RMSE on all entries or on mask-selected entries."""
    if matrix_a.shape != matrix_b.shape:
        raise ValueError("shape mismatch in rmse")

    diff = matrix_a - matrix_b
    if mask is None:
        return float(np.sqrt(np.mean(diff**2)))

    count = max(float(np.sum(mask)), 1.0)
    return float(np.sqrt(np.sum((mask * diff) ** 2) / count))


def main() -> None:
    # Synthetic low-rank matrix completion setup.
    m, n, true_rank = 60, 50, 4
    observe_ratio = 0.35
    matrix_true = make_low_rank_matrix(m=m, n=n, rank=true_rank, seed=7, noise_std=0.01)
    mask = make_observation_mask(shape=matrix_true.shape, observe_ratio=observe_ratio, seed=99)
    matrix_observed = mask * matrix_true

    # Hyperparameters for proximal gradient with nuclear norm regularization.
    lam = 0.35
    step = 1.0
    max_iter = 250
    tol = 1e-5

    result = recover_low_rank_matrix_svt(
        observed=matrix_observed,
        mask=mask,
        lam=lam,
        step=step,
        max_iter=max_iter,
        tol=tol,
    )

    recovered = result.recovered
    zero_fill_baseline = matrix_observed

    full_rmse_baseline = rmse(zero_fill_baseline, matrix_true)
    full_rmse_recovered = rmse(recovered, matrix_true)
    obs_rmse_recovered = rmse(recovered, matrix_true, mask=mask)
    unobs_mask = 1.0 - mask
    unobs_rmse_recovered = rmse(recovered, matrix_true, mask=unobs_mask)
    est_rank = int(np.linalg.matrix_rank(recovered, tol=1e-6))

    print("=== Low-Rank Matrix Recovery via SVT ===")
    print(f"Shape: {m}x{n}, true rank: {true_rank}, observe ratio: {observe_ratio:.2f}")
    print(
        f"Iterations: {result.iterations}, final relative change: "
        f"{result.history['rel_change'][-1]:.3e}"
    )
    print(f"Final objective: {result.history['objective'][-1]:.6f}")
    print(f"Estimated recovered rank: {est_rank}")
    print(f"Observed RMSE (recovered): {obs_rmse_recovered:.6f}")
    print(f"Unobserved RMSE (recovered): {unobs_rmse_recovered:.6f}")
    print(f"Full RMSE baseline (zero-fill): {full_rmse_baseline:.6f}")
    print(f"Full RMSE recovered (SVT): {full_rmse_recovered:.6f}")

    # Basic quality checks for this synthetic setting.
    if not full_rmse_recovered < 0.8 * full_rmse_baseline:
        raise RuntimeError(
            "Recovery quality check failed: SVT did not significantly beat zero-fill baseline."
        )
    if not result.history["obs_rmse"][-1] <= result.history["obs_rmse"][0]:
        raise RuntimeError("Convergence check failed: observed RMSE did not decrease.")

    print("Quality checks passed.")


if __name__ == "__main__":
    main()

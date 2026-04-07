"""ADMM MVP: solve LASSO via variable splitting.

Problem:
    min_x 0.5 * ||A x - b||_2^2 + lam * ||x||_1

Rewrite with x = z:
    min_{x,z} 0.5 * ||A x - b||_2^2 + lam * ||z||_1
    s.t. x - z = 0

Scaled-form ADMM updates:
    x^{k+1} = argmin_x 0.5||A x - b||^2 + (rho/2)||x - z^k + u^k||^2
    z^{k+1} = S_{lam/rho}(x^{k+1} + u^k)
    u^{k+1} = u^k + x^{k+1} - z^{k+1}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class AdmmResult:
    x: np.ndarray
    z: np.ndarray
    u: np.ndarray
    history: List[Dict[str, float]]
    converged: bool
    iterations: int


def soft_threshold(v: np.ndarray, kappa: float) -> np.ndarray:
    """Elementwise soft-thresholding operator S_kappa(v)."""
    return np.sign(v) * np.maximum(np.abs(v) - kappa, 0.0)


def solve_lasso_admm(
    A: np.ndarray,
    b: np.ndarray,
    lam: float,
    rho: float = 1.0,
    max_iters: int = 1000,
    abs_tol: float = 1e-4,
    rel_tol: float = 1e-3,
) -> AdmmResult:
    """Solve LASSO using scaled ADMM.

    Returns z as the sparse primal estimate.
    """
    m, n = A.shape
    x = np.zeros(n)
    z = np.zeros(n)
    u = np.zeros(n)

    AtA = A.T @ A
    Atb = A.T @ b
    eye_n = np.eye(n)

    # The x-update is a ridge-regularized least squares solve.
    # We factor once because rho is fixed across iterations.
    L = np.linalg.cholesky(AtA + rho * eye_n)

    history: List[Dict[str, float]] = []
    converged = False

    for k in range(1, max_iters + 1):
        q = Atb + rho * (z - u)
        y = np.linalg.solve(L, q)
        x = np.linalg.solve(L.T, y)

        z_old = z.copy()
        x_plus_u = x + u
        z = soft_threshold(x_plus_u, lam / rho)

        u = u + x - z

        primal_res = np.linalg.norm(x - z)
        dual_res = np.linalg.norm(rho * (z - z_old))

        eps_pri = np.sqrt(n) * abs_tol + rel_tol * max(np.linalg.norm(x), np.linalg.norm(z))
        eps_dual = np.sqrt(n) * abs_tol + rel_tol * np.linalg.norm(rho * u)

        obj = 0.5 * np.linalg.norm(A @ x - b) ** 2 + lam * np.linalg.norm(z, 1)
        history.append(
            {
                "iter": float(k),
                "objective": float(obj),
                "primal_res": float(primal_res),
                "dual_res": float(dual_res),
                "eps_pri": float(eps_pri),
                "eps_dual": float(eps_dual),
            }
        )

        if primal_res <= eps_pri and dual_res <= eps_dual:
            converged = True
            break

    return AdmmResult(x=x, z=z, u=u, history=history, converged=converged, iterations=len(history))


def build_synthetic_problem(
    seed: int = 42,
    m: int = 120,
    n: int = 60,
    sparsity: int = 8,
    noise_std: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a reproducible sparse linear inverse problem."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(m, n)) / np.sqrt(m)

    x_true = np.zeros(n)
    support = rng.choice(n, size=sparsity, replace=False)
    x_true[support] = rng.normal(loc=0.0, scale=2.0, size=sparsity)

    noise = noise_std * rng.normal(size=m)
    b = A @ x_true + noise
    return A, b, x_true


def support_set(v: np.ndarray, tol: float = 1e-3) -> np.ndarray:
    return np.flatnonzero(np.abs(v) > tol)


def main() -> None:
    A, b, x_true = build_synthetic_problem()

    lam = 0.12
    rho = 1.0
    result = solve_lasso_admm(A, b, lam=lam, rho=rho, max_iters=1500)

    x_est = result.z
    mse = float(np.mean((x_est - x_true) ** 2))

    true_supp = support_set(x_true)
    pred_supp = support_set(x_est)
    overlap = np.intersect1d(true_supp, pred_supp)

    first_obj = result.history[0]["objective"]
    last_obj = result.history[-1]["objective"]
    last_primal = result.history[-1]["primal_res"]
    last_dual = result.history[-1]["dual_res"]

    print("=== ADMM LASSO MVP ===")
    print(f"shape(A): {A.shape}, lambda={lam:.3f}, rho={rho:.3f}")
    print(f"iterations: {result.iterations}, converged: {result.converged}")
    print(f"objective: {first_obj:.6f} -> {last_obj:.6f}")
    print(f"residuals: primal={last_primal:.3e}, dual={last_dual:.3e}")
    print(f"MSE to ground truth: {mse:.6f}")
    print(f"true support size: {true_supp.size}, predicted support size: {pred_supp.size}")
    print(f"support overlap: {overlap.size}/{true_supp.size}")

    topk = np.argsort(np.abs(x_est))[-10:][::-1]
    print("Top-10 |x_est| entries (index: value):")
    for idx in topk:
        print(f"  {idx:2d}: {x_est[idx]: .6f}")


if __name__ == "__main__":
    main()

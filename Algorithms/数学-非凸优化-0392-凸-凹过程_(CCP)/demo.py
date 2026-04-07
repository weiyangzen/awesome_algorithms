"""Minimal runnable MVP for CCP (Convex-Concave Procedure).

This demo solves a nonconvex L1-L2 regularized regression problem:

    min_x  (1/(2n)) * ||Ax - b||_2^2 + lambda1 * ||x||_1 - lambda2 * ||x||_2

Write objective as g(x) - h(x):
- g(x) = (1/(2n))||Ax-b||^2 + lambda1||x||_1 (convex)
- h(x) = lambda2||x||_2 (convex)

CCP linearizes h at current iterate x_k, then solves a convex subproblem.
The convex subproblem is optimized by a hand-written ISTA loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class CCPResult:
    x: np.ndarray
    history: List[Dict[str, float]]
    converged: bool


def validate_inputs(
    a: np.ndarray,
    b: np.ndarray,
    lambda1: float,
    lambda2: float,
    outer_max_iter: int,
    inner_max_iter: int,
    outer_tol: float,
    inner_tol: float,
) -> None:
    if a.ndim != 2:
        raise ValueError("A must be a 2D array")
    if b.ndim != 1:
        raise ValueError("b must be a 1D array")
    if a.shape[0] != b.shape[0]:
        raise ValueError("A and b have incompatible shapes")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("A and b must contain only finite values")
    if lambda1 <= 0.0 or lambda2 < 0.0:
        raise ValueError("Require lambda1 > 0 and lambda2 >= 0")
    # lambda1 > lambda2 makes the regularizer coercive in this setup.
    if lambda1 <= lambda2:
        raise ValueError("Require lambda1 > lambda2 for a well-behaved MVP")
    if outer_max_iter <= 0 or inner_max_iter <= 0:
        raise ValueError("Iteration limits must be positive")
    if outer_tol <= 0.0 or inner_tol <= 0.0:
        raise ValueError("Tolerances must be positive")


def soft_threshold(x: np.ndarray, tau: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


def objective(
    a: np.ndarray,
    b: np.ndarray,
    x: np.ndarray,
    lambda1: float,
    lambda2: float,
) -> float:
    n_samples = a.shape[0]
    residual = a @ x - b
    loss = 0.5 * np.dot(residual, residual) / n_samples
    reg = lambda1 * np.linalg.norm(x, 1) - lambda2 * np.linalg.norm(x, 2)
    return float(loss + reg)


def convex_subproblem_ista(
    a: np.ndarray,
    b: np.ndarray,
    lambda1: float,
    linear_term: np.ndarray,
    x0: np.ndarray,
    lipschitz: float,
    max_iter: int,
    tol: float,
) -> Tuple[np.ndarray, int, float]:
    """Solve

    min_x (1/(2n))||Ax-b||^2 + lambda1||x||_1 - linear_term^T x

    with proximal gradient (ISTA).
    """
    step = 1.0 / lipschitz
    x = x0.copy()
    n_samples = a.shape[0]

    for it in range(1, max_iter + 1):
        grad_smooth = (a.T @ (a @ x - b)) / n_samples - linear_term
        x_next = soft_threshold(x - step * grad_smooth, lambda1 * step)
        dx = float(np.linalg.norm(x_next - x, 2))
        x = x_next
        if dx < tol:
            return x, it, dx

    return x, max_iter, dx


def ccp_l1_minus_l2(
    a: np.ndarray,
    b: np.ndarray,
    lambda1: float,
    lambda2: float,
    outer_max_iter: int = 40,
    inner_max_iter: int = 400,
    outer_tol: float = 1e-6,
    inner_tol: float = 1e-7,
) -> CCPResult:
    validate_inputs(a, b, lambda1, lambda2, outer_max_iter, inner_max_iter, outer_tol, inner_tol)

    n_features = a.shape[1]
    x = np.zeros(n_features, dtype=float)
    history: List[Dict[str, float]] = []

    # L = largest eigenvalue of (A^T A / n)
    lipschitz = (np.linalg.norm(a, 2) ** 2) / a.shape[0]
    eps = 1e-12
    converged = False

    for outer_it in range(1, outer_max_iter + 1):
        norm_x = np.linalg.norm(x, 2)
        u = x / max(norm_x, eps)
        linear_term = lambda2 * u

        x_new, inner_used, inner_dx = convex_subproblem_ista(
            a=a,
            b=b,
            lambda1=lambda1,
            linear_term=linear_term,
            x0=x,
            lipschitz=lipschitz,
            max_iter=inner_max_iter,
            tol=inner_tol,
        )

        outer_dx = float(np.linalg.norm(x_new - x, 2))
        x = x_new
        obj = objective(a, b, x, lambda1, lambda2)
        nnz = int(np.sum(np.abs(x) > 1e-8))

        history.append(
            {
                "outer_iter": float(outer_it),
                "objective": obj,
                "outer_dx": outer_dx,
                "inner_iters": float(inner_used),
                "inner_dx": float(inner_dx),
                "nnz": float(nnz),
            }
        )

        if outer_dx < outer_tol:
            converged = True
            break

    return CCPResult(x=x, history=history, converged=converged)


def make_synthetic_regression(
    n_samples: int = 140,
    n_features: int = 50,
    sparsity: int = 8,
    noise_std: float = 0.10,
    seed: int = 7,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    a = rng.normal(size=(n_samples, n_features))
    a = (a - a.mean(axis=0, keepdims=True)) / (a.std(axis=0, keepdims=True) + 1e-12)

    true_x = np.zeros(n_features, dtype=float)
    support = rng.choice(n_features, size=sparsity, replace=False)
    true_x[support] = rng.uniform(1.0, 2.5, size=sparsity) * rng.choice([-1.0, 1.0], size=sparsity)

    b = a @ true_x + noise_std * rng.normal(size=n_samples)
    return a, b, true_x


def support_metrics(true_x: np.ndarray, est_x: np.ndarray, tol: float = 1e-4) -> Tuple[float, float]:
    true_support = set(np.where(np.abs(true_x) > tol)[0].tolist())
    est_support = set(np.where(np.abs(est_x) > tol)[0].tolist())

    if not est_support:
        precision = 1.0 if not true_support else 0.0
    else:
        precision = len(true_support & est_support) / len(est_support)

    if not true_support:
        recall = 1.0
    else:
        recall = len(true_support & est_support) / len(true_support)

    return precision, recall


def objective_is_almost_monotone(history: List[Dict[str, float]], slack: float = 1e-8) -> bool:
    values = [h["objective"] for h in history]
    for i in range(1, len(values)):
        if values[i] > values[i - 1] + slack:
            return False
    return True


def print_history(history: List[Dict[str, float]], stride: int = 5) -> None:
    print("iter | objective     | outer_dx      | inner_iters | nnz")
    print("-----+---------------+---------------+------------+-----")
    for idx, row in enumerate(history):
        if idx % stride == 0 or idx == len(history) - 1:
            print(
                f"{int(row['outer_iter']):4d} | "
                f"{row['objective']:13.6f} | "
                f"{row['outer_dx']:13.3e} | "
                f"{int(row['inner_iters']):10d} | "
                f"{int(row['nnz']):3d}"
            )


def run_case(
    a: np.ndarray,
    b: np.ndarray,
    true_x: np.ndarray,
    lambda1: float,
    lambda2: float,
) -> None:
    print("=" * 78)
    print(f"Case: lambda1={lambda1:.3f}, lambda2={lambda2:.3f} (nonconvex, CCP)")

    result = ccp_l1_minus_l2(
        a=a,
        b=b,
        lambda1=lambda1,
        lambda2=lambda2,
        outer_max_iter=45,
        inner_max_iter=500,
        outer_tol=1e-6,
        inner_tol=1e-7,
    )

    x_hat = result.x
    mse = float(np.mean((a @ x_hat - b) ** 2))
    l2_err = float(np.linalg.norm(x_hat - true_x, 2))
    precision, recall = support_metrics(true_x, x_hat)

    print_history(result.history, stride=4)
    print(f"converged: {result.converged}")
    print(f"outer iterations used: {len(result.history)}")
    print(f"final objective: {result.history[-1]['objective']:.6f}")
    print(f"train mse: {mse:.6f}")
    print(f"coef L2 error: {l2_err:.6f}")
    print(f"support precision: {precision:.3f}, recall: {recall:.3f}")
    print(f"objective almost monotone: {objective_is_almost_monotone(result.history)}")

    nz_idx = np.where(np.abs(x_hat) > 1e-4)[0]
    print(f"estimated nnz: {nz_idx.size}, first active indices: {nz_idx[:12].tolist()}")


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)

    a, b, true_x = make_synthetic_regression()
    print("CCP demo for nonconvex L1-L2 regression")
    print(f"data shape: A={a.shape}, b={b.shape}, true nnz={int(np.sum(np.abs(true_x) > 0))}")

    # Keep lambda1 > lambda2 for stable regularization in this MVP.
    run_case(a, b, true_x, lambda1=0.24, lambda2=0.10)
    run_case(a, b, true_x, lambda1=0.30, lambda2=0.12)


if __name__ == "__main__":
    main()

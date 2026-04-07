"""Proximal Gradient Method (ISTA) MVP for MATH-0379.

Problem solved:
    min_x  (1 / (2m)) * ||Xx - y||_2^2 + lam * ||x||_1

This is a convex composite objective:
    f(x) = (1 / (2m)) * ||Xx - y||_2^2   (smooth convex)
    g(x) = lam * ||x||_1                 (non-smooth convex)

ISTA update:
    x_{k+1} = prox_{step*g}(x_k - step * grad f(x_k))
            = soft_threshold(x_k - step * grad f(x_k), lam * step)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

HistoryItem = Tuple[int, float, float, float]


@dataclass
class IstaResult:
    """Container for ISTA outputs."""

    x: np.ndarray
    history: List[HistoryItem]
    converged: bool
    iterations: int
    step_size: float
    lipschitz: float


def validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    step_scale: float,
    max_iters: int,
    tol: float,
) -> None:
    """Validate shape, finiteness, and scalar hyper-parameters."""
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape={X.shape}.")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D, got shape={y.shape}.")
    m, n = X.shape
    if m <= 0 or n <= 0:
        raise ValueError("X must have positive shape.")
    if y.shape[0] != m:
        raise ValueError(f"y length mismatch: len(y)={y.shape[0]}, expected {m}.")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("X or y contains non-finite values.")
    if lam < 0.0:
        raise ValueError("lam must be >= 0.")
    if not (0.0 < step_scale <= 1.0):
        raise ValueError("step_scale must be in (0, 1].")
    if max_iters <= 0:
        raise ValueError("max_iters must be > 0.")
    if tol <= 0.0:
        raise ValueError("tol must be > 0.")


def soft_threshold(v: np.ndarray, tau: float) -> np.ndarray:
    """Elementwise proximal operator of tau * ||x||_1."""
    return np.sign(v) * np.maximum(np.abs(v) - tau, 0.0)


def objective_lasso(X: np.ndarray, y: np.ndarray, x: np.ndarray, lam: float) -> float:
    """Compute (1/(2m))*||Xx-y||^2 + lam*||x||_1."""
    m = X.shape[0]
    residual = X @ x - y
    return 0.5 * float(residual.T @ residual) / float(m) + lam * float(np.sum(np.abs(x)))


def grad_smooth(X: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Compute gradient of smooth term: X^T(Xx-y)/m."""
    m = X.shape[0]
    residual = X @ x - y
    return (X.T @ residual) / float(m)


def lipschitz_constant(X: np.ndarray) -> float:
    """Lipschitz constant of grad_smooth: ||X||_2^2 / m."""
    m = X.shape[0]
    spectral_norm = float(np.linalg.norm(X, ord=2))
    return (spectral_norm * spectral_norm) / float(m)


def proximal_gradient_mapping_norm(
    x: np.ndarray,
    grad: np.ndarray,
    step: float,
    lam: float,
) -> float:
    """Compute ||G_step(x)|| where G_step(x)=(x-prox(x-step*grad))/step."""
    prox_point = soft_threshold(x - step * grad, lam * step)
    mapping = (x - prox_point) / step
    return float(np.linalg.norm(mapping))


def ista_lasso(
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    step_scale: float = 0.98,
    max_iters: int = 6000,
    tol: float = 1e-9,
) -> IstaResult:
    """Run ISTA for Lasso with fixed step size step_scale / L."""
    validate_inputs(
        X=X,
        y=y,
        lam=lam,
        step_scale=step_scale,
        max_iters=max_iters,
        tol=tol,
    )

    _, n = X.shape
    L = lipschitz_constant(X)
    if not np.isfinite(L) or L <= 0.0:
        raise ValueError(f"Invalid Lipschitz constant L={L}.")
    step = step_scale / L

    x = np.zeros(n, dtype=float)
    history: List[HistoryItem] = []
    converged = False

    for k in range(1, max_iters + 1):
        grad = grad_smooth(X, y, x)
        x_next = soft_threshold(x - step * grad, lam * step)

        obj_next = objective_lasso(X, y, x_next, lam)
        step_norm = float(np.linalg.norm(x_next - x))
        pg_norm = proximal_gradient_mapping_norm(x, grad, step, lam)
        history.append((k, obj_next, step_norm, pg_norm))

        if not np.isfinite(obj_next) or not np.isfinite(step_norm) or not np.isfinite(pg_norm):
            raise RuntimeError("Encountered non-finite values during ISTA iterations.")

        if step_norm <= tol * (1.0 + float(np.linalg.norm(x_next))) and pg_norm <= 5.0 * tol:
            converged = True
            x = x_next
            break

        x = x_next

    return IstaResult(
        x=x,
        history=history,
        converged=converged,
        iterations=len(history),
        step_size=step,
        lipschitz=L,
    )


def build_orthonormal_lasso_case(
    seed: int = 379,
    m: int = 120,
    n: int = 40,
    sparsity: int = 8,
    noise_std: float = 0.03,
    lam: float = 0.12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Create a deterministic Lasso case with (X^T X)/m = I.

    With this design, the closed-form optimum is:
        x_star = soft_threshold((X^T y)/m, lam)
    """
    if not (0 < sparsity < n):
        raise ValueError("sparsity must satisfy 0 < sparsity < n.")
    if noise_std < 0.0:
        raise ValueError("noise_std must be >= 0.")
    if m < n:
        raise ValueError("Need m >= n for reduced QR with orthonormal columns.")

    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(m, n))
    q, _ = np.linalg.qr(raw, mode="reduced")
    X = np.sqrt(float(m)) * q

    x_true = np.zeros(n, dtype=float)
    support = rng.choice(n, size=sparsity, replace=False)
    x_true[support] = rng.normal(loc=0.0, scale=1.6, size=sparsity)

    noise = noise_std * rng.normal(size=m)
    y = X @ x_true + noise

    x_star_closed_form = soft_threshold((X.T @ y) / float(m), lam)
    return X, y, x_true, x_star_closed_form, lam


def objective_monotone_violations(history: Sequence[HistoryItem], eps: float = 1e-11) -> int:
    """Count objective increases larger than eps."""
    if not history:
        return 0
    objs = np.asarray([item[1] for item in history], dtype=float)
    diffs = np.diff(objs)
    return int(np.sum(diffs > eps))


def support_set(x: np.ndarray, threshold: float = 1e-4) -> np.ndarray:
    """Return indices of entries with magnitude above threshold."""
    return np.flatnonzero(np.abs(x) > threshold)


def print_history(history: Sequence[HistoryItem], max_lines: int = 10) -> None:
    """Print the first few records and the final record."""
    print("iter | objective        | ||dx||            | ||G_step(x)||")
    print("-" * 68)
    for k, obj, dx, pg in history[:max_lines]:
        print(f"{k:4d} | {obj:16.9e} | {dx:16.9e} | {pg:16.9e}")
    if len(history) > max_lines:
        last_k, last_obj, last_dx, last_pg = history[-1]
        print(f"... ({len(history) - max_lines} iterations omitted)")
        print(f"{last_k:4d} | {last_obj:16.9e} | {last_dx:16.9e} | {last_pg:16.9e}  <- last")


def main() -> None:
    print("Proximal Gradient Method MVP (MATH-0379)")
    print("Objective: (1/(2m))*||Xx-y||^2 + lam*||x||_1")
    print("=" * 72)

    X, y, x_true, x_star, lam = build_orthonormal_lasso_case()
    result = ista_lasso(X, y, lam=lam, step_scale=0.98, max_iters=6000, tol=1e-9)
    x_est = result.x

    print(f"shape(X): {X.shape}")
    print(f"lambda: {lam:.4f}")
    print(f"Lipschitz L: {result.lipschitz:.6f}")
    print(f"step size: {result.step_size:.6f}")
    print(f"iterations: {result.iterations}, converged: {result.converged}")

    print_history(result.history, max_lines=8)

    obj_est = objective_lasso(X, y, x_est, lam)
    obj_star = objective_lasso(X, y, x_star, lam)
    abs_gap = obj_est - obj_star
    rel_gap = abs_gap / (1.0 + abs(obj_star))
    l2_err_to_closed = float(np.linalg.norm(x_est - x_star))
    l2_err_to_true = float(np.linalg.norm(x_est - x_true))

    monotone_violations = objective_monotone_violations(result.history)
    supp_true = support_set(x_true)
    supp_est = support_set(x_est)
    overlap = np.intersect1d(supp_true, supp_est)

    print("\n=== Metrics ===")
    print(f"objective(ista): {obj_est:.9e}")
    print(f"objective(closed form): {obj_star:.9e}")
    print(f"absolute objective gap: {abs_gap:.9e}")
    print(f"relative objective gap: {rel_gap:.9e}")
    print(f"||x_ista - x_closed||_2: {l2_err_to_closed:.9e}")
    print(f"||x_ista - x_true||_2: {l2_err_to_true:.9e}")
    print(f"objective monotone violations: {monotone_violations}")
    print(f"true support size: {supp_true.size}")
    print(f"estimated support size: {supp_est.size}")
    print(f"support overlap: {overlap.size}/{supp_true.size}")

    if rel_gap > 1e-8:
        raise RuntimeError(f"Relative objective gap too large: {rel_gap:.3e} > 1e-8")
    if l2_err_to_closed > 2e-6:
        raise RuntimeError(
            f"Distance to closed-form solution too large: {l2_err_to_closed:.3e} > 2e-6"
        )
    if monotone_violations > 0:
        raise RuntimeError("Objective should be monotone for fixed step <= 1/L.")
    if not result.converged:
        raise RuntimeError("ISTA did not satisfy stopping criteria.")

    print("All checks passed.")


if __name__ == "__main__":
    main()

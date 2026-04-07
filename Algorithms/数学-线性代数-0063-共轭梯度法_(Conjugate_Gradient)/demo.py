"""Minimal runnable MVP for Conjugate Gradient (CG)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class CGResult:
    solution: np.ndarray
    iterations: int
    converged: bool
    residual_norm: float
    relative_residual: float
    residual_history: List[float]


def validate_inputs(
    a: np.ndarray,
    b: np.ndarray,
    x0: Optional[np.ndarray],
    max_iters: int,
    tol: float,
) -> None:
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("A must be a square matrix.")
    if b.ndim != 1 or b.shape[0] != a.shape[0]:
        raise ValueError("b must be a 1D vector with length equal to A.shape[0].")
    if x0 is not None and (x0.ndim != 1 or x0.shape[0] != a.shape[0]):
        raise ValueError("x0 must be None or a 1D vector with length equal to A.shape[0].")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("A and b must contain only finite values.")
    if x0 is not None and not np.isfinite(x0).all():
        raise ValueError("x0 must contain only finite values.")
    if max_iters <= 0:
        raise ValueError("max_iters must be positive.")
    if tol <= 0.0:
        raise ValueError("tol must be positive.")

    if not np.allclose(a, a.T, rtol=1e-12, atol=1e-12):
        raise ValueError("A must be symmetric for classical CG.")

    try:
        # Cholesky succeeds iff A is SPD (for finite symmetric matrices).
        np.linalg.cholesky(a)
    except np.linalg.LinAlgError as exc:
        raise ValueError("A must be symmetric positive definite (SPD).") from exc


def conjugate_gradient(
    a: np.ndarray,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    max_iters: Optional[int] = None,
    tol: float = 1e-10,
) -> CGResult:
    """
    Solve Ax=b using the classical Conjugate Gradient method.

    Stopping rule: ||r_k|| / max(||b||, 1) <= tol
    """
    n = a.shape[0]
    if max_iters is None:
        max_iters = 5 * n

    validate_inputs(a=a, b=b, x0=x0, max_iters=max_iters, tol=tol)

    x = np.zeros(n, dtype=float) if x0 is None else x0.astype(float, copy=True)
    r = b - a @ x
    p = r.copy()
    rs_old = float(np.dot(r, r))

    b_norm = float(np.linalg.norm(b))
    denom = b_norm if b_norm > 0.0 else 1.0

    residual = float(np.sqrt(rs_old))
    relative_residual = residual / denom
    residual_history: List[float] = [residual]

    if relative_residual <= tol:
        return CGResult(
            solution=x,
            iterations=0,
            converged=True,
            residual_norm=residual,
            relative_residual=relative_residual,
            residual_history=residual_history,
        )

    for it in range(1, max_iters + 1):
        ap = a @ p
        p_ap = float(np.dot(p, ap))
        if p_ap <= 0.0 or not np.isfinite(p_ap):
            raise RuntimeError(
                "Encountered non-positive curvature p^T A p; matrix may not be SPD numerically."
            )

        alpha = rs_old / p_ap
        x = x + alpha * p
        r = r - alpha * ap

        rs_new = float(np.dot(r, r))
        if rs_new < 0.0 or not np.isfinite(rs_new):
            raise RuntimeError("Encountered invalid residual norm squared during CG iteration.")

        residual = float(np.sqrt(rs_new))
        relative_residual = residual / denom
        residual_history.append(residual)

        if relative_residual <= tol:
            return CGResult(
                solution=x,
                iterations=it,
                converged=True,
                residual_norm=residual,
                relative_residual=relative_residual,
                residual_history=residual_history,
            )

        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    return CGResult(
        solution=x,
        iterations=max_iters,
        converged=False,
        residual_norm=residual,
        relative_residual=relative_residual,
        residual_history=residual_history,
    )


def build_demo_system(n: int = 6, seed: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a deterministic SPD system Ax=b with known ground-truth x."""
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((n, n))
    a = m.T @ m + 0.5 * np.eye(n, dtype=float)

    x_true = rng.standard_normal(n)
    b = a @ x_true
    return a, b, x_true


def run_checks(a: np.ndarray, b: np.ndarray, x_true: np.ndarray, result: CGResult) -> None:
    if not result.converged:
        raise AssertionError("CG did not converge within max_iters.")

    x_direct = np.linalg.solve(a, b)
    direct_err = float(np.linalg.norm(result.solution - x_direct))
    true_err = float(np.linalg.norm(result.solution - x_true))

    if result.residual_norm > 1e-8:
        raise AssertionError(f"Residual too large: {result.residual_norm:.3e}")
    if direct_err > 1e-8:
        raise AssertionError(f"Deviation from direct solve too large: {direct_err:.3e}")
    if true_err > 1e-8:
        raise AssertionError(f"Deviation from planted solution too large: {true_err:.3e}")

    if not np.isfinite(result.solution).all():
        raise AssertionError("Solution contains non-finite values.")
    if len(result.residual_history) != result.iterations + 1:
        raise AssertionError("Residual history length does not match iteration count.")
    if result.residual_history[-1] > result.residual_history[0]:
        raise AssertionError("Residual did not decrease from the initial value.")


def main() -> None:
    a, b, x_true = build_demo_system(n=6, seed=7)
    result = conjugate_gradient(a=a, b=b, x0=None, max_iters=60, tol=1e-12)
    run_checks(a=a, b=b, x_true=x_true, result=result)

    x_direct = np.linalg.solve(a, b)
    direct_err = float(np.linalg.norm(result.solution - x_direct))
    true_err = float(np.linalg.norm(result.solution - x_true))
    cond_a = float(np.linalg.cond(a))

    print("Conjugate Gradient demo")
    print(f"matrix_shape={a.shape}")
    print(f"iterations={result.iterations}")
    print(f"converged={result.converged}")
    print(f"condition_number={cond_a:.6f}")
    print(f"residual_norm={result.residual_norm:.3e}")
    print(f"relative_residual={result.relative_residual:.3e}")
    print(f"error_vs_direct={direct_err:.3e}")
    print(f"error_vs_true={true_err:.3e}")

    head = result.residual_history[:5]
    tail = result.residual_history[-3:]
    print(f"residual_history_head={[f'{v:.3e}' for v in head]}")
    print(f"residual_history_tail={[f'{v:.3e}' for v in tail]}")
    print("All checks passed.")


if __name__ == "__main__":
    main()

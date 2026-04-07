"""Minimal runnable MVP for Successive Over-Relaxation (SOR)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class SORResult:
    solution: np.ndarray
    residual_norm: float
    iterations: int
    converged: bool
    omega: float
    residual_history: List[float]
    step_history: List[float]


def validate_inputs(a: np.ndarray, b: np.ndarray, omega: float, max_iters: int, tol: float) -> None:
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Input matrix A must be square.")
    if b.ndim != 1 or b.shape[0] != a.shape[0]:
        raise ValueError("Input vector b must be one-dimensional with length equal to A.shape[0].")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Input A and b must contain only finite values.")
    if np.any(np.isclose(np.diag(a), 0.0)):
        raise ValueError("SOR requires all diagonal entries of A to be non-zero.")
    if not (0.0 < omega < 2.0):
        raise ValueError("omega must satisfy 0 < omega < 2.")
    if max_iters <= 0:
        raise ValueError("max_iters must be positive.")
    if tol <= 0.0:
        raise ValueError("tol must be positive.")


def sor(
    a: np.ndarray,
    b: np.ndarray,
    omega: float = 1.1,
    max_iters: int = 500,
    tol: float = 1e-10,
    x0: np.ndarray | None = None,
) -> SORResult:
    """Solve Ax=b by SOR iteration with in-place Gauss-Seidel style updates."""
    validate_inputs(a, b, omega=omega, max_iters=max_iters, tol=tol)

    n = a.shape[0]
    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.asarray(x0, dtype=float).copy()
        if x.shape != (n,):
            raise ValueError("x0 must have shape (n,).")
        if not np.isfinite(x).all():
            raise ValueError("x0 must contain only finite values.")

    residual_history: List[float] = []
    step_history: List[float] = []

    residual = float(np.linalg.norm(a @ x - b, ord=np.inf))
    if residual < tol:
        return SORResult(
            solution=x,
            residual_norm=residual,
            iterations=0,
            converged=True,
            omega=omega,
            residual_history=[residual],
            step_history=[0.0],
        )

    for it in range(1, max_iters + 1):
        x_old = x.copy()

        for i in range(n):
            lower_sum = float(np.dot(a[i, :i], x[:i]))
            upper_sum = float(np.dot(a[i, i + 1 :], x_old[i + 1 :]))

            gs_value = (b[i] - lower_sum - upper_sum) / a[i, i]
            x[i] = (1.0 - omega) * x_old[i] + omega * gs_value

        residual = float(np.linalg.norm(a @ x - b, ord=np.inf))
        step = float(np.linalg.norm(x - x_old, ord=np.inf))
        residual_history.append(residual)
        step_history.append(step)

        if residual < tol and step < tol:
            return SORResult(
                solution=x,
                residual_norm=residual,
                iterations=it,
                converged=True,
                omega=omega,
                residual_history=residual_history,
                step_history=step_history,
            )

    return SORResult(
        solution=x,
        residual_norm=residual,
        iterations=max_iters,
        converged=False,
        omega=omega,
        residual_history=residual_history,
        step_history=step_history,
    )


def run_checks(result: SORResult, a: np.ndarray, b: np.ndarray, x_ref: np.ndarray) -> None:
    if not result.converged:
        raise AssertionError("SOR did not converge within max_iters.")

    if not np.isfinite(result.solution).all():
        raise AssertionError("Estimated solution contains non-finite values.")

    residual = float(np.linalg.norm(a @ result.solution - b, ord=np.inf))
    if residual > 1e-9:
        raise AssertionError(f"Residual too large: {residual:.3e}")

    error_inf = float(np.linalg.norm(result.solution - x_ref, ord=np.inf))
    if error_inf > 1e-8:
        raise AssertionError(f"Infinity-norm error too large: {error_inf:.3e}")


def main() -> None:
    # SPD tridiagonal system. With 0 < omega < 2, SOR is guaranteed to converge.
    a = np.array(
        [
            [4.0, -1.0, 0.0, 0.0],
            [-1.0, 4.0, -1.0, 0.0],
            [0.0, -1.0, 4.0, -1.0],
            [0.0, 0.0, -1.0, 3.0],
        ],
        dtype=float,
    )
    x_true = np.array([5.0, 5.0, 5.0, 5.0], dtype=float)
    b = a @ x_true

    omega = 1.1
    result = sor(a, b, omega=omega, max_iters=400, tol=1e-12)

    x_ref = np.linalg.solve(a, b)
    run_checks(result, a, b, x_ref)

    err_inf = float(np.linalg.norm(result.solution - x_ref, ord=np.inf))

    print("SOR demo")
    print(f"matrix_shape={a.shape}")
    print(f"omega={result.omega:.2f}")
    print(f"iterations={result.iterations}")
    print(f"converged={result.converged}")
    print(f"residual_inf={result.residual_norm:.3e}")
    print(f"solution_inf_error_vs_numpy={err_inf:.3e}")
    print(f"x_hat={np.array2string(result.solution, precision=10)}")

    head = result.residual_history[:5]
    tail = result.residual_history[-3:]
    print(f"residual_head={[f'{v:.3e}' for v in head]}")
    print(f"residual_tail={[f'{v:.3e}' for v in tail]}")

    print("omega sweep (iteration count comparison)")
    for w in [0.8, 1.0, 1.1, 1.3, 1.6]:
        trial = sor(a, b, omega=w, max_iters=400, tol=1e-10)
        print(
            f"  omega={w:.1f} iterations={trial.iterations:3d} "
            f"converged={trial.converged} residual_inf={trial.residual_norm:.3e}"
        )

    print("All checks passed.")


if __name__ == "__main__":
    main()

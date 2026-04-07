"""Minimal runnable MVP for solving a continuous-time algebraic Riccati equation.

The demo implements a Newton-Kleinman style iteration and solves each Lyapunov
subproblem via Kronecker linearization with NumPy only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class RiccatiResult:
    X: np.ndarray
    converged: bool
    iterations: int
    residual_norm: float
    step_norm: float
    residual_history: List[float]
    step_history: List[float]


def validate_inputs(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> None:
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")
    n = A.shape[0]
    if B.ndim != 2 or B.shape[0] != n:
        raise ValueError("B must be a 2D matrix with the same row count as A.")
    m = B.shape[1]
    if Q.shape != (n, n):
        raise ValueError("Q must have shape (n, n).")
    if R.shape != (m, m):
        raise ValueError("R must have shape (m, m).")

    for name, mat in (("A", A), ("B", B), ("Q", Q), ("R", R)):
        if not np.all(np.isfinite(mat)):
            raise ValueError(f"{name} contains non-finite values.")

    if not np.allclose(Q, Q.T, atol=1e-12):
        raise ValueError("Q must be symmetric.")
    if not np.allclose(R, R.T, atol=1e-12):
        raise ValueError("R must be symmetric.")

    # Cholesky factorization is a practical check for positive definiteness.
    try:
        np.linalg.cholesky(R)
    except np.linalg.LinAlgError as exc:
        raise ValueError("R must be positive definite.") from exc


def solve_continuous_lyapunov_via_kron(A: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Solve A^T X + X A + C = 0 via vec/Kronecker linearization."""
    n = A.shape[0]
    K = np.kron(np.eye(n), A.T) + np.kron(A.T, np.eye(n))
    rhs = -C.reshape(n * n, order="F")
    vec_x = np.linalg.solve(K, rhs)
    X = vec_x.reshape(n, n, order="F")
    return 0.5 * (X + X.T)


def riccati_residual(A: np.ndarray, G: np.ndarray, Q: np.ndarray, X: np.ndarray) -> np.ndarray:
    return A.T @ X + X @ A - X @ G @ X + Q


def solve_care_newton_kleinman(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    *,
    step_tol: float = 1e-12,
    residual_tol: float = 1e-8,
    max_iter: int = 50,
) -> RiccatiResult:
    """Solve CARE: A^T X + X A - X B R^{-1} B^T X + Q = 0."""
    validate_inputs(A, B, Q, R)

    R_inv = np.linalg.inv(R)
    G = B @ R_inv @ B.T

    # Initial guess from the linearized problem (drop quadratic term).
    X = solve_continuous_lyapunov_via_kron(A, Q)

    residual_history: List[float] = [
        float(np.linalg.norm(riccati_residual(A, G, Q, X), ord="fro"))
    ]
    step_history: List[float] = []

    converged = False
    step_norm = float("inf")

    for iteration in range(1, max_iter + 1):
        A_cl = A - G @ X
        C = X @ G @ X + Q

        X_next = solve_continuous_lyapunov_via_kron(A_cl, C)
        step_norm = float(np.linalg.norm(X_next - X, ord="fro"))
        residual_norm = float(
            np.linalg.norm(riccati_residual(A, G, Q, X_next), ord="fro")
        )

        step_history.append(step_norm)
        residual_history.append(residual_norm)
        X = X_next

        if step_norm <= step_tol * (1.0 + np.linalg.norm(X, ord="fro")) and residual_norm <= residual_tol:
            converged = True
            break

    return RiccatiResult(
        X=X,
        converged=converged,
        iterations=len(step_history),
        residual_norm=residual_history[-1],
        step_norm=step_history[-1] if step_history else 0.0,
        residual_history=residual_history,
        step_history=step_history,
    )


def run_checks(result: RiccatiResult, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> None:
    R_inv = np.linalg.inv(R)
    G = B @ R_inv @ B.T

    if not result.converged:
        raise RuntimeError("Newton-Kleinman did not converge within max_iter.")

    if result.residual_norm > 1e-8:
        raise RuntimeError(f"Residual too large: {result.residual_norm:.3e}")

    if not np.allclose(result.X, result.X.T, atol=1e-10):
        raise RuntimeError("Computed X is not symmetric.")

    eigvals_x = np.linalg.eigvalsh(result.X)
    if np.min(eigvals_x) <= -1e-10:
        raise RuntimeError("Computed X is not positive semidefinite.")

    eigvals_closed = np.linalg.eigvals(A - G @ result.X)
    if np.max(np.real(eigvals_closed)) >= -1e-8:
        raise RuntimeError("Closed-loop matrix is not stable.")

    if result.residual_history[-1] >= result.residual_history[0]:
        raise RuntimeError("Residual did not improve from the initial iterate.")


def main() -> None:
    A = np.array(
        [
            [-1.0, 0.2, 0.0],
            [-0.1, -1.3, 0.3],
            [0.0, -0.4, -0.9],
        ],
        dtype=float,
    )
    B = np.array(
        [
            [1.0, 0.0],
            [0.2, 1.0],
            [0.0, 0.5],
        ],
        dtype=float,
    )
    Q = np.diag([2.0, 1.5, 1.0])
    R = np.array(
        [
            [1.2, 0.1],
            [0.1, 1.0],
        ],
        dtype=float,
    )

    result = solve_care_newton_kleinman(
        A,
        B,
        Q,
        R,
        step_tol=1e-12,
        residual_tol=1e-8,
        max_iter=40,
    )
    run_checks(result, A, B, Q, R)

    R_inv = np.linalg.inv(R)
    G = B @ R_inv @ B.T
    closed_loop_eigs = np.linalg.eigvals(A - G @ result.X)

    print("=== Riccati (CARE) Newton-Kleinman MVP ===")
    print(f"converged: {result.converged}")
    print(f"iterations: {result.iterations}")
    print(f"final step norm (Fro): {result.step_norm:.3e}")
    print(f"final residual norm (Fro): {result.residual_norm:.3e}")
    print(
        "residual history (first/last): "
        f"{result.residual_history[0]:.3e} -> {result.residual_history[-1]:.3e}"
    )
    print("closed-loop eigenvalues:")
    print(closed_loop_eigs)
    print("solution X:")
    print(result.X)
    print("All checks passed.")


if __name__ == "__main__":
    main()

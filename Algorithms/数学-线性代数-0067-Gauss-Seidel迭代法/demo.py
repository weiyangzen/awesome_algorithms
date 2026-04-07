"""Gauss-Seidel iteration MVP demo.

Run:
    python3 demo.py
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


HistoryRecord = Tuple[int, float, float]


def is_strictly_diagonally_dominant(A: np.ndarray) -> bool:
    """Check strict diagonal dominance by rows."""
    diag = np.abs(np.diag(A))
    off_diag = np.sum(np.abs(A), axis=1) - diag
    return bool(np.all(diag > off_diag))


def iteration_matrix_and_spectral_radius(A: np.ndarray) -> Tuple[np.ndarray, float]:
    """Return Gauss-Seidel iteration matrix T and its spectral radius rho(T)."""
    D = np.diag(np.diag(A))
    L = np.tril(A, k=-1)
    U = np.triu(A, k=1)

    T = -np.linalg.solve(D + L, U)
    eigvals = np.linalg.eigvals(T)
    rho = float(np.max(np.abs(eigvals)))
    return T, rho


def gauss_seidel(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray | None = None,
    tol: float = 1e-10,
    max_iter: int = 500,
) -> Tuple[np.ndarray, bool, int, List[HistoryRecord]]:
    """Solve Ax=b with Gauss-Seidel iteration.

    Returns:
        x: approximate solution
        converged: whether convergence criteria were met
        iters: iteration count used
        history: list of (iteration, residual_inf, step_inf)
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")

    n = A.shape[0]
    if b.shape[0] != n:
        raise ValueError("b length must match A dimension")

    if np.any(np.isclose(np.diag(A), 0.0)):
        raise ValueError("A has zero (or near-zero) diagonal entry")

    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.asarray(x0, dtype=float).reshape(-1).copy()
        if x.shape[0] != n:
            raise ValueError("x0 length must match A dimension")

    history: List[HistoryRecord] = []

    for k in range(1, max_iter + 1):
        x_old = x.copy()

        for i in range(n):
            sigma_left = float(np.dot(A[i, :i], x[:i]))
            sigma_right = float(np.dot(A[i, i + 1 :], x_old[i + 1 :]))
            x[i] = (b[i] - sigma_left - sigma_right) / A[i, i]

        residual_inf = float(np.linalg.norm(A @ x - b, ord=np.inf))
        step_inf = float(np.linalg.norm(x - x_old, ord=np.inf))
        history.append((k, residual_inf, step_inf))

        if residual_inf < tol and step_inf < tol:
            return x, True, k, history

    return x, False, max_iter, history


def print_history_tail(history: List[HistoryRecord], tail: int = 5) -> None:
    """Print the last few history records for quick diagnosis."""
    print("iter | residual_inf | step_inf")
    for k, res, step in history[-tail:]:
        print(f"{k:4d} | {res:12.3e} | {step:9.3e}")


def case_strictly_diagonally_dominant() -> None:
    print("=== Case 1: Strictly Diagonally Dominant ===")
    A = np.array(
        [
            [4.0, 1.0, 2.0],
            [3.0, 5.0, 1.0],
            [1.0, 1.0, 3.0],
        ]
    )
    b = np.array([4.0, 7.0, 3.0])

    print(f"strict diagonal dominance: {is_strictly_diagonally_dominant(A)}")
    _, rho = iteration_matrix_and_spectral_radius(A)
    print(f"spectral radius rho(T_GS): {rho:.6f}")

    x, ok, iters, history = gauss_seidel(A, b, tol=1e-12, max_iter=200)
    x_ref = np.linalg.solve(A, b)

    print(f"converged: {ok}, iterations: {iters}")
    print(f"x_gs   = {x}")
    print(f"x_ref  = {x_ref}")
    print(f"||x_gs-x_ref||_inf = {np.linalg.norm(x - x_ref, ord=np.inf):.3e}")
    print_history_tail(history, tail=5)

    assert ok
    assert np.allclose(x, x_ref, atol=1e-9)


def case_spd_tridiagonal() -> None:
    print("\n=== Case 2: SPD Tridiagonal (1D Poisson-like) ===")
    n = 6
    A = 2.0 * np.eye(n)
    A += -1.0 * np.eye(n, k=1)
    A += -1.0 * np.eye(n, k=-1)
    b = np.ones(n)

    print(f"strict diagonal dominance: {is_strictly_diagonally_dominant(A)}")
    _, rho = iteration_matrix_and_spectral_radius(A)
    print(f"spectral radius rho(T_GS): {rho:.6f}")

    x, ok, iters, history = gauss_seidel(A, b, tol=1e-11, max_iter=2000)
    x_ref = np.linalg.solve(A, b)

    print(f"converged: {ok}, iterations: {iters}")
    print(f"x_gs   = {x}")
    print(f"x_ref  = {x_ref}")
    print(f"||Ax-b||_inf = {np.linalg.norm(A @ x - b, ord=np.inf):.3e}")
    print_history_tail(history, tail=5)

    assert ok
    assert np.allclose(x, x_ref, atol=1e-8)


def case_non_convergent() -> None:
    print("\n=== Case 3: Non-convergent Example ===")
    A = np.array(
        [
            [1.0, 3.0],
            [2.0, 1.0],
        ]
    )
    b = np.array([5.0, 5.0])

    print(f"strict diagonal dominance: {is_strictly_diagonally_dominant(A)}")
    _, rho = iteration_matrix_and_spectral_radius(A)
    print(f"spectral radius rho(T_GS): {rho:.6f}")

    x, ok, iters, history = gauss_seidel(A, b, tol=1e-12, max_iter=20)

    print(f"converged within {iters} iterations: {ok}")
    print(f"last iterate x = {x}")
    print(f"last residual_inf = {history[-1][1]:.3e}")
    print_history_tail(history, tail=5)

    assert not ok
    assert rho > 1.0


def main() -> None:
    case_strictly_diagonally_dominant()
    case_spd_tridiagonal()
    case_non_convergent()
    print("\nAll Gauss-Seidel demos finished successfully.")


if __name__ == "__main__":
    main()

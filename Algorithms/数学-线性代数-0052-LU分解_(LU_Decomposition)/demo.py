"""LU decomposition MVP with partial pivoting (manual implementation)."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def validate_square_matrix(a: np.ndarray) -> np.ndarray:
    """Return a finite float square matrix, or raise ValueError."""
    arr = np.asarray(a, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"A must be a square matrix, got shape={arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("A contains non-finite values")
    return arr


def lu_decompose_partial_pivot(
    a: np.ndarray, tol: float = 1e-12
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Compute PA=LU using Doolittle form with partial pivoting.

    Returns:
        P, L, U, swap_count
    """
    if tol <= 0:
        raise ValueError("tol must be > 0")

    mat = validate_square_matrix(a)
    n = mat.shape[0]

    p = np.eye(n, dtype=float)
    l = np.eye(n, dtype=float)
    u = mat.copy()
    swap_count = 0

    for k in range(n):
        pivot_row = k + int(np.argmax(np.abs(u[k:, k])))
        pivot_value = float(u[pivot_row, k])
        if abs(pivot_value) <= tol:
            raise ValueError(f"Matrix is singular to tolerance at column {k}")

        if pivot_row != k:
            u[[k, pivot_row], :] = u[[pivot_row, k], :]
            p[[k, pivot_row], :] = p[[pivot_row, k], :]
            if k > 0:
                l[[k, pivot_row], :k] = l[[pivot_row, k], :k]
            swap_count += 1

        for i in range(k + 1, n):
            multiplier = u[i, k] / u[k, k]
            l[i, k] = multiplier
            u[i, k:] = u[i, k:] - multiplier * u[k, k:]
            u[i, k] = 0.0

    return p, l, u, swap_count


def forward_substitution(l: np.ndarray, b: np.ndarray, tol: float = 1e-15) -> np.ndarray:
    """Solve Ly=b for lower-triangular L."""
    n = l.shape[0]
    y = np.zeros(n, dtype=float)

    for i in range(n):
        diag = float(l[i, i])
        if abs(diag) <= tol:
            raise ValueError(f"Zero diagonal encountered in L at row {i}")
        y[i] = (b[i] - np.dot(l[i, :i], y[:i])) / diag

    return y


def backward_substitution(u: np.ndarray, y: np.ndarray, tol: float = 1e-15) -> np.ndarray:
    """Solve Ux=y for upper-triangular U."""
    n = u.shape[0]
    x = np.zeros(n, dtype=float)

    for i in range(n - 1, -1, -1):
        diag = float(u[i, i])
        if abs(diag) <= tol:
            raise ValueError(f"Zero diagonal encountered in U at row {i}")
        x[i] = (y[i] - np.dot(u[i, i + 1 :], x[i + 1 :])) / diag

    return x


def solve_with_lu(p: np.ndarray, l: np.ndarray, u: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve Ax=b from PA=LU."""
    b_vec = np.asarray(b, dtype=float)
    if b_vec.ndim != 1:
        raise ValueError(f"b must be a vector, got ndim={b_vec.ndim}")
    if p.shape[0] != b_vec.shape[0]:
        raise ValueError("Dimension mismatch between matrix and vector")

    pb = p @ b_vec
    y = forward_substitution(l, pb)
    x = backward_substitution(u, y)
    return x


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)

    # This matrix triggers row pivoting at the first column.
    a = np.array(
        [
            [0.0, 2.0, 1.0, 1.0],
            [1.0, -2.0, -1.0, 2.0],
            [2.0, 3.0, 1.0, 0.0],
            [1.0, 1.0, 2.0, 1.0],
        ],
        dtype=float,
    )
    b = np.array([4.0, 1.0, 7.0, 6.0], dtype=float)

    p, l, u, swap_count = lu_decompose_partial_pivot(a, tol=1e-12)
    x_lu = solve_with_lu(p, l, u, b)

    reconstruction_error = np.linalg.norm(p @ a - l @ u, ord="fro")
    residual = np.linalg.norm(a @ x_lu - b, ord=2)

    x_numpy = np.linalg.solve(a, b)
    solution_gap = np.linalg.norm(x_lu - x_numpy, ord=2)

    det_from_lu = ((-1) ** swap_count) * float(np.prod(np.diag(u)))
    det_from_numpy = float(np.linalg.det(a))

    print("=== Input Matrix A ===")
    print(a)
    print("\n=== Vector b ===")
    print(b)

    print("\n=== LU Decomposition with Partial Pivoting ===")
    print("P =")
    print(p)
    print("L =")
    print(l)
    print("U =")
    print(u)

    print("\n=== Solve Ax=b ===")
    print("x (from LU)    =", x_lu)
    print("x (from NumPy) =", x_numpy)

    print("\n=== Quality Metrics ===")
    print(f"||P@A - L@U||_F       = {reconstruction_error:.3e}")
    print(f"||A@x - b||_2         = {residual:.3e}")
    print(f"||x_lu - x_numpy||_2  = {solution_gap:.3e}")
    print(f"det(A) from LU        = {det_from_lu:.6f}")
    print(f"det(A) from NumPy     = {det_from_numpy:.6f}")

    singular = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    try:
        _ = lu_decompose_partial_pivot(singular)
    except ValueError as exc:
        print("\nExpected failure on singular matrix:")
        print(exc)


if __name__ == "__main__":
    main()

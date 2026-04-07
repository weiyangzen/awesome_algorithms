"""Minimal runnable MVP for Gaussian Elimination (MATH-0051)."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def gaussian_elimination_solve(
    a: Sequence[Sequence[float]],
    b: Sequence[float],
    tol: float = 1e-12,
) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray]:
    """Solve Ax=b using Gaussian elimination with partial pivoting.

    Returns:
        x: Solution vector.
        ops: Human-readable elimination operations.
        u: Upper-triangular matrix after elimination.
        c: Transformed right-hand side after elimination.
    """
    a_mat = np.array(a, dtype=float, copy=True)
    b_vec = np.array(b, dtype=float, copy=True).reshape(-1)

    if a_mat.ndim != 2:
        raise ValueError("A must be a 2D matrix.")
    n, m = a_mat.shape
    if n != m:
        raise ValueError("A must be square.")
    if b_vec.shape[0] != n:
        raise ValueError("Dimension mismatch between A and b.")

    ops: List[str] = []

    # Forward elimination.
    for col in range(n):
        pivot_row = col + int(np.argmax(np.abs(a_mat[col:, col])))
        pivot_val = a_mat[pivot_row, col]

        if abs(pivot_val) < tol:
            raise ValueError("Matrix is singular or near-singular under tolerance.")

        if pivot_row != col:
            a_mat[[col, pivot_row], :] = a_mat[[pivot_row, col], :]
            b_vec[[col, pivot_row]] = b_vec[[pivot_row, col]]
            ops.append(f"swap R{col} <-> R{pivot_row}")

        for row in range(col + 1, n):
            factor = a_mat[row, col] / a_mat[col, col]
            if abs(factor) < tol:
                a_mat[row, col] = 0.0
                continue

            a_mat[row, col:] -= factor * a_mat[col, col:]
            b_vec[row] -= factor * b_vec[col]
            a_mat[row, col] = 0.0
            ops.append(f"R{row} <- R{row} - ({factor:.6g}) * R{col}")

    # Back substitution.
    x = np.zeros(n, dtype=float)
    for row in range(n - 1, -1, -1):
        diag = a_mat[row, row]
        if abs(diag) < tol:
            raise ValueError("Zero diagonal encountered during back substitution.")
        rhs = b_vec[row] - np.dot(a_mat[row, row + 1 :], x[row + 1 :])
        x[row] = rhs / diag

    return x, ops, a_mat, b_vec


def residual_l2(a: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    """Return the L2 norm of residual ||Ax - b||_2."""
    return float(np.linalg.norm(a @ x - b, ord=2))


def run_case(name: str, a: Sequence[Sequence[float]], b: Sequence[float]) -> None:
    """Run one deterministic test case and print summary."""
    a_np = np.array(a, dtype=float)
    b_np = np.array(b, dtype=float)

    x, ops, u, c = gaussian_elimination_solve(a_np, b_np)
    x_ref = np.linalg.solve(a_np, b_np)

    res = residual_l2(a_np, x, b_np)
    inf_err = float(np.linalg.norm(x - x_ref, ord=np.inf))

    assert res < 1e-9, f"Residual too large for {name}: {res}"
    assert inf_err < 1e-9, f"Solution mismatch for {name}: {inf_err}"

    print(f"[{name}]")
    print(f"solution x = {x}")
    print(f"residual ||Ax-b||_2 = {res:.3e}")
    print(f"vs numpy solve (inf-norm error) = {inf_err:.3e}")
    print(f"upper-triangular diag(U) = {np.diag(u)}")
    print(f"transformed rhs c = {c}")
    if ops:
        print("row ops (first 4):")
        for line in ops[:4]:
            print(f"  - {line}")
    print("-" * 72)


def main() -> None:
    print("Gaussian Elimination MVP (MATH-0051)")
    print("=" * 72)

    run_case(
        "classic_3x3",
        [[2.0, 1.0, -1.0], [-3.0, -1.0, 2.0], [-2.0, 1.0, 2.0]],
        [8.0, -11.0, -3.0],
    )

    run_case(
        "pivot_needed_2x2",
        [[1.0e-20, 1.0], [1.0, 1.0]],
        [1.0, 2.0],
    )

    run_case(
        "deterministic_4x4",
        [[4.0, -2.0, 1.0, 3.0], [3.0, 6.0, -4.0, -2.0], [2.0, 1.0, 8.0, -5.0], [1.0, -3.0, 2.0, 7.0]],
        [20.0, -33.0, 17.0, 9.0],
    )

    print("singular matrix check:")
    try:
        gaussian_elimination_solve([[1.0, 2.0], [2.0, 4.0]], [3.0, 6.0])
    except ValueError as exc:
        print(f"  expected failure: {exc}")
    else:
        raise AssertionError("Singular system should have raised ValueError.")

    print("=" * 72)
    print("All checks passed.")


if __name__ == "__main__":
    main()

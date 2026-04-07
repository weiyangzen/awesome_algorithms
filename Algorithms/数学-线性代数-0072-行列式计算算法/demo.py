"""MVP: determinant calculation via Gaussian elimination with partial pivoting."""

from __future__ import annotations

import numpy as np


def determinant_gaussian_elimination(matrix: np.ndarray, pivot_tol: float = 1e-12) -> float:
    """Compute determinant using row elimination with partial pivoting.

    The algorithm tracks:
    1) row swap parity (sign),
    2) upper-triangular diagonal product after elimination.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input must be a square 2D matrix.")

    a = matrix.astype(float, copy=True)
    n = a.shape[0]
    sign = 1.0

    for col in range(n):
        # Partial pivoting: choose the row with max |value| in current column.
        pivot_offset = int(np.argmax(np.abs(a[col:, col])))
        pivot_row = col + pivot_offset
        pivot_value = a[pivot_row, col]

        if abs(pivot_value) < pivot_tol:
            return 0.0

        if pivot_row != col:
            a[[col, pivot_row], :] = a[[pivot_row, col], :]
            sign *= -1.0

        # Eliminate entries below the pivot.
        for row in range(col + 1, n):
            factor = a[row, col] / a[col, col]
            a[row, col] = 0.0
            a[row, col + 1 :] -= factor * a[col, col + 1 :]

    return float(sign * np.prod(np.diag(a)))


def run_examples() -> None:
    examples = {
        "A_3x3": np.array(
            [
                [2.0, -1.0, 0.0],
                [1.0, 3.0, 4.0],
                [0.0, 5.0, 2.0],
            ]
        ),
        "B_4x4": np.array(
            [
                [3.0, 2.0, -1.0, 4.0],
                [2.0, 1.0, 5.0, 7.0],
                [0.0, 5.0, 2.0, -6.0],
                [-1.0, 2.0, 3.0, 0.0],
            ]
        ),
        "C_singular": np.array(
            [
                [1.0, 2.0, 3.0],
                [2.0, 4.0, 6.0],
                [1.0, 0.0, 1.0],
            ]
        ),
    }

    print("=== Determinant examples ===")
    for name, mat in examples.items():
        det_mvp = determinant_gaussian_elimination(mat)
        det_numpy = float(np.linalg.det(mat))
        abs_err = abs(det_mvp - det_numpy)
        print(f"{name}:")
        print(mat)
        print(f"  det_mvp   = {det_mvp:.10f}")
        print(f"  det_numpy = {det_numpy:.10f}")
        print(f"  abs_err   = {abs_err:.3e}")


def run_stress_test(seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    worst_rel_err = 0.0

    for n in (2, 3, 5, 8):
        for _ in range(60):
            mat = rng.normal(size=(n, n))
            det_mvp = determinant_gaussian_elimination(mat)
            det_numpy = float(np.linalg.det(mat))
            rel_err = abs(det_mvp - det_numpy) / (1.0 + abs(det_numpy))
            worst_rel_err = max(worst_rel_err, rel_err)

    print("\n=== Random stress test ===")
    print(f"Worst relative error: {worst_rel_err:.3e}")
    # Numerical computation has rounding error; this threshold is practical.
    assert worst_rel_err < 1e-8, "Unexpectedly large numerical error."


def main() -> None:
    run_examples()
    run_stress_test()
    print("\nDone.")


if __name__ == "__main__":
    main()

"""Matrix inversion MVP via Gauss-Jordan elimination with partial pivoting."""

from __future__ import annotations

import numpy as np


def gauss_jordan_inverse(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Compute inverse(matrix) using Gauss-Jordan elimination.

    Args:
        matrix: Square matrix of shape (n, n).
        eps: Pivot threshold below which the matrix is treated as singular.

    Returns:
        Inverse matrix of shape (n, n).

    Raises:
        ValueError: If input is not a square 2D matrix.
        np.linalg.LinAlgError: If matrix is singular or near-singular for given eps.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input must be a square 2D matrix.")

    n = matrix.shape[0]
    a = matrix.astype(np.float64, copy=True)
    identity = np.eye(n, dtype=np.float64)
    aug = np.hstack([a, identity])  # [A | I]

    for col in range(n):
        pivot_row = col + int(np.argmax(np.abs(aug[col:, col])))
        pivot_value = aug[pivot_row, col]
        if abs(pivot_value) < eps:
            raise np.linalg.LinAlgError(
                f"Matrix is singular to working precision at column {col}."
            )

        if pivot_row != col:
            aug[[col, pivot_row]] = aug[[pivot_row, col]]

        aug[col] = aug[col] / aug[col, col]

        for row in range(n):
            if row == col:
                continue
            factor = aug[row, col]
            if factor != 0.0:
                aug[row] = aug[row] - factor * aug[col]

    return aug[:, n:]


def compute_quality_metrics(a: np.ndarray, a_inv: np.ndarray) -> dict[str, float]:
    """Return diagnostic metrics for inversion quality."""
    n = a.shape[0]
    identity = np.eye(n, dtype=np.float64)
    left_residual = np.linalg.norm(a @ a_inv - identity, ord=np.inf)
    right_residual = np.linalg.norm(a_inv @ a - identity, ord=np.inf)
    reference_inv = np.linalg.inv(a)
    distance_to_reference = np.linalg.norm(a_inv - reference_inv, ord=np.inf)
    return {
        "residual_left": float(left_residual),
        "residual_right": float(right_residual),
        "distance_to_numpy_inv": float(distance_to_reference),
    }


def run_case(name: str, a: np.ndarray, eps: float = 1e-12) -> None:
    """Run one test case and print diagnostics."""
    print(f"\n=== Case: {name} ===")
    print("A =")
    print(np.array2string(a, precision=6, suppress_small=True))
    print(f"shape={a.shape}, cond(A)={np.linalg.cond(a):.6e}")
    try:
        a_inv = gauss_jordan_inverse(a, eps=eps)
        metrics = compute_quality_metrics(a, a_inv)
        print("A_inv (Gauss-Jordan) =")
        print(np.array2string(a_inv, precision=6, suppress_small=True))
        for key, value in metrics.items():
            print(f"{key}: {value:.6e}")
    except np.linalg.LinAlgError as err:
        print(f"Expected failure for non-invertible matrix: {err}")


def main() -> None:
    """Execute non-interactive MVP demo."""
    rng = np.random.default_rng(42)

    # Case 1: random well-conditioned matrix (shifted for invertibility).
    m = rng.normal(size=(4, 4))
    a1 = m + 2.5 * np.eye(4)

    # Case 2: Hilbert matrix, invertible but ill-conditioned.
    n2 = 5
    i = np.arange(1, n2 + 1, dtype=np.float64)
    a2 = 1.0 / (i[:, None] + i[None, :] - 1.0)

    # Case 3: singular matrix (row dependency).
    a3 = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )

    run_case("Random invertible 4x4", a1)
    run_case("Ill-conditioned Hilbert 5x5", a2)
    run_case("Singular 3x3", a3)


if __name__ == "__main__":
    main()

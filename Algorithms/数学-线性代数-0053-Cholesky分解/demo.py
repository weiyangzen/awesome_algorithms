"""Cholesky decomposition MVP (manual implementation + NumPy verification)."""

from __future__ import annotations

import numpy as np


def make_spd_matrix(n: int, seed: int = 42) -> np.ndarray:
    """Create a deterministic symmetric positive definite matrix."""
    rng = np.random.default_rng(seed)
    m = rng.normal(size=(n, n))
    return m @ m.T + n * np.eye(n)


def manual_cholesky(a: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Compute lower-triangular L such that A = L @ L.T for SPD matrix A."""
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Input must be a square matrix.")

    if not np.allclose(a, a.T, atol=tol, rtol=0.0):
        raise ValueError("Input matrix must be symmetric.")

    n = a.shape[0]
    l = np.zeros_like(a, dtype=float)

    for i in range(n):
        for j in range(i + 1):
            s = float(np.dot(l[i, :j], l[j, :j]))
            if i == j:
                diag = a[i, i] - s
                if diag <= tol:
                    raise ValueError(
                        "Matrix is not strictly positive definite "
                        f"(pivot {i} gives {diag:.3e})."
                    )
                l[i, j] = np.sqrt(diag)
            else:
                l[i, j] = (a[i, j] - s) / l[j, j]

    return l


def solve_with_cholesky(l: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve Ax=b using A=LL^T with forward and backward substitution."""
    n = l.shape[0]
    y = np.zeros(n, dtype=float)

    for i in range(n):
        y[i] = (b[i] - np.dot(l[i, :i], y[:i])) / l[i, i]

    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(l.T[i, i + 1 :], x[i + 1 :])) / l.T[i, i]

    return x


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)

    a = make_spd_matrix(n=4, seed=7)
    b = np.array([1.0, -2.0, 0.5, 3.0], dtype=float)

    l_manual = manual_cholesky(a)
    l_numpy = np.linalg.cholesky(a)

    factor_error = np.linalg.norm(a - l_manual @ l_manual.T, ord="fro")
    impl_gap = np.linalg.norm(l_manual - l_numpy, ord="fro")

    x = solve_with_cholesky(l_manual, b)
    residual = np.linalg.norm(a @ x - b, ord=2)

    print("=== SPD Matrix A ===")
    print(a)
    print("\n=== Manual Cholesky L ===")
    print(l_manual)
    print("\n=== NumPy Cholesky L ===")
    print(l_numpy)
    print("\n=== Quality Metrics ===")
    print(f"Reconstruction error ||A - LL^T||_F: {factor_error:.3e}")
    print(f"Implementation gap ||L_manual - L_numpy||_F: {impl_gap:.3e}")
    print(f"Linear solve residual ||Ax - b||_2: {residual:.3e}")

    bad = np.array([[1.0, 2.0], [2.0, 1.0]], dtype=float)
    try:
        _ = manual_cholesky(bad)
    except ValueError as exc:
        print("\nExpected failure on non-SPD matrix:")
        print(exc)


if __name__ == "__main__":
    main()

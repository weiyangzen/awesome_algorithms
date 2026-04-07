"""Sylvester equation solver MVP.

Primary path: explicit vec+Kronecker linearization with NumPy.
Optional path: compare against scipy.linalg.solve_sylvester when SciPy is available.
"""

from __future__ import annotations

import numpy as np

try:
    from scipy.linalg import solve_sylvester as scipy_solve_sylvester
except Exception:  # SciPy may be unavailable in minimal environments.
    scipy_solve_sylvester = None


def validate_inputs(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> tuple[int, int]:
    """Validate matrix shapes and numeric finiteness for AX + XB = C."""
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("A must be a square 2D matrix.")
    if b.ndim != 2 or b.shape[0] != b.shape[1]:
        raise ValueError("B must be a square 2D matrix.")

    m = a.shape[0]
    n = b.shape[0]
    if c.shape != (m, n):
        raise ValueError(f"C must have shape ({m}, {n}), got {c.shape}.")

    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)) or not np.all(np.isfinite(c)):
        raise ValueError("A, B, C must contain only finite values.")

    return m, n


def spectral_separation(a: np.ndarray, b: np.ndarray) -> float:
    """Compute min |lambda_i(A) + lambda_j(B)| as a conditioning indicator."""
    eig_a = np.linalg.eigvals(a)
    eig_b = np.linalg.eigvals(b)
    pairwise = np.abs(eig_a[:, None] + eig_b[None, :])
    return float(np.min(pairwise))


def solve_sylvester_kron(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Solve AX + XB = C by vectorization and Kronecker linearization."""
    m, n = validate_inputs(a, b, c)
    eye_m = np.eye(m, dtype=float)
    eye_n = np.eye(n, dtype=float)

    # vec(AX + XB) = (I_n ⊗ A + B^T ⊗ I_m) vec(X)
    k = np.kron(eye_n, a) + np.kron(b.T, eye_m)
    rhs = c.reshape(-1, order="F")

    x_vec = np.linalg.solve(k, rhs)
    x = x_vec.reshape((m, n), order="F")
    return x


def residual_norm(a: np.ndarray, b: np.ndarray, c: np.ndarray, x: np.ndarray) -> float:
    """Frobenius norm of residual matrix AX + XB - C."""
    r = a @ x + x @ b - c
    return float(np.linalg.norm(r, ord="fro"))


def demo_unique_case() -> None:
    """Run a deterministic unique-solution example."""
    a = np.array(
        [
            [3.0, 1.0, 0.0],
            [0.0, 2.0, -1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    b = np.array(
        [
            [-5.0, 1.0],
            [0.0, -2.5],
        ],
        dtype=float,
    )
    c = np.array(
        [
            [1.0, -2.0],
            [0.5, 3.0],
            [4.0, -1.0],
        ],
        dtype=float,
    )

    sep = spectral_separation(a, b)
    x_kron = solve_sylvester_kron(a, b, c)
    res_kron = residual_norm(a, b, c, x_kron)

    print("=== Unique-Solution Example ===")
    print("A =")
    print(a)
    print("B =")
    print(b)
    print("C =")
    print(c)
    print(f"Spectral separation min|lambda(A)+lambda(B)|: {sep:.3e}")
    print("\nX (Kronecker solve) =")
    print(x_kron)
    print(f"Residual ||AX + XB - C||_F: {res_kron:.3e}")

    if scipy_solve_sylvester is not None:
        x_scipy = scipy_solve_sylvester(a, b, c)
        res_scipy = residual_norm(a, b, c, x_scipy)
        gap = float(np.linalg.norm(x_kron - x_scipy, ord="fro"))
        print("\nSciPy comparison is available.")
        print(f"Residual (SciPy) ||AX + XB - C||_F: {res_scipy:.3e}")
        print(f"Gap ||X_kron - X_scipy||_F: {gap:.3e}")
    else:
        print("\nSciPy not installed; skipped scipy.linalg.solve_sylvester comparison.")


def demo_infeasible_case() -> None:
    """Demonstrate a known infeasible case (expected failure)."""
    # A = I, B = -I => AX + XB = X - X = 0, so C != 0 is infeasible.
    a = np.eye(2, dtype=float)
    b = -np.eye(2, dtype=float)
    c = np.ones((2, 2), dtype=float)

    print("\n=== Infeasible Example (Expected Failure) ===")
    try:
        _ = solve_sylvester_kron(a, b, c)
        print("Unexpectedly solved an infeasible case; check conditioning and solver behavior.")
    except np.linalg.LinAlgError as exc:
        print("Expected singular-system failure captured:")
        print(exc)


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)
    demo_unique_case()
    demo_infeasible_case()


if __name__ == "__main__":
    main()

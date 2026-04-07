"""Minimal runnable MVP for Kronecker product computation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KronReport:
    output_shape: tuple[int, int]
    max_abs_error_vs_numpy: float
    mixed_product_error: float
    bilinear_error: float
    associativity_error: float
    frobenius_norm: float


def validate_matrix(name: str, mat: np.ndarray) -> None:
    if mat.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix.")
    if not np.isfinite(mat).all():
        raise ValueError(f"{name} must contain only finite values.")


def kronecker_product_manual(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute A ⊗ B explicitly by block expansion."""
    validate_matrix("A", a)
    validate_matrix("B", b)

    m, n = a.shape
    p, q = b.shape

    out = np.empty((m * p, n * q), dtype=np.result_type(a.dtype, b.dtype, np.float64))

    for i in range(m):
        for j in range(n):
            r0 = i * p
            r1 = (i + 1) * p
            c0 = j * q
            c1 = (j + 1) * q
            out[r0:r1, c0:c1] = a[i, j] * b

    return out


def run_checks(report: KronReport, tol: float = 1e-12) -> None:
    if report.max_abs_error_vs_numpy > tol:
        raise AssertionError(
            f"Manual-vs-NumPy error too large: {report.max_abs_error_vs_numpy:.3e}"
        )
    if report.mixed_product_error > tol:
        raise AssertionError(f"Mixed-product identity error too large: {report.mixed_product_error:.3e}")
    if report.bilinear_error > tol:
        raise AssertionError(f"Bilinearity check error too large: {report.bilinear_error:.3e}")
    if report.associativity_error > tol:
        raise AssertionError(
            f"Associativity check error too large: {report.associativity_error:.3e}"
        )


def main() -> None:
    # Example matrices for A ⊗ B.
    a = np.array(
        [
            [1.0, 2.0],
            [-1.0, 3.0],
            [0.5, 4.0],
        ],
        dtype=float,
    )
    b = np.array(
        [
            [2.0, -1.0, 0.0],
            [1.0, 0.0, 2.0],
        ],
        dtype=float,
    )

    k_manual = kronecker_product_manual(a, b)
    k_ref = np.kron(a, b)

    max_abs_error = float(np.max(np.abs(k_manual - k_ref)))

    # Identity: (A ⊗ B)(x ⊗ y) = (Ax) ⊗ (By)
    x = np.array([1.5, -2.0], dtype=float)
    y = np.array([0.5, 1.0, -1.0], dtype=float)
    left = k_manual @ np.kron(x, y)
    right = np.kron(a @ x, b @ y)
    mixed_error = float(np.linalg.norm(left - right, ord=2))

    # Bilinearity in first argument: (alpha*A1 + beta*A2) ⊗ B
    a2 = np.array(
        [
            [0.0, 1.0],
            [2.0, -1.0],
            [1.0, 0.5],
        ],
        dtype=float,
    )
    alpha = 0.75
    beta = -1.2
    lhs_bilinear = kronecker_product_manual(alpha * a + beta * a2, b)
    rhs_bilinear = alpha * kronecker_product_manual(a, b) + beta * kronecker_product_manual(a2, b)
    bilinear_error = float(np.max(np.abs(lhs_bilinear - rhs_bilinear)))

    # Associativity: (A ⊗ B) ⊗ C = A ⊗ (B ⊗ C)
    c = np.array([[1.0], [-2.0]], dtype=float)
    assoc_left = kronecker_product_manual(kronecker_product_manual(a, b), c)
    assoc_right = kronecker_product_manual(a, kronecker_product_manual(b, c))
    associativity_error = float(np.max(np.abs(assoc_left - assoc_right)))

    report = KronReport(
        output_shape=k_manual.shape,
        max_abs_error_vs_numpy=max_abs_error,
        mixed_product_error=mixed_error,
        bilinear_error=bilinear_error,
        associativity_error=associativity_error,
        frobenius_norm=float(np.linalg.norm(k_manual, ord="fro")),
    )

    run_checks(report)

    print("Kronecker Product demo")
    print(f"A_shape={a.shape}, B_shape={b.shape}")
    print(f"kron_shape={report.output_shape}")
    print(f"frobenius_norm={report.frobenius_norm:.6f}")
    print(f"max_abs_error_vs_numpy={report.max_abs_error_vs_numpy:.3e}")
    print(f"mixed_product_error={report.mixed_product_error:.3e}")
    print(f"bilinear_error={report.bilinear_error:.3e}")
    print(f"associativity_error={report.associativity_error:.3e}")
    print("All checks passed.")


if __name__ == "__main__":
    main()

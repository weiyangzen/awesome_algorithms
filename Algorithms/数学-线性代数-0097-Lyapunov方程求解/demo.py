"""Minimal runnable MVP for solving a continuous Lyapunov equation.

Target equation in this demo:
    A^T X + X A + Q = 0

Core solver is implemented explicitly via Kronecker vectorization:
    [(I ⊗ A^T) + (A^T ⊗ I)] vec(X) = -vec(Q)

SciPy is used only as a reference implementation for cross-checking.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from scipy.linalg import solve_continuous_lyapunov

    HAS_SCIPY = True
except Exception:  # pragma: no cover - environment dependent fallback
    solve_continuous_lyapunov = None
    HAS_SCIPY = False


@dataclass
class LyapunovReport:
    matrix_shape: tuple[int, int]
    is_hurwitz: bool
    used_scipy_reference: bool
    residual_fro: float
    relative_residual: float
    max_abs_error_vs_scipy: float
    symmetry_error: float
    min_eig_of_sym_part: float


def validate_inputs(a: np.ndarray, q: np.ndarray) -> None:
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("A must be a square matrix.")
    if q.ndim != 2 or q.shape[0] != q.shape[1]:
        raise ValueError("Q must be a square matrix.")
    if a.shape != q.shape:
        raise ValueError("A and Q must have the same shape.")
    if not np.isfinite(a).all() or not np.isfinite(q).all():
        raise ValueError("A and Q must contain only finite values.")
    if not np.allclose(q, q.T, atol=1e-12, rtol=0.0):
        raise ValueError("Q must be symmetric for this MVP.")


def is_hurwitz(a: np.ndarray) -> bool:
    eigvals = np.linalg.eigvals(a)
    return bool(np.max(np.real(eigvals)) < 0.0)


def solve_lyapunov_via_kronecker(a: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Solve A^T X + X A + Q = 0 by vectorization and Kronecker products."""
    validate_inputs(a, q)

    n = a.shape[0]
    i = np.eye(n, dtype=float)

    k = np.kron(i, a.T) + np.kron(a.T, i)
    b = -q.reshape(n * n, order="F")

    x_vec = np.linalg.solve(k, b)
    x = x_vec.reshape((n, n), order="F")
    return x


def compute_residual_metrics(a: np.ndarray, q: np.ndarray, x: np.ndarray) -> tuple[float, float, float]:
    residual = a.T @ x + x @ a + q
    residual_fro = float(np.linalg.norm(residual, ord="fro"))
    q_fro = float(np.linalg.norm(q, ord="fro"))
    relative_residual = residual_fro / max(q_fro, 1e-15)
    symmetry_error = float(np.linalg.norm(x - x.T, ord="fro"))
    return residual_fro, relative_residual, symmetry_error


def run_checks(report: LyapunovReport) -> None:
    if report.residual_fro > 1e-8:
        raise AssertionError(f"Residual too large: {report.residual_fro:.3e}")
    if report.relative_residual > 1e-9:
        raise AssertionError(f"Relative residual too large: {report.relative_residual:.3e}")
    if report.used_scipy_reference and report.max_abs_error_vs_scipy > 1e-8:
        raise AssertionError(
            f"Mismatch vs SciPy too large: {report.max_abs_error_vs_scipy:.3e}"
        )
    if report.symmetry_error > 1e-8:
        raise AssertionError(f"Symmetry error too large: {report.symmetry_error:.3e}")
    if report.is_hurwitz and report.min_eig_of_sym_part <= 1e-10:
        raise AssertionError(
            "Expected positive-definite solution for Hurwitz A and SPD Q, "
            f"but min eigenvalue is {report.min_eig_of_sym_part:.3e}"
        )


def build_stable_demo_case() -> tuple[np.ndarray, np.ndarray]:
    """Build deterministic A (Hurwitz) and symmetric positive-definite Q."""
    rng = np.random.default_rng(2026)
    n = 4

    target_eigs = np.array([-0.6, -1.1, -2.4, -3.2], dtype=float)

    while True:
        v = rng.standard_normal((n, n))
        if np.linalg.matrix_rank(v) == n and np.linalg.cond(v) < 80.0:
            break

    a = v @ np.diag(target_eigs) @ np.linalg.inv(v)

    c = rng.standard_normal((n, n))
    q = c.T @ c + 0.6 * np.eye(n)
    q = 0.5 * (q + q.T)

    return a, q


def main() -> None:
    a, q = build_stable_demo_case()

    x_kron = solve_lyapunov_via_kronecker(a, q)
    if HAS_SCIPY:
        x_ref = solve_continuous_lyapunov(a.T, -q)
        max_abs_error = float(np.max(np.abs(x_kron - x_ref)))
    else:
        x_ref = None
        max_abs_error = 0.0

    residual_fro, relative_residual, symmetry_error = compute_residual_metrics(a, q, x_kron)

    x_sym = 0.5 * (x_kron + x_kron.T)
    min_eig = float(np.min(np.linalg.eigvalsh(x_sym)))

    stable = is_hurwitz(a)
    if not stable:
        print("Warning: A is not Hurwitz; positive-definite solution guarantee does not apply.")

    report = LyapunovReport(
        matrix_shape=a.shape,
        is_hurwitz=stable,
        used_scipy_reference=HAS_SCIPY,
        residual_fro=residual_fro,
        relative_residual=relative_residual,
        max_abs_error_vs_scipy=max_abs_error,
        symmetry_error=symmetry_error,
        min_eig_of_sym_part=min_eig,
    )

    run_checks(report)

    print("Continuous Lyapunov equation demo")
    print("equation: A^T X + X A + Q = 0")
    print(f"matrix_shape={report.matrix_shape}")
    print(f"is_hurwitz={report.is_hurwitz}")
    print(f"used_scipy_reference={report.used_scipy_reference}")
    print(f"residual_fro={report.residual_fro:.3e}")
    print(f"relative_residual={report.relative_residual:.3e}")
    if report.used_scipy_reference:
        print(f"max_abs_error_vs_scipy={report.max_abs_error_vs_scipy:.3e}")
    else:
        print("max_abs_error_vs_scipy=SKIPPED (scipy not installed)")
    print(f"symmetry_error={report.symmetry_error:.3e}")
    print(f"min_eig_of_sym_part={report.min_eig_of_sym_part:.6f}")
    print("All checks passed.")


if __name__ == "__main__":
    main()

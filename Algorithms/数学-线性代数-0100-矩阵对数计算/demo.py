"""Matrix logarithm MVP (SPD case) for MATH-0100."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


ArrayF = NDArray[np.float64]


@dataclass
class MatrixLogReport:
    matrix_a: ArrayF
    matrix_log_a: ArrayF
    matrix_exp_log_a: ArrayF
    relative_reconstruction_error: float
    relative_symmetry_error: float
    spectral_consistency_error: float
    commutator_fro_norm: float


def validate_spd_matrix(a: ArrayF, *, sym_tol: float = 1e-10, eig_floor: float = 1e-12) -> None:
    """Validate that a matrix is finite, square, symmetric, and positive definite."""
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Input must be a square matrix.")
    if not np.all(np.isfinite(a)):
        raise ValueError("Input contains NaN or Inf.")
    if not np.allclose(a, a.T, atol=sym_tol, rtol=0.0):
        raise ValueError("Input must be symmetric for the SPD-only MVP.")

    eigvals = np.linalg.eigvalsh(a)
    if float(np.min(eigvals)) <= eig_floor:
        raise ValueError(
            "Input is not positive definite (minimum eigenvalue <= eig_floor)."
        )


def matrix_log_spd(a: ArrayF) -> ArrayF:
    """Compute principal matrix logarithm for SPD matrices via spectral decomposition."""
    validate_spd_matrix(a)

    eigvals, eigvecs = np.linalg.eigh(a)
    log_eigvals = np.log(eigvals)
    log_a = eigvecs @ np.diag(log_eigvals) @ eigvecs.T
    # Numerical cleanup: keep exact symmetry up to rounding.
    log_a = 0.5 * (log_a + log_a.T)
    return log_a


def matrix_exp_symmetric(s: ArrayF) -> ArrayF:
    """Compute matrix exponential for symmetric matrices via spectral decomposition."""
    if s.ndim != 2 or s.shape[0] != s.shape[1]:
        raise ValueError("Input must be a square matrix.")
    if not np.allclose(s, s.T, atol=1e-10, rtol=0.0):
        raise ValueError("Input must be symmetric in this helper.")

    eigvals, eigvecs = np.linalg.eigh(s)
    exp_eigvals = np.exp(eigvals)
    exp_s = eigvecs @ np.diag(exp_eigvals) @ eigvecs.T
    exp_s = 0.5 * (exp_s + exp_s.T)
    return exp_s


def relative_fro_error(reference: ArrayF, estimate: ArrayF) -> float:
    """Compute relative Frobenius norm error."""
    denom = float(np.linalg.norm(reference, ord="fro"))
    if denom == 0.0:
        return float(np.linalg.norm(estimate, ord="fro"))
    return float(np.linalg.norm(reference - estimate, ord="fro") / denom)


def build_spd_matrix(
    n: int = 5,
    *,
    seed: int = 42,
    min_eig: float = 0.25,
    max_eig: float = 3.0,
) -> ArrayF:
    """Build a reproducible SPD matrix A = Q diag(lambda) Q^T."""
    if n <= 0:
        raise ValueError("n must be positive.")
    if min_eig <= 0.0 or max_eig <= min_eig:
        raise ValueError("Need 0 < min_eig < max_eig.")

    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(n, n))
    q, _ = np.linalg.qr(raw)
    eigvals = np.linspace(min_eig, max_eig, n)
    a = q @ np.diag(eigvals) @ q.T
    a = 0.5 * (a + a.T)
    return a.astype(np.float64)


def evaluate_matrix_log(a: ArrayF) -> MatrixLogReport:
    """Run matrix-log computation and produce diagnostics."""
    log_a = matrix_log_spd(a)
    exp_log_a = matrix_exp_symmetric(log_a)

    rec_err = relative_fro_error(a, exp_log_a)
    sym_err = relative_fro_error(log_a, log_a.T)

    evals_a = np.linalg.eigvalsh(a)
    evals_log_a = np.linalg.eigvalsh(log_a)
    spectral_err = float(np.max(np.abs(np.sort(evals_log_a) - np.log(np.sort(evals_a)))))

    commutator = a @ log_a - log_a @ a
    commutator_norm = float(np.linalg.norm(commutator, ord="fro"))

    return MatrixLogReport(
        matrix_a=a,
        matrix_log_a=log_a,
        matrix_exp_log_a=exp_log_a,
        relative_reconstruction_error=rec_err,
        relative_symmetry_error=sym_err,
        spectral_consistency_error=spectral_err,
        commutator_fro_norm=commutator_norm,
    )


def run_checks(report: MatrixLogReport) -> None:
    """Fail fast if numerical quality is below MVP expectations."""
    if report.relative_reconstruction_error > 1e-10:
        raise AssertionError(
            f"Reconstruction error too large: {report.relative_reconstruction_error:.3e}"
        )
    if report.relative_symmetry_error > 1e-12:
        raise AssertionError(
            f"Symmetry error too large: {report.relative_symmetry_error:.3e}"
        )
    if report.spectral_consistency_error > 1e-10:
        raise AssertionError(
            f"Spectral consistency error too large: {report.spectral_consistency_error:.3e}"
        )
    if report.commutator_fro_norm > 1e-10:
        raise AssertionError(
            f"Commutator norm too large: {report.commutator_fro_norm:.3e}"
        )


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)

    a = build_spd_matrix(n=5, seed=42, min_eig=0.3, max_eig=2.7)
    report = evaluate_matrix_log(a)
    run_checks(report)

    print("=== Matrix Logarithm MVP (SPD) ===")
    print(f"shape(A): {report.matrix_a.shape}")
    print(f"relative reconstruction error: {report.relative_reconstruction_error:.3e}")
    print(f"relative symmetry error(logA): {report.relative_symmetry_error:.3e}")
    print(f"spectral consistency error: {report.spectral_consistency_error:.3e}")
    print(f"commutator ||A logA - logA A||_F: {report.commutator_fro_norm:.3e}")

    print("\nA:")
    print(report.matrix_a)
    print("\nlog(A):")
    print(report.matrix_log_a)
    print("\nexp(log(A)):")
    print(report.matrix_exp_log_a)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()

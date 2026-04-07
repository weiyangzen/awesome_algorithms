"""Minimal runnable MVP for matrix square root computation (SPD case)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from scipy.linalg import sqrtm as scipy_sqrtm
except Exception:  # pragma: no cover - scipy is optional for this MVP
    scipy_sqrtm = None


EPS = 1e-12


@dataclass
class SqrtReport:
    name: str
    shape: tuple[int, int]
    min_eig_input: float
    max_eig_input: float
    residual_fro: float
    relative_residual: float
    symmetry_error: float
    scipy_diff_fro: float | None
    scipy_imag_norm: float | None


def validate_spd_matrix(a: np.ndarray, *, name: str = "A", tol: float = 1e-10) -> None:
    """Validate that input is a finite real symmetric positive definite matrix."""
    if a.ndim != 2:
        raise ValueError(f"{name} must be 2D, got ndim={a.ndim}")

    n, m = a.shape
    if n != m:
        raise ValueError(f"{name} must be square, got shape={a.shape}")

    if not np.isfinite(a).all():
        raise ValueError(f"{name} contains non-finite values")

    if not np.allclose(a, a.T, atol=tol, rtol=0.0):
        raise ValueError(f"{name} must be symmetric within tolerance={tol}")

    eigvals = np.linalg.eigvalsh(a)
    min_eig = float(eigvals.min())
    if min_eig <= tol:
        raise ValueError(
            f"{name} must be positive definite. min eigenvalue={min_eig:.3e}, tol={tol:.1e}"
        )


def matrix_sqrt_spd_eigh(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute principal square root of SPD matrix via eigen-decomposition."""
    eigvals, eigvecs = np.linalg.eigh(a)

    if np.any(eigvals <= 0.0):
        raise ValueError("matrix_sqrt_spd_eigh expects an SPD matrix with positive eigenvalues")

    sqrt_eigvals = np.sqrt(eigvals)
    x = (eigvecs * sqrt_eigvals) @ eigvecs.T

    # Remove tiny anti-symmetric floating noise.
    x = 0.5 * (x + x.T)
    return x, eigvals


def analyze_case(name: str, a: np.ndarray) -> SqrtReport:
    """Run sqrt computation and collect diagnostics for one matrix."""
    validate_spd_matrix(a, name=name)

    x, eigvals = matrix_sqrt_spd_eigh(a)
    reconstructed = x @ x

    residual_fro = float(np.linalg.norm(reconstructed - a, ord="fro"))
    denom = max(float(np.linalg.norm(a, ord="fro")), EPS)
    relative_residual = residual_fro / denom
    symmetry_error = float(np.linalg.norm(x - x.T, ord="fro"))

    scipy_diff_fro: float | None = None
    scipy_imag_norm: float | None = None

    if scipy_sqrtm is not None:
        ref = scipy_sqrtm(a)
        scipy_imag_norm = float(np.linalg.norm(np.imag(ref), ord="fro"))
        ref_real = np.real(ref)
        scipy_diff_fro = float(np.linalg.norm(x - ref_real, ord="fro"))

    return SqrtReport(
        name=name,
        shape=a.shape,
        min_eig_input=float(eigvals.min()),
        max_eig_input=float(eigvals.max()),
        residual_fro=residual_fro,
        relative_residual=relative_residual,
        symmetry_error=symmetry_error,
        scipy_diff_fro=scipy_diff_fro,
        scipy_imag_norm=scipy_imag_norm,
    )


def run_checks(report: SqrtReport) -> None:
    """Assert numerical quality thresholds for one report."""
    if report.relative_residual >= 1e-10:
        raise AssertionError(
            f"{report.name}: relative residual too large: {report.relative_residual:.3e}"
        )

    if report.symmetry_error >= 1e-12:
        raise AssertionError(f"{report.name}: symmetry error too large: {report.symmetry_error:.3e}")

    if report.scipy_imag_norm is not None and report.scipy_imag_norm >= 1e-10:
        raise AssertionError(
            f"{report.name}: scipy sqrtm imaginary part too large: {report.scipy_imag_norm:.3e}"
        )

    if report.scipy_diff_fro is not None and report.scipy_diff_fro >= 1e-8:
        raise AssertionError(
            f"{report.name}: difference vs scipy sqrtm too large: {report.scipy_diff_fro:.3e}"
        )


def build_demo_matrices() -> dict[str, np.ndarray]:
    """Create deterministic SPD test matrices."""
    handcrafted = np.array(
        [
            [4.0, 1.0, 1.0],
            [1.0, 3.0, 0.5],
            [1.0, 0.5, 2.5],
        ]
    )

    rng = np.random.default_rng(2026)
    g = rng.normal(size=(5, 5))
    random_spd = g.T @ g + 0.8 * np.eye(5)

    return {
        "handcrafted_spd_3x3": handcrafted,
        "random_spd_5x5": random_spd,
    }


def main() -> None:
    print("Matrix square root MVP (SPD via eigen-decomposition)")
    print("=" * 60)

    for case_name, a in build_demo_matrices().items():
        report = analyze_case(case_name, a)
        run_checks(report)

        print(f"Case: {report.name}")
        print(f"  shape                 : {report.shape}")
        print(
            f"  eig range             : [{report.min_eig_input:.6e}, {report.max_eig_input:.6e}]"
        )
        print(f"  residual_fro          : {report.residual_fro:.6e}")
        print(f"  relative_residual     : {report.relative_residual:.6e}")
        print(f"  symmetry_error        : {report.symmetry_error:.6e}")

        if report.scipy_diff_fro is None:
            print("  scipy check           : skipped (scipy not available)")
        else:
            print(f"  scipy_diff_fro        : {report.scipy_diff_fro:.6e}")
            print(f"  scipy_imag_norm       : {report.scipy_imag_norm:.6e}")

        print("-" * 60)

    print("All checks passed.")


if __name__ == "__main__":
    main()

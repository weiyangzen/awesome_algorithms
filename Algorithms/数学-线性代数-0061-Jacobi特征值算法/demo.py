"""Minimal runnable MVP for Jacobi eigenvalue algorithm (MATH-0061).

This demo implements the symmetric Jacobi rotation method explicitly:
- pick the largest off-diagonal entry as pivot
- apply a Jacobi plane rotation to annihilate that entry
- iterate until the off-diagonal norm is below tolerance
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class JacobiEigenResult:
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    diagonalized: np.ndarray
    iterations: int
    converged: bool
    final_offdiag_norm: float
    offdiag_history: List[float]
    max_offdiag_history: List[float]


def validate_inputs(a: np.ndarray, max_sweeps: int, tol: float) -> None:
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Input matrix A must be square.")
    if not np.isfinite(a).all():
        raise ValueError("Input matrix A must contain only finite values.")
    if max_sweeps <= 0:
        raise ValueError("max_sweeps must be positive.")
    if tol <= 0.0:
        raise ValueError("tol must be positive.")


def strict_lower_fro_norm(a: np.ndarray) -> float:
    """Return ||tril(A, -1)||_F."""
    return float(np.linalg.norm(np.tril(a, k=-1), ord="fro"))


def largest_offdiag_index(a: np.ndarray) -> Tuple[int, int, float]:
    """Return (p, q, |a[p,q]|) for the largest off-diagonal entry, with p < q."""
    n = a.shape[0]
    if n < 2:
        return 0, 0, 0.0

    iu, ju = np.triu_indices(n, k=1)
    abs_vals = np.abs(a[iu, ju])
    idx = int(np.argmax(abs_vals))
    p = int(iu[idx])
    q = int(ju[idx])
    return p, q, float(abs_vals[idx])


def apply_jacobi_rotation_inplace(a: np.ndarray, v: np.ndarray, p: int, q: int) -> None:
    """Apply one Jacobi rotation that annihilates a[p, q].

    Updates A <- J^T A J and V <- V J in place.
    """
    if p == q:
        return

    apq = float(a[p, q])
    if abs(apq) < 1e-30:
        return

    app = float(a[p, p])
    aqq = float(a[q, q])

    tau = (aqq - app) / (2.0 * apq)
    if tau >= 0.0:
        t = 1.0 / (tau + np.sqrt(1.0 + tau * tau))
    else:
        t = -1.0 / (-tau + np.sqrt(1.0 + tau * tau))

    c = 1.0 / np.sqrt(1.0 + t * t)
    s = t * c

    n = a.shape[0]
    for i in range(n):
        if i == p or i == q:
            continue
        aip = float(a[i, p])
        aiq = float(a[i, q])

        new_ip = c * aip - s * aiq
        new_iq = s * aip + c * aiq

        a[i, p] = new_ip
        a[p, i] = new_ip
        a[i, q] = new_iq
        a[q, i] = new_iq

    a[p, p] = c * c * app - 2.0 * s * c * apq + s * s * aqq
    a[q, q] = s * s * app + 2.0 * s * c * apq + c * c * aqq
    a[p, q] = 0.0
    a[q, p] = 0.0

    vp = v[:, p].copy()
    vq = v[:, q].copy()
    v[:, p] = c * vp - s * vq
    v[:, q] = s * vp + c * vq


def jacobi_eigen_symmetric(
    a: np.ndarray,
    max_sweeps: int = 50,
    tol: float = 1e-12,
) -> JacobiEigenResult:
    """Compute all eigenvalues/eigenvectors of a real symmetric matrix using Jacobi rotations."""
    validate_inputs(a, max_sweeps=max_sweeps, tol=tol)

    if not np.allclose(a, a.T, atol=1e-12, rtol=0.0):
        raise ValueError("This MVP expects a symmetric matrix.")

    n = a.shape[0]
    ak = a.astype(float).copy()
    eigvecs = np.eye(n, dtype=float)

    offdiag_history: List[float] = []
    max_offdiag_history: List[float] = []
    iterations = 0
    converged = False

    if n == 1:
        return JacobiEigenResult(
            eigenvalues=np.array([float(ak[0, 0])]),
            eigenvectors=np.array([[1.0]]),
            diagonalized=ak,
            iterations=0,
            converged=True,
            final_offdiag_norm=0.0,
            offdiag_history=[0.0],
            max_offdiag_history=[0.0],
        )

    max_rotations = max_sweeps * n * (n - 1) // 2

    for _ in range(max_rotations):
        offdiag = strict_lower_fro_norm(ak)
        p, q, max_offdiag = largest_offdiag_index(ak)

        offdiag_history.append(offdiag)
        max_offdiag_history.append(max_offdiag)

        if offdiag <= tol or max_offdiag <= tol:
            converged = True
            break

        apply_jacobi_rotation_inplace(ak, eigvecs, p, q)
        iterations += 1

    final_offdiag = strict_lower_fro_norm(ak)
    if not converged and final_offdiag <= tol:
        converged = True

    eigvals = np.diag(ak).copy()
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    return JacobiEigenResult(
        eigenvalues=eigvals,
        eigenvectors=eigvecs,
        diagonalized=ak,
        iterations=iterations,
        converged=converged,
        final_offdiag_norm=final_offdiag,
        offdiag_history=offdiag_history,
        max_offdiag_history=max_offdiag_history,
    )


def run_checks(a: np.ndarray, result: JacobiEigenResult, ref_eigs: np.ndarray) -> None:
    if not result.converged:
        raise AssertionError("Jacobi iteration did not converge within the sweep budget.")
    if not np.isfinite(result.eigenvalues).all():
        raise AssertionError("Estimated eigenvalues contain non-finite values.")

    eig_err = float(np.max(np.abs(np.sort(result.eigenvalues) - np.sort(ref_eigs))))
    if eig_err > 1e-8:
        raise AssertionError(f"Eigenvalue error too large: {eig_err:.3e}")

    v = result.eigenvectors
    ortho_err = float(np.linalg.norm(v.T @ v - np.eye(v.shape[1]), ord="fro"))
    if ortho_err > 1e-8:
        raise AssertionError(f"Eigenvector orthogonality error too large: {ortho_err:.3e}")

    recon = v @ np.diag(result.eigenvalues) @ v.T
    recon_err = float(np.linalg.norm(a - recon, ord="fro"))
    if recon_err > 1e-8:
        raise AssertionError(f"Reconstruction error too large: {recon_err:.3e}")

    if result.final_offdiag_norm > 1e-8:
        raise AssertionError(f"Final off-diagonal norm too large: {result.final_offdiag_norm:.3e}")


def build_demo_matrix() -> np.ndarray:
    """Build a deterministic symmetric matrix with known spectrum."""
    rng = np.random.default_rng(61)
    q, _ = np.linalg.qr(rng.standard_normal((6, 6)))
    target_eigs = np.array([7.0, 4.5, 2.0, 0.5, -1.0, -3.0], dtype=float)
    a = q @ np.diag(target_eigs) @ q.T
    return 0.5 * (a + a.T)


def main() -> None:
    a = build_demo_matrix()
    result = jacobi_eigen_symmetric(a, max_sweeps=50, tol=1e-12)
    ref_eigs = np.linalg.eigvalsh(a)

    run_checks(a, result, ref_eigs)

    eig_err = float(np.max(np.abs(np.sort(result.eigenvalues) - np.sort(ref_eigs))))
    trace_err = float(abs(np.sum(result.eigenvalues) - np.trace(a)))
    v = result.eigenvectors
    ortho_err = float(np.linalg.norm(v.T @ v - np.eye(v.shape[1]), ord="fro"))
    recon_err = float(np.linalg.norm(a - (v @ np.diag(result.eigenvalues) @ v.T), ord="fro"))

    print("Jacobi eigenvalue demo (symmetric matrix)")
    print(f"matrix_shape={a.shape}")
    print(f"iterations={result.iterations}")
    print(f"converged={result.converged}")
    print(f"final_offdiag_norm={result.final_offdiag_norm:.3e}")
    print(f"max_abs_eig_error={eig_err:.3e}")
    print(f"trace_error={trace_err:.3e}")
    print(f"orthogonality_error={ortho_err:.3e}")
    print(f"reconstruction_error={recon_err:.3e}")
    print("estimated_eigs_sorted=", np.sort(result.eigenvalues))
    print("reference_eigs_sorted=", np.sort(ref_eigs))

    if result.offdiag_history:
        head = result.offdiag_history[:5]
        tail = result.offdiag_history[-3:]
        print(f"offdiag_history_head={[f'{v0:.3e}' for v0 in head]}")
        print(f"offdiag_history_tail={[f'{v0:.3e}' for v0 in tail]}")
    else:
        print("offdiag_history=[]")

    print("nonsymmetric input check:")
    bad = a.copy()
    bad[0, 1] += 1e-2
    try:
        jacobi_eigen_symmetric(bad)
    except ValueError as exc:
        print(f"  expected failure: {exc}")
    else:
        raise AssertionError("Nonsymmetric matrix should have raised ValueError.")

    print("All checks passed.")


if __name__ == "__main__":
    main()

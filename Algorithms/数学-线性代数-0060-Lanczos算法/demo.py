"""Minimal runnable MVP for Lanczos iteration on symmetric matrices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LanczosResult:
    """Container for Lanczos basis, tridiagonal matrix, and diagnostics."""

    Qm: np.ndarray
    Tm: np.ndarray
    m_effective: int
    breakdown: bool
    beta_next: float
    q_next: np.ndarray
    orthogonality_error: float
    lanczos_relation_error: float
    projected_matrix_error: float
    off_tridiag_violation: float


def validate_inputs(A: np.ndarray, b: np.ndarray, m: int, tol: float) -> None:
    """Validate basic shape and numeric assumptions."""
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square 2D matrix.")
    n = A.shape[0]
    if b.ndim != 1 or b.shape[0] != n:
        raise ValueError("b must be a 1D vector with length equal to A.shape[0].")
    if m <= 0:
        raise ValueError("m must be positive.")
    if tol <= 0.0:
        raise ValueError("tol must be positive.")
    if np.linalg.norm(b) == 0.0:
        raise ValueError("b must be non-zero.")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
        raise ValueError("A and b must contain only finite numbers.")

    symmetry_error = np.linalg.norm(A - A.T, ord=np.inf)
    if symmetry_error > 1e-12:
        raise ValueError(f"A must be symmetric. ||A-A^T||_inf={symmetry_error:.3e}")


def off_tridiagonal_violation(M: np.ndarray) -> float:
    """Return max absolute value outside the tridiagonal band."""
    n = M.shape[0]
    if n <= 2:
        return 0.0
    i = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    mask = np.abs(i - j) > 1
    if not np.any(mask):
        return 0.0
    return float(np.max(np.abs(M[mask])))


def max_min_eigen_distance(approx: np.ndarray, reference: np.ndarray) -> float:
    """For each approx eigenvalue, take nearest reference distance and then max."""
    if approx.size == 0:
        return 0.0
    distances = [float(np.min(np.abs(lam - reference))) for lam in approx]
    return float(max(distances))


def lanczos_iteration(
    A: np.ndarray,
    b: np.ndarray,
    m: int,
    tol: float = 1e-12,
    reorthogonalize: bool = True,
) -> LanczosResult:
    """Run Lanczos with optional double full reorthogonalization."""
    validate_inputs(A, b, m, tol)

    n = A.shape[0]
    m_use = min(m, n)

    Q = np.zeros((n, m_use), dtype=float)
    alphas = np.zeros(m_use, dtype=float)
    betas = np.zeros(max(0, m_use - 1), dtype=float)

    q_prev = np.zeros(n, dtype=float)
    q = b / np.linalg.norm(b)
    beta_prev = 0.0

    breakdown = False
    beta_next = np.nan
    q_next = np.zeros(n, dtype=float)
    m_effective = 0

    for k in range(m_use):
        Q[:, k] = q

        z = A @ q
        if k > 0:
            z = z - beta_prev * q_prev

        alpha = float(np.dot(q, z))
        z = z - alpha * q

        if reorthogonalize and k >= 0:
            basis = Q[:, : k + 1]
            # Two passes make orthogonality much more stable in finite precision.
            coeff1 = basis.T @ z
            z = z - basis @ coeff1
            coeff2 = basis.T @ z
            z = z - basis @ coeff2

        beta = float(np.linalg.norm(z))

        alphas[k] = alpha
        m_effective = k + 1
        beta_next = beta

        if k < m_use - 1:
            betas[k] = beta

        if beta <= tol:
            breakdown = True
            q_next = np.zeros(n, dtype=float)
            break

        q_next = z / beta
        q_prev = q
        q = q_next
        beta_prev = beta

    Qm = Q[:, :m_effective]

    Tm = np.diag(alphas[:m_effective])
    if m_effective > 1:
        off = betas[: m_effective - 1]
        Tm = Tm + np.diag(off, k=1) + np.diag(off, k=-1)

    gram_error = float(np.linalg.norm(Qm.T @ Qm - np.eye(m_effective), ord=2))

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        AQ = A @ Qm
        projected = Qm.T @ AQ
    if not np.all(np.isfinite(projected)) or not np.all(np.isfinite(AQ)):
        raise FloatingPointError("Non-finite values detected during projected matrix build.")
    projected_matrix_error = float(np.linalg.norm(projected - Tm, ord=2))
    tri_violation = off_tridiagonal_violation(projected)
    QT = Qm @ Tm

    if breakdown:
        relation_residual = AQ - QT
    else:
        e_last = np.zeros(m_effective, dtype=float)
        e_last[-1] = 1.0
        correction = beta_next * np.outer(q_next, e_last)
        relation_residual = AQ - QT - correction

    lanczos_relation_error = float(np.linalg.norm(relation_residual, ord=2))

    return LanczosResult(
        Qm=Qm,
        Tm=Tm,
        m_effective=m_effective,
        breakdown=breakdown,
        beta_next=beta_next,
        q_next=q_next,
        orthogonality_error=gram_error,
        lanczos_relation_error=lanczos_relation_error,
        projected_matrix_error=projected_matrix_error,
        off_tridiag_violation=tri_violation,
    )


def build_symmetric_test_matrix(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Build a deterministic symmetric matrix and return its reference spectrum."""
    main_diag = np.linspace(2.0, 3.6, n)
    off_diag = np.linspace(0.3, 0.55, max(1, n - 1))

    A = np.diag(main_diag)
    if n > 1:
        A = A + np.diag(off_diag[: n - 1], k=1) + np.diag(off_diag[: n - 1], k=-1)

    # Add a small dense symmetric rank-1 term so A is not purely tridiagonal.
    u = np.linspace(1.0, 2.0, n)
    A = A + 0.02 * np.outer(u, u)
    A = 0.5 * (A + A.T)

    eigvals = np.linalg.eigvalsh(A)
    return A, eigvals


def run_checks(result: LanczosResult, ritz_gap: float) -> None:
    """Raise errors when key Lanczos invariants are violated."""
    if result.m_effective <= 0:
        raise AssertionError("Lanczos produced an empty basis.")
    if result.orthogonality_error > 1e-12:
        raise AssertionError(
            f"Orthogonality error too large: {result.orthogonality_error:.3e}"
        )
    if result.lanczos_relation_error > 1e-12:
        raise AssertionError(
            f"Lanczos relation residual too large: {result.lanczos_relation_error:.3e}"
        )
    if result.projected_matrix_error > 1e-12:
        raise AssertionError(
            f"Projected matrix mismatch too large: {result.projected_matrix_error:.3e}"
        )
    if result.off_tridiag_violation > 1e-12:
        raise AssertionError(
            f"Projected matrix is not tridiagonal enough: {result.off_tridiag_violation:.3e}"
        )
    if ritz_gap > 1e-9:
        raise AssertionError(f"Ritz spectrum mismatch too large: {ritz_gap:.3e}")


def main() -> None:
    n = 9
    m = n
    tol = 1e-14

    A, ref_eigs = build_symmetric_test_matrix(n)

    b = np.arange(1, n + 1, dtype=float)

    result = lanczos_iteration(A=A, b=b, m=m, tol=tol, reorthogonalize=True)

    ritz = np.linalg.eigvalsh(result.Tm)
    ritz_gap = max_min_eigen_distance(ritz, ref_eigs)

    run_checks(result=result, ritz_gap=ritz_gap)

    print("Lanczos MVP report")
    print(f"matrix_size                  : {n}")
    print(f"requested_steps              : {m}")
    print(f"effective_steps              : {result.m_effective}")
    print(f"breakdown_detected           : {result.breakdown}")
    print(f"next_beta                    : {result.beta_next:.3e}")
    print(f"orthogonality_error          : {result.orthogonality_error:.3e}")
    print(f"lanczos_relation_error       : {result.lanczos_relation_error:.3e}")
    print(f"projected_matrix_error       : {result.projected_matrix_error:.3e}")
    print(f"off_tridiagonal_violation    : {result.off_tridiag_violation:.3e}")
    print(f"max_ritz_to_ref_eig_dist     : {ritz_gap:.3e}")

    print("\nRitz eigenvalues (Tm):")
    for lam in ritz:
        print(f"  {lam:+.8f}")

    print("\nReference eigenvalues (A):")
    for lam in ref_eigs:
        print(f"  {lam:+.8f}")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()

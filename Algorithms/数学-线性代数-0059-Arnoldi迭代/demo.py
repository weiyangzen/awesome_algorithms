"""Minimal runnable MVP for Arnoldi iteration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ArnoldiResult:
    """Container for the Arnoldi basis and diagnostic errors."""

    Qm: np.ndarray
    Qext: np.ndarray
    Hm: np.ndarray
    Hbar: np.ndarray
    m_effective: int
    breakdown: bool
    next_subdiag: float
    orthogonality_error: float
    arnoldi_relation_error: float


def validate_inputs(A: np.ndarray, b: np.ndarray, m: int, tol: float) -> None:
    """Validate shapes and numerical sanity."""
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square 2D matrix.")
    n = A.shape[0]
    if b.ndim != 1 or b.shape[0] != n:
        raise ValueError("b must be a 1D vector with length equal to A.shape[0].")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
        raise ValueError("A and b must contain only finite numbers.")
    if m <= 0:
        raise ValueError("m must be positive.")
    if tol <= 0:
        raise ValueError("tol must be positive.")
    if np.linalg.norm(b) == 0.0:
        raise ValueError("b must be non-zero.")


def arnoldi_iteration(
    A: np.ndarray,
    b: np.ndarray,
    m: int,
    tol: float = 1e-12,
    reorthogonalize: bool = True,
) -> ArnoldiResult:
    """Run Arnoldi iteration with Modified Gram-Schmidt orthogonalization."""
    validate_inputs(A, b, m, tol)

    n = A.shape[0]
    m_use = min(m, n)

    Q = np.zeros((n, m_use + 1), dtype=float)
    H = np.zeros((m_use + 1, m_use), dtype=float)

    beta = np.linalg.norm(b)
    Q[:, 0] = b / beta

    breakdown = False
    next_subdiag = np.nan
    m_effective = 0

    for k in range(m_use):
        v = A @ Q[:, k]

        # First MGS pass.
        for j in range(k + 1):
            hij = float(np.dot(Q[:, j], v))
            H[j, k] = hij
            v = v - hij * Q[:, j]

        # Optional second pass improves orthogonality in finite precision.
        if reorthogonalize:
            for j in range(k + 1):
                correction = float(np.dot(Q[:, j], v))
                H[j, k] += correction
                v = v - correction * Q[:, j]

        H[k + 1, k] = np.linalg.norm(v)
        next_subdiag = float(H[k + 1, k])
        m_effective = k + 1

        if H[k + 1, k] <= tol:
            breakdown = True
            break

        Q[:, k + 1] = v / H[k + 1, k]

    Qm = Q[:, :m_effective]
    Hm = H[:m_effective, :m_effective]

    if breakdown:
        # Invariant subspace reached: AQm = Qm Hm.
        Qext = Qm
        Hbar = Hm
    else:
        Qext = Q[:, : m_effective + 1]
        Hbar = H[: m_effective + 1, :m_effective]

    orthogonality_error = float(np.linalg.norm(Qm.T @ Qm - np.eye(m_effective), ord=2))
    arnoldi_relation_error = float(np.linalg.norm(A @ Qm - Qext @ Hbar, ord=2))

    return ArnoldiResult(
        Qm=Qm,
        Qext=Qext,
        Hm=Hm,
        Hbar=Hbar,
        m_effective=m_effective,
        breakdown=breakdown,
        next_subdiag=next_subdiag,
        orthogonality_error=orthogonality_error,
        arnoldi_relation_error=arnoldi_relation_error,
    )


def build_cyclic_shift_matrix(n: int) -> np.ndarray:
    """Return a deterministic non-symmetric orthogonal matrix."""
    eye = np.eye(n)
    return np.roll(eye, shift=1, axis=0)


def hessenberg_violation(H: np.ndarray) -> float:
    """Maximum absolute value below the first subdiagonal."""
    if H.shape[0] <= 2:
        return 0.0
    mask = np.tril(np.ones_like(H, dtype=bool), k=-2)
    if not np.any(mask):
        return 0.0
    return float(np.max(np.abs(H[mask])))


def max_min_eigen_distance(approx: np.ndarray, reference: np.ndarray) -> float:
    """For each approx eigenvalue, compute nearest reference distance and take max."""
    distances = [float(np.min(np.abs(lam - reference))) for lam in approx]
    return float(max(distances)) if distances else 0.0


def run_checks(result: ArnoldiResult, hess_error: float, ritz_gap: float) -> None:
    """Fail fast when core Arnoldi invariants are violated."""
    if result.m_effective <= 0:
        raise AssertionError("Arnoldi produced an empty basis.")
    if result.orthogonality_error > 1e-12:
        raise AssertionError(f"Orthogonality error too large: {result.orthogonality_error:.3e}")
    if result.arnoldi_relation_error > 1e-12:
        raise AssertionError(
            f"Arnoldi relation residual too large: {result.arnoldi_relation_error:.3e}"
        )
    if hess_error > 1e-12:
        raise AssertionError(f"Hessenberg structure broken: {hess_error:.3e}")
    if ritz_gap > 1e-10:
        raise AssertionError(f"Ritz spectrum mismatch too large: {ritz_gap:.3e}")


def main() -> None:
    n = 8
    m = n
    tol = 1e-14

    A = build_cyclic_shift_matrix(n)
    b = np.zeros(n, dtype=float)
    b[0] = 1.0

    result = arnoldi_iteration(A=A, b=b, m=m, tol=tol, reorthogonalize=True)

    ritz_values = np.linalg.eigvals(result.Hm)
    ref_values = np.linalg.eigvals(A)

    hess_error = hessenberg_violation(result.Hm)
    ritz_gap = max_min_eigen_distance(ritz_values, ref_values)

    run_checks(result=result, hess_error=hess_error, ritz_gap=ritz_gap)

    print("Arnoldi MVP report")
    print(f"matrix_size                : {n}")
    print(f"requested_steps            : {m}")
    print(f"effective_steps            : {result.m_effective}")
    print(f"breakdown_detected         : {result.breakdown}")
    print(f"next_subdiag_h(m+1,m)      : {result.next_subdiag:.3e}")
    print(f"orthogonality_error        : {result.orthogonality_error:.3e}")
    print(f"arnoldi_relation_error     : {result.arnoldi_relation_error:.3e}")
    print(f"hessenberg_violation       : {hess_error:.3e}")
    print(f"max_ritz_to_ref_eig_dist   : {ritz_gap:.3e}")

    print("\nRitz eigenvalues (Hm):")
    for lam in ritz_values:
        print(f"  {lam.real:+.6f}{lam.imag:+.6f}j")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()

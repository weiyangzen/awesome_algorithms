"""Minimal runnable MVP for Givens rotations and Householder transforms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class QRResult:
    """Container for one QR factorization and diagnostics."""

    method: str
    Q: np.ndarray
    R: np.ndarray
    orthogonality_error: float
    reconstruction_error: float
    lower_triangle_error: float


def validate_matrix(A: np.ndarray) -> None:
    """Validate basic assumptions for this MVP."""
    if A.ndim != 2:
        raise ValueError("A must be a 2D matrix.")
    m, n = A.shape
    if m < n:
        raise ValueError("This demo expects m >= n (tall/square matrix).")
    if not np.all(np.isfinite(A)):
        raise ValueError("A must contain only finite numbers.")
    if np.linalg.matrix_rank(A) < n:
        raise ValueError("A must have full column rank for back-substitution.")


def build_qr_result(method: str, A: np.ndarray, Q: np.ndarray, R: np.ndarray) -> QRResult:
    """Compute common diagnostics for a QR pair."""
    m = Q.shape[0]
    orthogonality_error = float(np.linalg.norm(Q.T @ Q - np.eye(m), ord=2))
    reconstruction_error = float(np.linalg.norm(A - Q @ R, ord=2))
    lower_triangle_error = float(np.linalg.norm(np.tril(R, k=-1), ord="fro"))
    return QRResult(
        method=method,
        Q=Q,
        R=R,
        orthogonality_error=orthogonality_error,
        reconstruction_error=reconstruction_error,
        lower_triangle_error=lower_triangle_error,
    )


def householder_vector(x: np.ndarray) -> np.ndarray:
    """Return a unit Householder vector v so that H=I-2vv^T reflects x to axis direction."""
    norm_x = float(np.linalg.norm(x))
    if norm_x == 0.0:
        return np.zeros_like(x)

    v = x.astype(float, copy=True)
    sign = 1.0 if x[0] >= 0.0 else -1.0
    v[0] += sign * norm_x

    norm_v = float(np.linalg.norm(v))
    if norm_v == 0.0:
        return np.zeros_like(x)
    return v / norm_v


def householder_qr(A: np.ndarray) -> QRResult:
    """Compute full QR using explicit Householder reflections."""
    validate_matrix(A)
    m, n = A.shape

    R = A.astype(float, copy=True)
    Q = np.eye(m, dtype=float)

    for k in range(n):
        v = householder_vector(R[k:, k])
        if np.allclose(v, 0.0):
            continue

        # R <- H_k R on the trailing block.
        R_block = R[k:, k:]
        R[k:, k:] = R_block - 2.0 * np.outer(v, v @ R_block)

        # Q <- Q H_k (right-multiply on trailing columns).
        Q_block = Q[:, k:]
        Q[:, k:] = Q_block - 2.0 * np.outer(Q_block @ v, v)

    return build_qr_result(method="Householder", A=A, Q=Q, R=R)


def givens_coeffs(a: float, b: float) -> tuple[float, float]:
    """Return (c, s) such that [[c,s],[-s,c]] @ [a,b]^T = [r,0]^T."""
    if b == 0.0:
        return 1.0, 0.0
    r = float(np.hypot(a, b))
    return a / r, b / r


def apply_givens_left(M: np.ndarray, i: int, j: int, c: float, s: float, start_col: int) -> None:
    """Left-multiply rows i,j of M by one Givens rotation from start_col onward."""
    row_i = c * M[i, start_col:] + s * M[j, start_col:]
    row_j = -s * M[i, start_col:] + c * M[j, start_col:]
    M[i, start_col:] = row_i
    M[j, start_col:] = row_j


def givens_qr(A: np.ndarray) -> QRResult:
    """Compute full QR using explicit Givens rotations."""
    validate_matrix(A)
    m, n = A.shape

    R = A.astype(float, copy=True)
    Qt = np.eye(m, dtype=float)

    for col in range(n):
        for row in range(m - 1, col, -1):
            a = float(R[row - 1, col])
            b = float(R[row, col])
            if np.isclose(b, 0.0):
                continue
            c, s = givens_coeffs(a, b)

            apply_givens_left(R, row - 1, row, c, s, start_col=col)
            apply_givens_left(Qt, row - 1, row, c, s, start_col=0)

    Q = Qt.T
    return build_qr_result(method="Givens", A=A, Q=Q, R=R)


def least_squares_via_qr(Q: np.ndarray, R: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve min ||Ax-b|| via reduced QR blocks extracted from full Q,R."""
    n = R.shape[1]
    Q1 = Q[:, :n]
    R1 = R[:n, :]
    y = Q1.T @ b
    return np.linalg.solve(R1, y)


def run_checks(
    hh: QRResult,
    gv: QRResult,
    x_hh: np.ndarray,
    x_gv: np.ndarray,
    x_ref: np.ndarray,
    res_hh: float,
    res_gv: float,
    res_ref: float,
) -> None:
    """Fail fast when key numerical guarantees are broken."""
    for result in (hh, gv):
        if result.orthogonality_error > 1e-12:
            raise AssertionError(
                f"{result.method}: orthogonality error too large: {result.orthogonality_error:.3e}"
            )
        if result.reconstruction_error > 1e-11:
            raise AssertionError(
                f"{result.method}: reconstruction error too large: {result.reconstruction_error:.3e}"
            )
        if result.lower_triangle_error > 1e-11:
            raise AssertionError(
                f"{result.method}: lower-triangle residual too large: {result.lower_triangle_error:.3e}"
            )

    if np.linalg.norm(x_hh - x_ref) > 1e-10:
        raise AssertionError("Householder least-squares solution deviates from lstsq reference.")
    if np.linalg.norm(x_gv - x_ref) > 1e-10:
        raise AssertionError("Givens least-squares solution deviates from lstsq reference.")

    if abs(res_hh - res_ref) > 1e-10:
        raise AssertionError("Householder residual norm mismatch with reference.")
    if abs(res_gv - res_ref) > 1e-10:
        raise AssertionError("Givens residual norm mismatch with reference.")


def main() -> None:
    rng = np.random.default_rng(2026)
    m, n = 8, 5

    A = rng.normal(size=(m, n))
    A[:n, :] += 0.75 * np.eye(n)
    b = rng.normal(size=m)

    validate_matrix(A)

    hh = householder_qr(A)
    gv = givens_qr(A)

    x_hh = least_squares_via_qr(hh.Q, hh.R, b)
    x_gv = least_squares_via_qr(gv.Q, gv.R, b)
    x_ref, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    res_hh = float(np.linalg.norm(A @ x_hh - b))
    res_gv = float(np.linalg.norm(A @ x_gv - b))
    res_ref = float(np.linalg.norm(A @ x_ref - b))

    run_checks(
        hh=hh,
        gv=gv,
        x_hh=x_hh,
        x_gv=x_gv,
        x_ref=x_ref,
        res_hh=res_hh,
        res_gv=res_gv,
        res_ref=res_ref,
    )

    print("Givens & Householder MVP report")
    print(f"shape                          : {m} x {n}")
    print()
    for result in (hh, gv):
        print(f"[{result.method}]")
        print(f"orthogonality_error            : {result.orthogonality_error:.3e}")
        print(f"reconstruction_error           : {result.reconstruction_error:.3e}")
        print(f"lower_triangle_error           : {result.lower_triangle_error:.3e}")
        print()

    print("[Least-squares comparison]")
    print(f"||x_householder - x_ref||      : {np.linalg.norm(x_hh - x_ref):.3e}")
    print(f"||x_givens - x_ref||           : {np.linalg.norm(x_gv - x_ref):.3e}")
    print(f"residual_householder           : {res_hh:.3e}")
    print(f"residual_givens                : {res_gv:.3e}")
    print(f"residual_reference_lstsq       : {res_ref:.3e}")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()

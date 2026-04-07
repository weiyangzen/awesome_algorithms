"""Minimal runnable MVP for QR decomposition (Householder reflections)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class QRResult:
    q: np.ndarray
    r: np.ndarray
    reconstruction_error: float
    orthogonality_error: float
    lower_triangle_norm: float


def validate_matrix(a: np.ndarray) -> np.ndarray:
    """Return a finite 2-D float matrix with m>=n, or raise ValueError."""
    arr = np.asarray(a, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"A must be 2-D, got ndim={arr.ndim}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("A contains non-finite values")
    m, n = arr.shape
    if m < n:
        raise ValueError(f"This MVP expects m>=n, got shape={arr.shape}")
    return arr


def householder_qr(a: np.ndarray, tol: float = 1e-12) -> QRResult:
    """Compute thin QR decomposition A=QR by explicit Householder reflections."""
    if tol <= 0:
        raise ValueError("tol must be > 0")

    mat = validate_matrix(a)
    m, n = mat.shape

    q_full = np.eye(m, dtype=float)
    r_full = mat.copy()

    for k in range(n):
        x = r_full[k:, k]
        norm_x = float(np.linalg.norm(x))
        if norm_x <= tol:
            continue

        sign = 1.0 if x[0] >= 0.0 else -1.0
        u = x.copy()
        u[0] += sign * norm_x
        norm_u = float(np.linalg.norm(u))
        if norm_u <= tol:
            continue

        v = u / norm_u

        # Apply reflector on R: R_k <- (I - 2vv^T) R_k
        r_block = r_full[k:, k:]
        r_full[k:, k:] = r_block - 2.0 * np.outer(v, v @ r_block)

        # Accumulate Q on the right side of previously built orthogonal basis.
        q_block = q_full[:, k:]
        q_full[:, k:] = q_block - 2.0 * np.outer(q_block @ v, v)

    q = q_full[:, :n].copy()
    r = r_full[:n, :].copy()

    # Clean numeric noise below diagonal to make upper-triangular structure explicit.
    for i in range(n):
        r[i + 1 :, i] = 0.0

    reconstruction_error = float(np.linalg.norm(mat - q @ r, ord="fro"))
    orthogonality_error = float(np.linalg.norm(q.T @ q - np.eye(n), ord="fro"))
    lower_triangle_norm = float(np.linalg.norm(np.tril(r, k=-1), ord="fro"))

    return QRResult(
        q=q,
        r=r,
        reconstruction_error=reconstruction_error,
        orthogonality_error=orthogonality_error,
        lower_triangle_norm=lower_triangle_norm,
    )


def backward_substitution(r: np.ndarray, y: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Solve Rx=y where R is an upper-triangular square matrix."""
    upper = np.asarray(r, dtype=float)
    rhs = np.asarray(y, dtype=float)

    if upper.ndim != 2 or upper.shape[0] != upper.shape[1]:
        raise ValueError(f"R must be square, got shape={upper.shape}")
    if rhs.ndim != 1 or rhs.shape[0] != upper.shape[0]:
        raise ValueError("Dimension mismatch between R and y")

    n = upper.shape[0]
    x = np.zeros(n, dtype=float)

    for i in range(n - 1, -1, -1):
        diag = float(upper[i, i])
        if abs(diag) <= tol:
            raise ValueError(
                f"R is singular or rank-deficient at diagonal index {i} (|diag|={abs(diag):.3e})"
            )
        x[i] = (rhs[i] - np.dot(upper[i, i + 1 :], x[i + 1 :])) / diag

    return x


def solve_least_squares_qr(result: QRResult, b: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Solve min ||Ax-b||_2 via thin QR factors A=QR."""
    b_vec = np.asarray(b, dtype=float)
    if b_vec.ndim != 1:
        raise ValueError(f"b must be a vector, got ndim={b_vec.ndim}")
    if b_vec.shape[0] != result.q.shape[0]:
        raise ValueError("Dimension mismatch between Q and b")

    y = result.q.T @ b_vec
    x = backward_substitution(result.r, y, tol=tol)
    return x


def build_demo_problem() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct a deterministic full-column-rank least-squares problem."""
    a = np.array(
        [
            [1.0, 2.0, -1.0, 0.0],
            [2.0, -1.0, 0.0, 1.0],
            [0.0, 3.0, 1.0, -2.0],
            [1.0, 0.0, 2.0, 1.0],
            [2.0, 1.0, 1.0, 3.0],
            [-1.0, 2.0, 0.0, 2.0],
        ],
        dtype=float,
    )
    x_true = np.array([1.5, -2.0, 0.5, 3.0], dtype=float)
    b = a @ x_true
    return a, b, x_true


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)

    a, b, x_true = build_demo_problem()
    qr_result = householder_qr(a, tol=1e-12)
    x_qr = solve_least_squares_qr(qr_result, b, tol=1e-12)

    x_np, residuals, rank, singular_values = np.linalg.lstsq(a, b, rcond=None)

    residual_norm = float(np.linalg.norm(a @ x_qr - b, ord=2))
    solution_gap = float(np.linalg.norm(x_qr - x_np, ord=2))
    true_gap = float(np.linalg.norm(x_qr - x_true, ord=2))

    print("=== QR Decomposition MVP (Householder) ===")
    print(f"shape(A) = {a.shape}")
    print(f"rank(A) from NumPy = {rank}")
    print(f"singular_values(A) = {singular_values}")

    print("\n=== Factors ===")
    print("Q =")
    print(qr_result.q)
    print("R =")
    print(qr_result.r)

    print("\n=== Solve min ||Ax-b||_2 ===")
    print(f"x_true          = {x_true}")
    print(f"x (QR)          = {x_qr}")
    print(f"x (NumPy lstsq) = {x_np}")

    print("\n=== Quality Metrics ===")
    print(f"||A - Q@R||_F       = {qr_result.reconstruction_error:.3e}")
    print(f"||Q^TQ - I||_F      = {qr_result.orthogonality_error:.3e}")
    print(f"||tril(R,-1)||_F    = {qr_result.lower_triangle_norm:.3e}")
    print(f"||A@x - b||_2       = {residual_norm:.3e}")
    print(f"||x_qr - x_np||_2   = {solution_gap:.3e}")
    print(f"||x_qr - x_true||_2 = {true_gap:.3e}")
    if residuals.size > 0:
        print(f"NumPy residual sum  = {float(residuals[0]):.3e}")

    # Rank-deficient example: expect solve failure due to near-zero diagonal in R.
    a_rank_def = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    b_rank_def = np.array([1.0, 2.0, 0.0, 1.0], dtype=float)
    try:
        rd_result = householder_qr(a_rank_def, tol=1e-12)
        _ = solve_least_squares_qr(rd_result, b_rank_def, tol=1e-12)
    except ValueError as exc:
        print("\nExpected failure on rank-deficient system:")
        print(exc)


if __name__ == "__main__":
    main()

"""Minimal runnable MVP for the Moore-Penrose pseudoinverse."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PseudoInverseResult:
    """Container for pseudoinverse outputs and diagnostics."""

    A: np.ndarray
    A_pinv: np.ndarray
    singular_values: np.ndarray
    rcond: float
    cutoff: float
    effective_rank: int
    penrose_residual_1: float
    penrose_residual_2: float
    penrose_residual_3: float
    penrose_residual_4: float
    np_pinv_diff: float
    ls_residual_norm: float
    normal_eq_residual_norm: float
    lstsq_solution_diff: float


def validate_inputs(A: np.ndarray, rcond: float | None) -> None:
    """Validate matrix and threshold assumptions."""
    if A.ndim != 2:
        raise ValueError("A must be a 2D matrix.")
    if not np.all(np.isfinite(A)):
        raise ValueError("A must contain only finite numbers.")
    if rcond is not None and rcond < 0.0:
        raise ValueError("rcond must be non-negative.")


def build_rank_deficient_matrix() -> np.ndarray:
    """Build a deterministic rectangular rank-deficient matrix (5x3, rank=2)."""
    return np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [-1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [3.0, 6.0, 9.0],
        ],
        dtype=float,
    )


def moore_penrose_pinv_svd(A: np.ndarray, rcond: float | None = None) -> PseudoInverseResult:
    """
    Compute Moore-Penrose pseudoinverse via explicit SVD steps.

    For A = U Sigma V^T, pseudoinverse is A^+ = V Sigma^+ U^T where
    Sigma^+ inverts only singular values above a threshold.
    """
    validate_inputs(A, rcond)

    m, n = A.shape
    U, s, Vt = np.linalg.svd(A, full_matrices=False)

    if rcond is None:
        rcond_use = np.finfo(float).eps * max(m, n)
    else:
        rcond_use = float(rcond)

    sigma_max = float(s[0]) if s.size > 0 else 0.0
    cutoff = rcond_use * sigma_max

    s_inv = np.zeros_like(s)
    nonzero_mask = s > cutoff
    s_inv[nonzero_mask] = 1.0 / s[nonzero_mask]
    effective_rank = int(np.count_nonzero(nonzero_mask))

    A_pinv = (Vt.T * s_inv) @ U.T

    P1 = A @ A_pinv @ A
    P2 = A_pinv @ A @ A_pinv
    P3 = A @ A_pinv
    P4 = A_pinv @ A
    penrose_1 = float(np.linalg.norm(P1 - A, ord="fro"))
    penrose_2 = float(np.linalg.norm(P2 - A_pinv, ord="fro"))
    penrose_3 = float(np.linalg.norm(P3.T - P3, ord="fro"))
    penrose_4 = float(np.linalg.norm(P4.T - P4, ord="fro"))

    np_pinv_ref = np.linalg.pinv(A, rcond=rcond_use)
    np_pinv_diff = float(np.linalg.norm(A_pinv - np_pinv_ref, ord="fro"))

    b = np.array([1.0, 2.0, 0.0, 1.0, 3.0], dtype=float)
    x_pinv = A_pinv @ b
    residual = b - A @ x_pinv
    ls_residual_norm = float(np.linalg.norm(residual, ord=2))
    normal_eq_residual_norm = float(np.linalg.norm(A.T @ residual, ord=2))

    x_lstsq, *_ = np.linalg.lstsq(A, b, rcond=rcond_use)
    lstsq_solution_diff = float(np.linalg.norm(x_pinv - x_lstsq, ord=2))

    return PseudoInverseResult(
        A=A,
        A_pinv=A_pinv,
        singular_values=s,
        rcond=rcond_use,
        cutoff=cutoff,
        effective_rank=effective_rank,
        penrose_residual_1=penrose_1,
        penrose_residual_2=penrose_2,
        penrose_residual_3=penrose_3,
        penrose_residual_4=penrose_4,
        np_pinv_diff=np_pinv_diff,
        ls_residual_norm=ls_residual_norm,
        normal_eq_residual_norm=normal_eq_residual_norm,
        lstsq_solution_diff=lstsq_solution_diff,
    )


def run_checks(result: PseudoInverseResult) -> None:
    """Raise assertions when key identities are violated."""
    if result.effective_rank != 2:
        raise AssertionError(f"Unexpected effective rank: {result.effective_rank}")

    tol_penrose = 1e-10
    if result.penrose_residual_1 > tol_penrose:
        raise AssertionError(f"Penrose-1 residual too large: {result.penrose_residual_1:.3e}")
    if result.penrose_residual_2 > tol_penrose:
        raise AssertionError(f"Penrose-2 residual too large: {result.penrose_residual_2:.3e}")
    if result.penrose_residual_3 > tol_penrose:
        raise AssertionError(f"Penrose-3 residual too large: {result.penrose_residual_3:.3e}")
    if result.penrose_residual_4 > tol_penrose:
        raise AssertionError(f"Penrose-4 residual too large: {result.penrose_residual_4:.3e}")

    if result.np_pinv_diff > 1e-11:
        raise AssertionError(f"Difference to numpy.linalg.pinv too large: {result.np_pinv_diff:.3e}")
    if result.normal_eq_residual_norm > 1e-10:
        raise AssertionError(
            f"Least-squares normal equation residual too large: {result.normal_eq_residual_norm:.3e}"
        )
    if result.lstsq_solution_diff > 1e-10:
        raise AssertionError(
            f"Difference to numpy.linalg.lstsq solution too large: {result.lstsq_solution_diff:.3e}"
        )


def main() -> None:
    A = build_rank_deficient_matrix()
    result = moore_penrose_pinv_svd(A=A, rcond=None)
    run_checks(result)

    print("Moore-Penrose pseudoinverse MVP report")
    print(f"matrix_shape                 : {A.shape}")
    print(f"effective_rank               : {result.effective_rank}")
    print(f"rcond_used                   : {result.rcond:.3e}")
    print(f"singular_value_cutoff        : {result.cutoff:.3e}")
    print("singular_values              :")
    for s in result.singular_values:
        print(f"  {s:.8e}")

    print(f"penrose_residual_1           : {result.penrose_residual_1:.3e}")
    print(f"penrose_residual_2           : {result.penrose_residual_2:.3e}")
    print(f"penrose_residual_3           : {result.penrose_residual_3:.3e}")
    print(f"penrose_residual_4           : {result.penrose_residual_4:.3e}")
    print(f"np_pinv_difference           : {result.np_pinv_diff:.3e}")
    print(f"least_squares_residual_norm  : {result.ls_residual_norm:.3e}")
    print(f"normal_equation_residual     : {result.normal_eq_residual_norm:.3e}")
    print(f"lstsq_solution_difference    : {result.lstsq_solution_diff:.3e}")

    print("\nComputed pseudoinverse A^+:")
    print(result.A_pinv)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()

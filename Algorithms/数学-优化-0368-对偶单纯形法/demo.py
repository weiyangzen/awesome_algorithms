"""Minimal runnable MVP for the dual simplex method.

This script solves a small linear program in standard equality form:
    minimize c^T x
    subject to A x = b, x >= 0

The initial basis is dual-feasible but primal-infeasible, so dual simplex is
appropriate and can restore primal feasibility while preserving dual
feasibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


Array = np.ndarray


@dataclass
class IterationLog:
    iter_idx: int
    objective: float
    min_basic_value: float
    leaving_row: int
    leaving_var: int
    entering_var: int
    pivot_ratio: float


@dataclass
class DualSimplexResult:
    status: str
    x: Array
    objective: float
    iterations: int
    basis: List[int]
    history: List[IterationLog]


def complement_indices(n: int, basis: Sequence[int]) -> List[int]:
    basis_set = set(basis)
    return [j for j in range(n) if j not in basis_set]


def validate_inputs(A: Array, b: Array, c: Array, basis: Sequence[int]) -> None:
    if A.ndim != 2:
        raise ValueError("A must be a 2D matrix")

    m, n = A.shape
    if b.shape != (m,):
        raise ValueError(f"b must have shape ({m},)")
    if c.shape != (n,):
        raise ValueError(f"c must have shape ({n},)")
    if len(basis) != m:
        raise ValueError("basis length must equal number of constraints")
    if len(set(basis)) != len(basis):
        raise ValueError("basis indices must be unique")
    if min(basis) < 0 or max(basis) >= n:
        raise ValueError("basis indices out of range")


def dual_simplex(
    A: Array,
    b: Array,
    c: Array,
    basis: Sequence[int],
    tol: float = 1e-9,
    max_iter: int = 100,
) -> DualSimplexResult:
    """Solve a linear program by dual simplex from a dual-feasible basis.

    The LP form is:
        minimize c^T x
        subject to A x = b, x >= 0
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    basis = list(basis)

    validate_inputs(A, b, c, basis)

    m, n = A.shape
    history: List[IterationLog] = []

    for it in range(max_iter + 1):
        nonbasis = complement_indices(n, basis)

        B = A[:, basis]
        N = A[:, nonbasis]

        try:
            B_inv = np.linalg.inv(B)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("Current basis matrix is singular") from exc

        x_B = B_inv @ b
        c_B = c[basis]
        c_N = c[nonbasis]

        # lambda solves B^T lambda = c_B, i.e. lambda^T = c_B^T B^{-1}
        lam = np.linalg.solve(B.T, c_B)
        reduced_costs = c_N - N.T @ lam

        # Dual simplex requires reduced costs >= 0 for minimization.
        if np.min(reduced_costs) < -tol:
            raise RuntimeError(
                "Basis is not dual-feasible; reduced costs contain negative values"
            )

        min_basic = float(np.min(x_B))
        objective = float(c_B @ x_B)

        # If all basic variables are nonnegative, we have primal + dual feasibility.
        if min_basic >= -tol:
            x = np.zeros(n, dtype=float)
            x[basis] = np.maximum(x_B, 0.0)
            return DualSimplexResult(
                status="optimal",
                x=x,
                objective=float(c @ x),
                iterations=it,
                basis=basis.copy(),
                history=history,
            )

        leaving_row = int(np.argmin(x_B))
        row_coeffs = (B_inv @ N)[leaving_row, :]

        # Need row coefficient < 0 so that increasing entering variable can lift
        # the negative basic variable.
        candidates = np.where(row_coeffs < -tol)[0]
        if candidates.size == 0:
            return DualSimplexResult(
                status="infeasible",
                x=np.full(n, np.nan),
                objective=np.nan,
                iterations=it,
                basis=basis.copy(),
                history=history,
            )

        ratios = reduced_costs[candidates] / (-row_coeffs[candidates])
        pick = int(candidates[int(np.argmin(ratios))])

        entering_var = nonbasis[pick]
        leaving_var = basis[leaving_row]

        history.append(
            IterationLog(
                iter_idx=it,
                objective=objective,
                min_basic_value=min_basic,
                leaving_row=leaving_row,
                leaving_var=leaving_var,
                entering_var=entering_var,
                pivot_ratio=float(ratios[np.argmin(ratios)]),
            )
        )

        # Pivot by exchanging basis variable indices.
        basis[leaving_row] = entering_var

    return DualSimplexResult(
        status="max_iter_reached",
        x=np.full(n, np.nan),
        objective=np.nan,
        iterations=max_iter,
        basis=basis,
        history=history,
    )


def print_history(history: Sequence[IterationLog]) -> None:
    if not history:
        print("No pivot steps were needed.")
        return

    print("iter | obj      | min(x_B)  | leave(row,var) | enter(var) | ratio")
    print("-----+----------+-----------+----------------+------------+----------")
    for item in history:
        print(
            f"{item.iter_idx:4d} | {item.objective: .6f} | {item.min_basic_value: .6f}"
            f" | ({item.leaving_row:2d},{item.leaving_var:2d})"
            f"         | {item.entering_var:10d} | {item.pivot_ratio: .6f}"
        )


def main() -> None:
    # Example LP in equality form with slack variables x3,x4,x5:
    #   min z = x1 + 2 x2
    #   s.t. -x1 - x2 + x3       = -1
    #        -2x1 - x2      + x4 = -2
    #         x1 + x2            + x5 = 4
    #        x >= 0
    # Initial basis [x3, x4, x5] is dual-feasible (reduced costs >= 0)
    # but primal-infeasible (two basic values are negative).
    A = np.array(
        [
            [-1.0, -1.0, 1.0, 0.0, 0.0],
            [-2.0, -1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    b = np.array([-1.0, -2.0, 4.0], dtype=float)
    c = np.array([1.0, 2.0, 0.0, 0.0, 0.0], dtype=float)
    initial_basis = [2, 3, 4]

    result = dual_simplex(A=A, b=b, c=c, basis=initial_basis, tol=1e-9, max_iter=50)

    print("Dual simplex demo")
    print(f"Status: {result.status}")
    print(f"Iterations: {result.iterations}")
    print(f"Final basis: {result.basis}")

    if result.status != "optimal":
        raise RuntimeError(f"Dual simplex failed with status={result.status}")

    print(f"Optimal x: {result.x.tolist()}")
    print(f"Optimal objective: {result.objective:.8f}")

    print("\nPivot history:")
    print_history(result.history)

    # Deterministic self-check for this benchmark LP.
    expected_x = np.array([1.0, 0.0, 0.0, 0.0, 3.0], dtype=float)
    expected_obj = 1.0

    if not np.allclose(result.x, expected_x, atol=1e-8):
        raise RuntimeError(f"Unexpected solution: got {result.x}, expected {expected_x}")
    if not np.isclose(result.objective, expected_obj, atol=1e-10):
        raise RuntimeError(
            f"Unexpected objective: got {result.objective}, expected {expected_obj}"
        )
    if not np.allclose(A @ result.x, b, atol=1e-8):
        raise RuntimeError("Final solution does not satisfy Ax=b")
    if np.min(result.x) < -1e-8:
        raise RuntimeError("Final solution violates x>=0")

    print("\nValidation checks passed.")


if __name__ == "__main__":
    main()

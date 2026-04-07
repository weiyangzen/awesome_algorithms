"""Simplex method MVP for LP in form: maximize c^T x, s.t. Ax <= b, x >= 0."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

try:
    # Optional: only used as a validator/reference, not core algorithm.
    from scipy.optimize import linprog
except Exception:  # pragma: no cover - fallback path when scipy is unavailable
    linprog = None


@dataclass
class SimplexResult:
    status: str
    x_opt: np.ndarray
    objective: float
    iterations: int
    basis: List[int]
    tableau: np.ndarray
    objective_history: List[float]


def choose_entering_variable(objective_row: np.ndarray, tol: float) -> Optional[int]:
    """Pick entering column by Bland's rule among negative reduced costs."""
    candidates = np.where(objective_row[:-1] < -tol)[0]
    if candidates.size == 0:
        return None
    return int(candidates[0])


def choose_leaving_row(
    tableau: np.ndarray, entering_col: int, basis: List[int], tol: float
) -> Optional[int]:
    """Minimum ratio test with Bland tie-break on basis index."""
    m = tableau.shape[0] - 1
    rhs = tableau[:m, -1]
    col = tableau[:m, entering_col]

    ratios: List[Tuple[float, int, int]] = []
    for i in range(m):
        if col[i] > tol:
            ratios.append((rhs[i] / col[i], basis[i], i))

    if not ratios:
        return None
    ratios.sort(key=lambda x: (x[0], x[1]))
    return ratios[0][2]


def pivot_in_place(tableau: np.ndarray, pivot_row: int, pivot_col: int) -> None:
    """Perform Gauss-Jordan pivot on tableau[pivot_row, pivot_col]."""
    pivot = tableau[pivot_row, pivot_col]
    tableau[pivot_row, :] /= pivot

    for r in range(tableau.shape[0]):
        if r == pivot_row:
            continue
        factor = tableau[r, pivot_col]
        if factor != 0.0:
            tableau[r, :] -= factor * tableau[pivot_row, :]


def simplex_max_leq(
    c: np.ndarray,
    a_ub: np.ndarray,
    b_ub: np.ndarray,
    *,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> SimplexResult:
    """Solve max c^T x s.t. A x <= b, x >= 0 by tableau simplex."""
    c = np.asarray(c, dtype=float)
    a_ub = np.asarray(a_ub, dtype=float)
    b_ub = np.asarray(b_ub, dtype=float)

    if a_ub.ndim != 2:
        raise ValueError("a_ub must be a 2D array.")
    if c.ndim != 1 or b_ub.ndim != 1:
        raise ValueError("c and b_ub must be 1D arrays.")

    m, n = a_ub.shape
    if c.shape[0] != n:
        raise ValueError("Length of c must match number of columns in a_ub.")
    if b_ub.shape[0] != m:
        raise ValueError("Length of b_ub must match number of rows in a_ub.")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")
    if tol <= 0:
        raise ValueError("tol must be positive.")

    if not (np.isfinite(c).all() and np.isfinite(a_ub).all() and np.isfinite(b_ub).all()):
        raise ValueError("Inputs contain non-finite values.")
    if np.any(b_ub < -tol):
        raise ValueError("This MVP requires b_ub >= 0 for an initial feasible slack basis.")

    # Build initial tableau:
    # [ A | I | b ]
    # [ -c| 0 | 0 ]
    tableau = np.zeros((m + 1, n + m + 1), dtype=float)
    tableau[:m, :n] = a_ub
    tableau[:m, n : n + m] = np.eye(m)
    tableau[:m, -1] = b_ub
    tableau[m, :n] = -c

    basis = list(range(n, n + m))  # Slack variables as initial basic variables.
    objective_history: List[float] = []

    for iteration in range(max_iter):
        objective_history.append(float(tableau[m, -1]))
        entering = choose_entering_variable(tableau[m, :], tol)
        if entering is None:
            # Optimal reached.
            x_full = np.zeros(n + m, dtype=float)
            for r, var_idx in enumerate(basis):
                x_full[var_idx] = tableau[r, -1]
            x_opt = x_full[:n]
            return SimplexResult(
                status="optimal",
                x_opt=x_opt,
                objective=float(tableau[m, -1]),
                iterations=iteration,
                basis=basis.copy(),
                tableau=tableau.copy(),
                objective_history=objective_history,
            )

        leaving = choose_leaving_row(tableau, entering, basis, tol)
        if leaving is None:
            return SimplexResult(
                status="unbounded",
                x_opt=np.full(n, np.nan),
                objective=float("inf"),
                iterations=iteration,
                basis=basis.copy(),
                tableau=tableau.copy(),
                objective_history=objective_history,
            )

        pivot_in_place(tableau, leaving, entering)
        basis[leaving] = entering

    x_full = np.zeros(n + m, dtype=float)
    for r, var_idx in enumerate(basis):
        x_full[var_idx] = tableau[r, -1]
    return SimplexResult(
        status="max_iter_reached",
        x_opt=x_full[:n],
        objective=float(tableau[m, -1]),
        iterations=max_iter,
        basis=basis.copy(),
        tableau=tableau.copy(),
        objective_history=objective_history,
    )


def build_demo_lp() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # max 3x1 + 2x2
    # s.t. x1 + x2 <= 4
    #      x1 <= 2
    #      x2 <= 3
    #      x1, x2 >= 0
    c = np.array([3.0, 2.0], dtype=float)
    a_ub = np.array(
        [
            [1.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    b_ub = np.array([4.0, 2.0, 3.0], dtype=float)
    return c, a_ub, b_ub


def main() -> None:
    c, a_ub, b_ub = build_demo_lp()
    result = simplex_max_leq(c, a_ub, b_ub, max_iter=100, tol=1e-10)

    print("=== Simplex MVP (Tableau) ===")
    print(f"status: {result.status}")
    print(f"x*: {np.round(result.x_opt, 8)}")
    print(f"objective: {result.objective:.8f}")
    print(f"iterations: {result.iterations}")
    print(f"basis (column indices): {result.basis}")

    lhs = a_ub @ result.x_opt
    max_violation = float(np.max(lhs - b_ub))
    min_x = float(np.min(result.x_opt))
    primal_obj = float(c @ result.x_opt)
    print(f"max(Ax-b): {max_violation:.3e}")
    print(f"min(x): {min_x:.3e}")
    print(f"objective consistency |c^T x - tableau_obj|: {abs(primal_obj - result.objective):.3e}")

    if linprog is not None:
        ref = linprog(-c, A_ub=a_ub, b_ub=b_ub, bounds=[(0, None)] * len(c), method="highs")
        if ref.success:
            ref_obj = float(-ref.fun)
            print(f"scipy objective: {ref_obj:.8f}")
            print(f"|simplex - scipy|: {abs(result.objective - ref_obj):.3e}")
            assert abs(result.objective - ref_obj) < 1e-8, "Objective mismatch with scipy."
        else:
            print("scipy check skipped: linprog did not report success.")
    else:
        print("scipy check skipped: scipy is not available.")

    assert result.status == "optimal", "Simplex did not converge to optimal on demo instance."
    assert max_violation <= 1e-8, "Constraint violation detected."
    assert min_x >= -1e-8, "Negative variable detected."
    assert abs(primal_obj - result.objective) < 1e-8, "Objective mismatch within simplex outputs."

    print("All checks passed.")


if __name__ == "__main__":
    main()

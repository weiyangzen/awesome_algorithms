"""MVP: Integer Programming via Gomory fractional cutting planes.

This script implements a small, educational solver for pure integer linear programs:

    max c^T x
    s.t. A x <= b
         x >= 0, x in Z^n

Method:
1) Solve LP relaxation with primal simplex.
2) If LP solution is fractional, add one Gomory fractional cut from a valid tableau row.
3) Re-optimize with dual simplex.
4) Repeat until integral or cut limit reached.

The implementation favors clarity over performance and is intended for small dense toy problems.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class SolveResult:
    status: str
    objective: float
    x: np.ndarray
    cuts_added: int
    history: List[str]


class GomoryCuttingPlaneSolver:
    def __init__(self, A: np.ndarray, b: np.ndarray, c: np.ndarray, tol: float = 1e-9) -> None:
        self.A = np.asarray(A, dtype=float)
        self.b = np.asarray(b, dtype=float)
        self.c = np.asarray(c, dtype=float)
        self.tol = tol

        if self.A.ndim != 2:
            raise ValueError("A must be a 2D matrix")
        m, n = self.A.shape
        if self.b.shape != (m,):
            raise ValueError("b must have shape (m,)")
        if self.c.shape != (n,):
            raise ValueError("c must have shape (n,)")
        if np.any(~np.isfinite(self.A)) or np.any(~np.isfinite(self.b)) or np.any(~np.isfinite(self.c)):
            raise ValueError("A, b, c must be finite")
        if np.any(self.b < -self.tol):
            raise ValueError("This MVP assumes b >= 0 for initial simplex feasibility")

        self.m = m
        self.n = n

        # Variable layout in tableau columns:
        # [x_0..x_{n-1}, slack_0..slack_{m-1}, cut_slack_0, cut_slack_1, ...]
        self.num_vars = n + m
        self.integer_var_indices = set(range(n + m))  # original variables + original slacks

        # Build initial tableau for Ax + s = b, objective max c^T x.
        self.tableau = self._build_initial_tableau()
        self.basis = list(range(n, n + m))

    def _build_initial_tableau(self) -> np.ndarray:
        m, n = self.m, self.n
        # m constraint rows + 1 objective row; num_vars + 1 RHS column
        T = np.zeros((m + 1, n + m + 1), dtype=float)
        T[:m, :n] = self.A
        T[:m, n : n + m] = np.eye(m)
        T[:m, -1] = self.b

        # Objective row: negative coefficients for primal simplex maximization.
        T[m, :n] = -self.c
        return T

    @staticmethod
    def _fractional_part(x: float) -> float:
        return x - np.floor(x)

    def _is_integer(self, x: float) -> bool:
        return abs(x - round(x)) <= self.tol

    def _pivot(self, row: int, col: int) -> None:
        T = self.tableau
        piv = T[row, col]
        if abs(piv) <= self.tol:
            raise RuntimeError("Pivot element is numerically zero")

        T[row, :] = T[row, :] / piv
        for r in range(T.shape[0]):
            if r == row:
                continue
            factor = T[r, col]
            if abs(factor) > self.tol:
                T[r, :] -= factor * T[row, :]

        self.basis[row] = col

    def _current_solution(self) -> np.ndarray:
        x = np.zeros(self.num_vars, dtype=float)
        for i, bidx in enumerate(self.basis):
            x[bidx] = self.tableau[i, -1]
        # Clip tiny negative noise.
        x[np.abs(x) <= 10 * self.tol] = 0.0
        return x

    def primal_simplex(self, max_iter: int = 10_000) -> None:
        """Optimize LP for current constraints from a primal-feasible BFS."""
        T = self.tableau
        m = T.shape[0] - 1

        for _ in range(max_iter):
            obj = T[m, :-1]
            entering = int(np.argmin(obj))
            if obj[entering] >= -self.tol:
                return  # optimal

            col = T[:m, entering]
            rhs = T[:m, -1]
            candidates: List[Tuple[float, int]] = []
            for i in range(m):
                if col[i] > self.tol:
                    candidates.append((rhs[i] / col[i], i))

            if not candidates:
                raise RuntimeError("LP relaxation is unbounded in primal simplex")

            _, leaving_row = min(candidates, key=lambda x: (x[0], x[1]))
            self._pivot(leaving_row, entering)

        raise RuntimeError("Primal simplex exceeded iteration limit")

    def dual_simplex(self, max_iter: int = 10_000) -> None:
        """Restore primal feasibility while preserving dual feasibility."""
        T = self.tableau
        m = T.shape[0] - 1

        for _ in range(max_iter):
            rhs = T[:m, -1]
            leaving_row = int(np.argmin(rhs))
            if rhs[leaving_row] >= -self.tol:
                return  # feasible -> optimal (dual feasible maintained)

            row_coeff = T[leaving_row, :-1]
            obj = T[m, :-1]
            candidates: List[Tuple[float, int]] = []
            for j in range(self.num_vars):
                a_rj = row_coeff[j]
                if a_rj < -self.tol:
                    # Keep reduced costs nonnegative in this convention.
                    ratio = obj[j] / (-a_rj)
                    candidates.append((ratio, j))

            if not candidates:
                raise RuntimeError("Dual simplex failed: no entering variable, problem infeasible")

            _, entering = min(candidates, key=lambda x: (x[0], x[1]))
            self._pivot(leaving_row, entering)

        raise RuntimeError("Dual simplex exceeded iteration limit")

    def select_gomory_row(self) -> Optional[int]:
        """Pick a row whose basic integer variable has fractional RHS."""
        best_row = None
        best_frac = 0.0

        for i, basic_idx in enumerate(self.basis):
            if basic_idx not in self.integer_var_indices:
                continue

            rhs = self.tableau[i, -1]
            frac = self._fractional_part(rhs)
            frac = min(frac, 1.0 - frac) if frac > 0.5 else frac
            if frac <= self.tol:
                continue

            if frac > best_frac + self.tol:
                best_frac = frac
                best_row = i

        return best_row

    def add_gomory_fractional_cut(self, row: int) -> None:
        """Add one GMI-style Gomory cut and keep basis with the new cut slack."""
        old_rows, old_cols_plus_rhs = self.tableau.shape
        old_cols = old_cols_plus_rhs - 1

        # Expand tableau: +1 variable column, +1 constraint row.
        new_T = np.zeros((old_rows + 1, old_cols + 2), dtype=float)

        # Copy old constraint rows and objective row.
        new_T[: old_rows - 1, :old_cols] = self.tableau[: old_rows - 1, :old_cols]
        new_T[: old_rows - 1, -1] = self.tableau[: old_rows - 1, -1]
        new_T[old_rows - 1, :old_cols] = self.tableau[old_rows - 1, :old_cols]
        new_T[old_rows - 1, -1] = self.tableau[old_rows - 1, -1]

        nonbasic = set(range(self.num_vars)) - set(self.basis)
        rhs = self.tableau[row, -1]
        f0 = self._fractional_part(rhs)
        if f0 <= self.tol or (1.0 - f0) <= self.tol:
            raise RuntimeError("Selected row has near-integer RHS; cannot build Gomory cut")

        # GMI cut in normalized form:
        #   sum_j alpha_j * x_j >= 1
        # where alpha_j depends on integer/continuous type and row coefficient.
        # Convert to tableau row with new slack g >= 0:
        #   g + sum_j (-alpha_j) * x_j = -1
        cut_row = old_rows - 1
        for j in nonbasic:
            coeff = self.tableau[row, j]
            if abs(coeff) <= self.tol:
                continue

            if j in self.integer_var_indices:
                fj = self._fractional_part(coeff)
                if fj <= f0 + self.tol:
                    alpha = fj / f0
                else:
                    alpha = (1.0 - fj) / (1.0 - f0)
            else:
                if coeff >= 0.0:
                    alpha = coeff / f0
                else:
                    alpha = -coeff / (1.0 - f0)

            if alpha > self.tol:
                new_T[cut_row, j] = -alpha

        new_slack_idx = old_cols
        new_T[cut_row, new_slack_idx] = 1.0
        new_T[cut_row, -1] = -1.0

        # New objective row already zeros in new column and row except copied row.
        self.tableau = new_T

        # Basis: existing rows unchanged + new cut slack variable as basic in cut row.
        self.basis.append(new_slack_idx)
        self.num_vars += 1

        # Cut slack is continuous; do NOT add to integer_var_indices.

    def solve(self, max_cuts: int = 10) -> SolveResult:
        history: List[str] = []

        # 1) LP relaxation.
        self.primal_simplex()

        for cut_count in range(max_cuts + 1):
            sol = self._current_solution()
            x = sol[: self.n]
            obj = float(self.c @ x)
            history.append(
                f"iter={cut_count:02d}, cuts={cut_count}, x={np.array2string(x, precision=6)}, obj={obj:.6f}"
            )

            if all(self._is_integer(v) for v in x):
                return SolveResult(
                    status="optimal_integer",
                    objective=obj,
                    x=np.round(x).astype(float),
                    cuts_added=cut_count,
                    history=history,
                )

            if cut_count == max_cuts:
                return SolveResult(
                    status="cut_limit_reached",
                    objective=obj,
                    x=x,
                    cuts_added=cut_count,
                    history=history,
                )

            row = self.select_gomory_row()
            if row is None:
                return SolveResult(
                    status="no_valid_cut_row",
                    objective=obj,
                    x=x,
                    cuts_added=cut_count,
                    history=history,
                )

            self.add_gomory_fractional_cut(row)
            self.dual_simplex()

        # Defensive fallback; logically unreachable.
        return SolveResult(
            status="internal_error",
            objective=float("nan"),
            x=np.zeros(self.n),
            cuts_added=max_cuts,
            history=history,
        )


def brute_force_reference(
    A: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    upper: Sequence[int],
) -> Tuple[np.ndarray, float]:
    """Tiny brute-force verifier for demonstration only."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    upper = list(upper)

    if len(upper) != A.shape[1]:
        raise ValueError("upper bounds length mismatch")

    best_x: Optional[np.ndarray] = None
    best_obj = -float("inf")

    grids = [range(u + 1) for u in upper]
    for x0 in grids[0]:
        for x1 in grids[1]:
            x = np.array([x0, x1], dtype=float)
            if np.all(A @ x <= b + 1e-9):
                obj = float(c @ x)
                if obj > best_obj + 1e-12:
                    best_obj = obj
                    best_x = x.copy()

    if best_x is None:
        raise RuntimeError("No feasible integer point in brute-force box")

    return best_x, best_obj


def main() -> None:
    # A classic toy ILP whose LP relaxation optimum is fractional.
    # max x1 + x2
    # s.t. 2x1 + x2 <= 4
    #      x1 + 2x2 <= 4
    #      x1, x2 >= 0 integer
    A = np.array([[2, 1], [1, 2]], dtype=float)
    b = np.array([4, 4], dtype=float)
    c = np.array([1, 1], dtype=float)

    solver = GomoryCuttingPlaneSolver(A, b, c)
    result = solver.solve(max_cuts=6)

    print("=== Gomory Cutting Plane MVP ===")
    print(f"status: {result.status}")
    print(f"cuts_added: {result.cuts_added}")
    print(f"x: {np.array2string(result.x, precision=6)}")
    print(f"objective: {result.objective:.6f}")
    print("progress:")
    for line in result.history:
        print("  " + line)

    # Independent tiny verifier (bounded box from constraints for this toy case).
    bf_x, bf_obj = brute_force_reference(A, b, c, upper=[4, 4])
    print("bruteforce_best_x:", bf_x)
    print(f"bruteforce_best_obj: {bf_obj:.6f}")

    # Minimal assertions for reproducible validation.
    if result.status != "optimal_integer":
        raise RuntimeError(f"Unexpected solver status: {result.status}")
    if not np.allclose(result.objective, bf_obj, atol=1e-7):
        raise RuntimeError("Objective does not match brute-force reference")
    if not np.all(result.x >= -1e-9):
        raise RuntimeError("Returned solution violates non-negativity")
    if not np.all(A @ result.x <= b + 1e-9):
        raise RuntimeError("Returned solution violates Ax<=b")

    print("All checks passed.")


if __name__ == "__main__":
    main()

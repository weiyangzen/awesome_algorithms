"""Column Generation MVP on a 1D cutting-stock LP relaxation.

Run:
    python3 demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

try:
    from scipy.optimize import linprog
except ModuleNotFoundError:  # pragma: no cover - depends on runtime environment
    linprog = None


@dataclass(frozen=True)
class CuttingStockInstance:
    """Data container for a 1D cutting stock instance."""

    stock_length: int
    item_lengths: np.ndarray
    demands: np.ndarray

    def validate(self) -> None:
        if self.stock_length <= 0:
            raise ValueError("stock_length must be positive")
        if self.item_lengths.ndim != 1 or self.demands.ndim != 1:
            raise ValueError("item_lengths and demands must be 1D arrays")
        if len(self.item_lengths) != len(self.demands):
            raise ValueError("item_lengths and demands must have the same length")
        if np.any(self.item_lengths <= 0):
            raise ValueError("all item lengths must be positive")
        if np.any(self.demands < 0):
            raise ValueError("demands must be nonnegative")
        if np.any(self.item_lengths > self.stock_length):
            raise ValueError("some item lengths exceed stock length; instance infeasible")


def build_initial_patterns(instance: CuttingStockInstance) -> List[np.ndarray]:
    """Create one single-item-max pattern per item type."""
    patterns: List[np.ndarray] = []
    m = len(instance.item_lengths)
    for i in range(m):
        cnt = instance.stock_length // int(instance.item_lengths[i])
        if cnt <= 0:
            raise ValueError(f"cannot create initial pattern for item index {i}")
        p = np.zeros(m, dtype=int)
        p[i] = cnt
        patterns.append(p)
    return patterns


def _extract_duals_from_highs(result: object) -> np.ndarray:
    """Get dual prices for original constraints A x >= d from HiGHS output."""
    ineqlin = getattr(result, "ineqlin", None)
    if ineqlin is None or not hasattr(ineqlin, "marginals"):
        raise RuntimeError(
            "SciPy/HiGHS result does not contain ineqlin.marginals; "
            "please use a SciPy build with HiGHS duals available."
        )
    # We solve -A x <= -d, so convert signs back to the original Ax >= d form.
    return -np.asarray(ineqlin.marginals, dtype=float)


def _solve_dual_with_revised_simplex(
    a_matrix: np.ndarray, demands: np.ndarray, tol: float = 1e-10, max_iter: int = 500
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Fallback LP solver when SciPy is unavailable.

    We solve the dual of RMP:
        max d^T y
        s.t. A^T y <= 1, y >= 0
    via revised simplex on the standard form with slacks.

    Returns:
        dual_prices y (length m),
        objective value,
        primal x (length k) recovered as simplex dual multipliers.
    """
    g = a_matrix.T.astype(float)  # shape: (k, m)
    c = demands.astype(float)  # length m
    k, m = g.shape

    # Standard form: G y + s = 1, (y, s) >= 0
    a_std = np.hstack([g, np.eye(k, dtype=float)])  # shape: (k, m+k)
    b_std = np.ones(k, dtype=float)
    c_std = np.concatenate([c, np.zeros(k, dtype=float)])
    n_total = m + k

    basis = list(range(m, n_total))  # initial basis: slacks

    for _ in range(max_iter):
        b_mat = a_std[:, basis]
        try:
            b_inv = np.linalg.inv(b_mat)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("Simplex basis became singular in fallback solver") from exc

        x_b = b_inv @ b_std
        c_b = c_std[basis]

        # u are dual multipliers for equality constraints, which map to primal x.
        u = c_b @ b_inv
        reduced = c_std - u @ a_std

        basis_set = set(basis)
        entering = None
        for j in range(n_total):  # Bland rule
            if j in basis_set:
                continue
            if reduced[j] > tol:
                entering = j
                break

        if entering is None:
            z = np.zeros(n_total, dtype=float)
            z[basis] = x_b
            y = z[:m]
            x = np.where(u > tol, u, 0.0)  # primal pattern usage
            obj = float(c @ y)
            return y, obj, x

        direction = b_inv @ a_std[:, entering]
        ratios = np.full(k, np.inf, dtype=float)
        for i in range(k):
            if direction[i] > tol:
                ratios[i] = x_b[i] / direction[i]

        theta = float(np.min(ratios))
        if not np.isfinite(theta):
            raise RuntimeError("Fallback simplex detected an unbounded dual LP")

        leaving_candidates = [i for i in range(k) if abs(ratios[i] - theta) <= 1e-12]
        leaving_row = min(leaving_candidates, key=lambda i: basis[i])  # Bland tie-break
        basis[leaving_row] = entering

    raise RuntimeError("Fallback simplex reached max iterations without convergence")


def solve_restricted_master(
    instance: CuttingStockInstance, patterns: Sequence[np.ndarray]
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Solve RMP LP and return (x, objective, dual_prices)."""
    if not patterns:
        raise ValueError("patterns cannot be empty")

    a_matrix = np.column_stack(patterns).astype(float)  # shape: (m, k)
    c = np.ones(a_matrix.shape[1], dtype=float)
    a_ub = -a_matrix
    b_ub = -instance.demands.astype(float)
    bounds = [(0.0, None)] * a_matrix.shape[1]

    if linprog is not None:
        res = linprog(c, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if not res.success:
            raise RuntimeError(f"RMP solve failed: {res.message}")
        dual_prices = _extract_duals_from_highs(res)
        return np.asarray(res.x, dtype=float), float(res.fun), dual_prices

    dual_prices, obj, x_primal = _solve_dual_with_revised_simplex(a_matrix, instance.demands)
    if np.any(a_matrix @ x_primal < instance.demands - 1e-7):
        raise RuntimeError("Fallback simplex produced an infeasible primal reconstruction")
    return x_primal, obj, dual_prices


def solve_pricing_subproblem(
    instance: CuttingStockInstance, dual_prices: np.ndarray
) -> Tuple[np.ndarray, float, float]:
    """Unbounded knapsack pricing: maximize dual value under stock length.

    Returns:
        pattern: best generated column (integer counts)
        reduced_cost: 1 - dual_value
        dual_value: max dual objective value of the pricing subproblem
    """
    stock = int(instance.stock_length)
    lengths = instance.item_lengths.astype(int)
    m = len(lengths)

    best_value = np.zeros(stock + 1, dtype=float)
    choice = np.full(stock + 1, -1, dtype=int)
    prev_cap = np.full(stock + 1, -1, dtype=int)

    for cap in range(1, stock + 1):
        value_here = best_value[cap]
        item_here = -1
        prev_here = -1
        for i in range(m):
            li = int(lengths[i])
            if li <= cap:
                candidate = best_value[cap - li] + float(dual_prices[i])
                if candidate > value_here + 1e-12:
                    value_here = candidate
                    item_here = i
                    prev_here = cap - li
        best_value[cap] = value_here
        choice[cap] = item_here
        prev_cap[cap] = prev_here

    best_cap = int(np.argmax(best_value))
    dual_obj = float(best_value[best_cap])
    reduced_cost = 1.0 - dual_obj

    pattern = np.zeros(m, dtype=int)
    cap = best_cap
    while cap > 0 and choice[cap] != -1:
        idx = int(choice[cap])
        pattern[idx] += 1
        cap = int(prev_cap[cap])

    return pattern, reduced_cost, dual_obj


def column_generation(
    instance: CuttingStockInstance, max_iter: int = 30, tol: float = 1e-8
) -> Tuple[List[np.ndarray], np.ndarray, float, List[Tuple[int, float, float, np.ndarray]]]:
    """Run column generation and return final data."""
    patterns = build_initial_patterns(instance)
    logs: List[Tuple[int, float, float, np.ndarray]] = []

    for it in range(1, max_iter + 1):
        x, obj, duals = solve_restricted_master(instance, patterns)
        new_pattern, reduced_cost, _ = solve_pricing_subproblem(instance, duals)
        logs.append((it, obj, reduced_cost, new_pattern.copy()))

        if reduced_cost >= -tol:
            return patterns, x, obj, logs

        if not new_pattern.any():
            # No useful pattern reconstructed; stop defensively.
            return patterns, x, obj, logs

        if any(np.array_equal(new_pattern, p) for p in patterns):
            # Avoid cycling on duplicate columns under degeneracy.
            return patterns, x, obj, logs

        patterns.append(new_pattern)

    # Max iterations reached: return last solved state.
    x, obj, _ = solve_restricted_master(instance, patterns)
    return patterns, x, obj, logs


def summarize_solution(
    instance: CuttingStockInstance, patterns: Sequence[np.ndarray], x_lp: np.ndarray
) -> None:
    """Print LP and a simple integer-feasible rounded solution."""
    a_matrix = np.column_stack(patterns).astype(int)

    x_int = np.where(x_lp > 1e-9, np.ceil(x_lp), 0.0).astype(int)
    coverage_int = a_matrix @ x_int
    feasible_int = bool(np.all(coverage_int >= instance.demands))

    print("\nFinal patterns and usages:")
    for j, p in enumerate(patterns):
        print(f"  pattern {j:02d}: {p.tolist()} | x_lp={x_lp[j]:.4f} | x_int={x_int[j]}")

    print(f"\nLP objective (lower bound): {np.sum(x_lp):.4f}")
    print(f"Rounded integer usage total: {int(np.sum(x_int))}")
    print(f"Rounded solution feasible: {feasible_int}")

    print("\nDemand coverage check (rounded solution):")
    for i, (need, got) in enumerate(zip(instance.demands.tolist(), coverage_int.tolist())):
        print(f"  item {i:02d}: demand={need}, covered={got}")


def main() -> None:
    # A small nontrivial instance where column generation adds mixed patterns.
    instance = CuttingStockInstance(
        stock_length=11,
        item_lengths=np.array([2, 3, 5, 6], dtype=int),
        demands=np.array([20, 18, 12, 8], dtype=int),
    )
    instance.validate()

    print("Column Generation MVP - 1D Cutting Stock")
    print(f"stock_length = {instance.stock_length}")
    print(f"item_lengths = {instance.item_lengths.tolist()}")
    print(f"demands      = {instance.demands.tolist()}")

    patterns, x_lp, lp_obj, logs = column_generation(instance, max_iter=50, tol=1e-8)

    print("\nIteration log:")
    for it, obj, rc, pattern in logs:
        print(
            f"  iter={it:02d} | RMP_obj={obj:.6f} | reduced_cost={rc:.6f} "
            f"| new_pattern={pattern.tolist()}"
        )

    print(f"\nTotal generated patterns: {len(patterns)}")
    print(f"Final LP objective: {lp_obj:.6f}")

    summarize_solution(instance, patterns, x_lp)


if __name__ == "__main__":
    main()

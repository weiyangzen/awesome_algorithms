"""Branch-and-Cut MVP for 0-1 Knapsack.

This demo intentionally keeps a small and transparent stack:
- numpy for vectorized numeric operations
- a tiny in-file simplex routine for LP relaxations (no external LP black box)

Algorithm sketch:
1) Solve LP relaxation at each B&B node.
2) Try separating violated knapsack cover cuts.
3) If fractional remains, branch on one fractional variable.
4) Track best incumbent and prune by bound/infeasibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


EPS = 1e-9


@dataclass
class BranchCutResult:
    best_value: float
    best_solution: np.ndarray
    explored_nodes: int
    branch_count: int
    cut_count: int
    cuts: List[Tuple[int, ...]]
    logs: List[str]


def simplex_maximize(
    A: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    tol: float = EPS,
    max_iter: int = 10_000,
) -> Optional[Tuple[float, np.ndarray]]:
    """Solve max c^T x s.t. A x <= b, x >= 0 via primal simplex tableau.

    Notes:
    - Assumes an obvious initial BFS from slack variables (thus needs b >= 0).
    - Uses Bland-style entering rule (first negative reduced cost) to reduce cycling.
    - Returns None for infeasible/unbounded/iteration-limit cases.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)

    m, n = A.shape

    if np.any(b < -tol):
        return None

    keep_rows: List[int] = []
    for i in range(m):
        if np.linalg.norm(A[i], ord=1) <= tol:
            if b[i] < -tol:
                return None
        else:
            keep_rows.append(i)

    if len(keep_rows) != m:
        A = A[keep_rows, :]
        b = b[keep_rows]
        m = A.shape[0]

    if m == 0:
        if np.any(c > tol):
            return None
        return 0.0, np.zeros(n, dtype=float)

    tableau = np.zeros((m + 1, n + m + 1), dtype=float)
    tableau[1:, :n] = A
    tableau[1:, n : n + m] = np.eye(m)
    tableau[1:, -1] = b
    tableau[0, :n] = -c

    basis = list(range(n, n + m))

    def pivot(row: int, col: int) -> None:
        piv = tableau[row, col]
        tableau[row, :] /= piv
        for rr in range(m + 1):
            if rr == row:
                continue
            factor = tableau[rr, col]
            if abs(factor) > tol:
                tableau[rr, :] -= factor * tableau[row, :]

    it = 0
    while it < max_iter:
        it += 1

        entering: Optional[int] = None
        for j in range(n + m):
            if tableau[0, j] < -tol:
                entering = j
                break

        if entering is None:
            break

        ratios: List[Tuple[float, int, int]] = []
        column = tableau[1:, entering]
        for i, val in enumerate(column, start=1):
            if val > tol:
                ratios.append((tableau[i, -1] / val, basis[i - 1], i))

        if not ratios:
            return None

        ratios.sort(key=lambda x: (x[0], x[1]))
        leaving_row = ratios[0][2]
        basis[leaving_row - 1] = entering
        pivot(leaving_row, entering)

    if it >= max_iter:
        return None

    sol_full = np.zeros(n + m, dtype=float)
    for row_idx, basic_var in enumerate(basis, start=1):
        sol_full[basic_var] = tableau[row_idx, -1]

    x = sol_full[:n]
    x[np.abs(x) < tol] = 0.0
    obj = float(tableau[0, -1])
    return obj, x


def solve_node_lp_relaxation(
    profits: np.ndarray,
    weights: np.ndarray,
    capacity: float,
    fixed: Dict[int, int],
    cuts: Sequence[Tuple[int, ...]],
) -> Optional[Tuple[float, np.ndarray]]:
    """Solve one B&B node LP relaxation with globally collected cuts."""
    n = profits.size

    base_x = np.zeros(n, dtype=float)
    for idx, val in fixed.items():
        if val not in (0, 1):
            return None
        base_x[idx] = float(val)

    residual_capacity = capacity - float(np.dot(weights, base_x))
    if residual_capacity < -EPS:
        return None

    free_idx = np.array([i for i in range(n) if i not in fixed], dtype=int)
    if free_idx.size == 0:
        return float(np.dot(profits, base_x)), base_x

    A_rows: List[np.ndarray] = [weights[free_idx].astype(float)]
    b_rows: List[float] = [float(residual_capacity)]

    fixed_one_set = {i for i, v in fixed.items() if v == 1}

    for cover in cuts:
        row = np.array([1.0 if j in cover else 0.0 for j in free_idx], dtype=float)
        rhs = float(len(cover) - 1 - len(fixed_one_set.intersection(cover)))
        if rhs < -EPS:
            return None
        A_rows.append(row)
        b_rows.append(rhs)

    # 0 <= x_j <= 1 on free vars -> simplex takes x >= 0 by default, add x <= 1.
    for j in range(free_idx.size):
        row = np.zeros(free_idx.size, dtype=float)
        row[j] = 1.0
        A_rows.append(row)
        b_rows.append(1.0)

    A = np.vstack(A_rows)
    b = np.array(b_rows, dtype=float)
    c = profits[free_idx].astype(float)

    simplex_out = simplex_maximize(A=A, b=b, c=c)
    if simplex_out is None:
        return None

    _, x_free = simplex_out
    x_full = base_x.copy()
    x_full[free_idx] = x_free

    obj_val = float(np.dot(profits, x_full))
    return obj_val, x_full


def _minimal_cover_from_order(
    order: np.ndarray,
    x: np.ndarray,
    weights: np.ndarray,
    capacity: float,
    eps: float,
) -> Optional[List[int]]:
    """Build a minimal cover C with sum(w_i) > capacity from an index order."""
    cover: List[int] = []
    total_w = 0.0

    for raw_idx in order:
        i = int(raw_idx)
        if x[i] <= eps:
            continue
        cover.append(i)
        total_w += float(weights[i])
        if total_w > capacity + eps:
            break

    if total_w <= capacity + eps or len(cover) < 2:
        return None

    changed = True
    while changed and len(cover) > 1:
        changed = False
        for i in cover.copy():
            if total_w - float(weights[i]) > capacity + eps:
                cover.remove(i)
                total_w -= float(weights[i])
                changed = True

    return cover


def find_violated_cover_cut(
    x: np.ndarray,
    weights: np.ndarray,
    capacity: float,
    eps: float = EPS,
) -> Tuple[Optional[Tuple[int, ...]], float]:
    """Heuristic separation for knapsack cover cuts.

    Looks for a cover C such that:
    - sum_{i in C} w_i > capacity
    - current LP point violates sum_{i in C} x_i <= |C| - 1
    """
    orders = [
        np.argsort(-x),
        np.argsort(-(x * weights)),
        np.argsort(-weights),
    ]

    for order in orders:
        cover = _minimal_cover_from_order(order, x, weights, capacity, eps)
        if cover is None:
            continue

        lhs = float(np.sum(x[cover]))
        rhs = float(len(cover) - 1)
        violation = lhs - rhs

        if violation > 1e-8:
            return tuple(sorted(cover)), violation

    return None, 0.0


def branch_and_cut_knapsack(
    profits: np.ndarray,
    weights: np.ndarray,
    capacity: float,
    max_nodes: int = 1_000,
    max_cuts: int = 8,
    max_cut_rounds_per_node: int = 3,
    int_tol: float = 1e-8,
) -> BranchCutResult:
    """Depth-first branch-and-cut for a small 0-1 knapsack instance."""
    n = profits.size

    stack: List[Dict[int, int]] = [dict()]
    best_value = -np.inf
    best_solution = np.zeros(n, dtype=float)

    global_cuts: List[Tuple[int, ...]] = []
    cut_set = set()

    explored_nodes = 0
    branch_count = 0
    logs: List[str] = []

    while stack and explored_nodes < max_nodes:
        fixed = stack.pop()
        explored_nodes += 1

        lp = solve_node_lp_relaxation(
            profits=profits,
            weights=weights,
            capacity=capacity,
            fixed=fixed,
            cuts=global_cuts,
        )

        if lp is None:
            continue

        ub, x_lp = lp

        if ub <= best_value + int_tol:
            continue

        pruned = False
        for _ in range(max_cut_rounds_per_node):
            cut, violation = find_violated_cover_cut(
                x=x_lp,
                weights=weights,
                capacity=capacity,
            )
            if cut is None or cut in cut_set or len(cut_set) >= max_cuts:
                break

            cut_set.add(cut)
            global_cuts.append(cut)
            logs.append(
                f"node={explored_nodes}: add cut {cut}, violation={violation:.4f}"
            )

            lp = solve_node_lp_relaxation(
                profits=profits,
                weights=weights,
                capacity=capacity,
                fixed=fixed,
                cuts=global_cuts,
            )

            if lp is None:
                pruned = True
                break

            ub, x_lp = lp
            if ub <= best_value + int_tol:
                pruned = True
                break

        if pruned:
            continue

        frac = np.abs(x_lp - np.round(x_lp))
        if float(np.max(frac)) <= int_tol:
            x_int = np.round(x_lp)
            val = float(np.dot(profits, x_int))
            if val > best_value + int_tol:
                best_value = val
                best_solution = x_int
                logs.append(
                    f"node={explored_nodes}: incumbent update value={best_value:.4f}"
                )
            continue

        frac_idx = np.where(frac > int_tol)[0]
        branch_var = int(frac_idx[np.argmin(np.abs(x_lp[frac_idx] - 0.5))])

        left = dict(fixed)
        right = dict(fixed)
        left[branch_var] = 0
        right[branch_var] = 1

        # DFS: push 0-branch first so 1-branch is explored next.
        stack.append(left)
        stack.append(right)
        branch_count += 1

    return BranchCutResult(
        best_value=float(best_value),
        best_solution=best_solution,
        explored_nodes=explored_nodes,
        branch_count=branch_count,
        cut_count=len(global_cuts),
        cuts=global_cuts,
        logs=logs,
    )


def brute_force_knapsack(
    profits: np.ndarray,
    weights: np.ndarray,
    capacity: float,
) -> Tuple[float, np.ndarray]:
    """Small-size oracle used only for MVP sanity check."""
    n = profits.size
    best_v = -np.inf
    best_x = np.zeros(n, dtype=float)

    for mask in range(1 << n):
        x = np.array([(mask >> i) & 1 for i in range(n)], dtype=float)
        if float(np.dot(weights, x)) <= capacity + EPS:
            v = float(np.dot(profits, x))
            if v > best_v + EPS:
                best_v = v
                best_x = x

    return best_v, best_x


def main() -> None:
    # Built-in deterministic instance.
    profits = np.array([20, 18, 14, 12, 10, 8, 7, 6], dtype=float)
    weights = np.array([11, 10, 8, 7, 6, 5, 4, 3], dtype=float)
    capacity = 23.0

    result = branch_and_cut_knapsack(
        profits=profits,
        weights=weights,
        capacity=capacity,
        max_nodes=1_000,
        max_cuts=8,
        max_cut_rounds_per_node=3,
    )

    brute_v, brute_x = brute_force_knapsack(
        profits=profits,
        weights=weights,
        capacity=capacity,
    )

    picked = np.where(result.best_solution > 0.5)[0].tolist()
    total_weight = float(np.dot(weights, result.best_solution))

    print("=== Branch-and-Cut Demo: 0-1 Knapsack ===")
    print(f"profits  = {profits.tolist()}")
    print(f"weights  = {weights.tolist()}")
    print(f"capacity = {capacity}")
    print()
    print("--- solver summary ---")
    print(f"best_value      = {result.best_value:.4f}")
    print(f"best_solution   = {result.best_solution.astype(int).tolist()}")
    print(f"picked_items    = {picked}")
    print(f"total_weight    = {total_weight:.4f}")
    print(f"explored_nodes  = {result.explored_nodes}")
    print(f"branch_count    = {result.branch_count}")
    print(f"generated_cuts  = {result.cut_count}")
    print()

    print("--- first logs (up to 12 lines) ---")
    for line in result.logs[:12]:
        print(line)
    if len(result.logs) > 12:
        print(f"... ({len(result.logs) - 12} more lines omitted)")
    print()

    print("--- brute-force cross-check ---")
    print(f"oracle_value    = {brute_v:.4f}")
    print(f"oracle_solution = {brute_x.astype(int).tolist()}")

    assert abs(result.best_value - brute_v) <= 1e-8, "best value mismatch vs oracle"
    assert total_weight <= capacity + 1e-8, "capacity violated"
    assert set(np.unique(result.best_solution)).issubset({0.0, 1.0}), "non-binary solution"

    print("checks: PASS")


if __name__ == "__main__":
    main()

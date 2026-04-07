"""Network simplex MVP (transportation special case of min-cost flow).

This script implements a minimal, auditable transportation-simplex solver,
which is a specialized form of the network simplex method on bipartite networks.
No interactive input is required.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

try:  # Optional: only for verification, not for the core algorithm.
    from scipy.optimize import linprog
except Exception:  # pragma: no cover - handled at runtime.
    linprog = None

Cell = Tuple[int, int]
HistoryItem = Tuple[int, float, float, Optional[Cell], Optional[Cell], float]
EPS = 1e-12


class DisjointSet:
    """Simple Union-Find for basis acyclicity checks."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1
        return True


def validate_inputs(cost: np.ndarray, supply: np.ndarray, demand: np.ndarray) -> None:
    if cost.ndim != 2:
        raise ValueError(f"cost must be 2D, got shape={cost.shape}")
    if supply.ndim != 1 or demand.ndim != 1:
        raise ValueError("supply and demand must be 1D vectors")
    if cost.shape != (supply.shape[0], demand.shape[0]):
        raise ValueError(
            "shape mismatch: cost shape must be (len(supply), len(demand))"
        )
    if not (np.all(np.isfinite(cost)) and np.all(np.isfinite(supply)) and np.all(np.isfinite(demand))):
        raise ValueError("cost/supply/demand must be finite")
    if np.any(supply < -EPS) or np.any(demand < -EPS):
        raise ValueError("supply and demand must be non-negative")

    total_supply = float(np.sum(supply))
    total_demand = float(np.sum(demand))
    if abs(total_supply - total_demand) > 1e-9:
        raise ValueError(
            f"unbalanced transportation problem: supply={total_supply}, demand={total_demand}"
        )


def node_id_row(i: int) -> int:
    return i


def node_id_col(j: int, m: int) -> int:
    return m + j


def build_initial_solution_northwest(
    supply: np.ndarray,
    demand: np.ndarray,
) -> Tuple[np.ndarray, Set[Cell]]:
    """Northwest-corner initialization + zero-cell augmentation to a tree basis."""
    m = supply.shape[0]
    n = demand.shape[0]
    x = np.zeros((m, n), dtype=float)
    basis: Set[Cell] = set()

    rem_supply = supply.astype(float).copy()
    rem_demand = demand.astype(float).copy()

    i = 0
    j = 0
    while i < m and j < n:
        amount = min(rem_supply[i], rem_demand[j])
        x[i, j] = amount
        basis.add((i, j))

        rem_supply[i] -= amount
        rem_demand[j] -= amount

        supply_done = rem_supply[i] <= EPS
        demand_done = rem_demand[j] <= EPS

        if supply_done and demand_done:
            i += 1
            j += 1
        elif supply_done:
            i += 1
        else:
            j += 1

    augment_basis_with_zero_cells(basis=basis, m=m, n=n)
    return x, basis


def augment_basis_with_zero_cells(basis: Set[Cell], m: int, n: int) -> None:
    """Ensure exactly m+n-1 basic cells and no cycle (spanning tree basis)."""
    needed = m + n - 1
    if len(basis) > needed:
        raise RuntimeError("initial basis has too many cells")

    dsu = DisjointSet(m + n)
    for i, j in sorted(basis):
        if not dsu.union(node_id_row(i), node_id_col(j, m)):
            raise RuntimeError("initial basis unexpectedly contains a cycle")

    if len(basis) == needed:
        return

    for i in range(m):
        for j in range(n):
            if (i, j) in basis:
                continue
            a = node_id_row(i)
            b = node_id_col(j, m)
            if dsu.union(a, b):
                basis.add((i, j))
                if len(basis) == needed:
                    return

    raise RuntimeError("failed to augment basis to size m+n-1")


def compute_potentials(cost: np.ndarray, basis: Set[Cell]) -> Tuple[np.ndarray, np.ndarray]:
    """Solve dual potentials u,v from basis equations c_ij = u_i + v_j."""
    m, n = cost.shape
    u = np.full(m, np.nan, dtype=float)
    v = np.full(n, np.nan, dtype=float)

    row_to_cols: Dict[int, List[int]] = {i: [] for i in range(m)}
    col_to_rows: Dict[int, List[int]] = {j: [] for j in range(n)}
    for i, j in basis:
        row_to_cols[i].append(j)
        col_to_rows[j].append(i)

    u[0] = 0.0
    q: deque[Tuple[str, int]] = deque([("r", 0)])

    while q:
        typ, idx = q.popleft()
        if typ == "r":
            for j in row_to_cols[idx]:
                if np.isnan(v[j]):
                    v[j] = cost[idx, j] - u[idx]
                    q.append(("c", j))
        else:
            for i in col_to_rows[idx]:
                if np.isnan(u[i]):
                    u[i] = cost[i, idx] - v[idx]
                    q.append(("r", i))

    if np.isnan(u).any() or np.isnan(v).any():
        raise RuntimeError("basis is not connected; potentials are under-determined")
    return u, v


def select_entering_cell(
    cost: np.ndarray,
    basis: Set[Cell],
    u: np.ndarray,
    v: np.ndarray,
    tol: float,
) -> Tuple[Optional[Cell], float]:
    """Choose non-basic cell with most negative reduced cost."""
    m, n = cost.shape
    best_cell: Optional[Cell] = None
    best_rc = float("inf")

    for i in range(m):
        for j in range(n):
            if (i, j) in basis:
                continue
            rc = float(cost[i, j] - u[i] - v[j])
            if rc < best_rc:
                best_rc = rc
                best_cell = (i, j)

    if best_cell is None:
        raise RuntimeError("no non-basic cell found")

    if best_rc >= -tol:
        return None, best_rc
    return best_cell, best_rc


def find_cycle_cells(basis: Set[Cell], entering: Cell, m: int, n: int) -> List[Cell]:
    """Return ordered cycle cells: [entering, edge1, edge2, ...]."""
    start = ("r", entering[0])
    target = ("c", entering[1])

    adj: Dict[Tuple[str, int], List[Tuple[Tuple[str, int], Cell]]] = {}
    for i, j in basis:
        r = ("r", i)
        c = ("c", j)
        adj.setdefault(r, []).append((c, (i, j)))
        adj.setdefault(c, []).append((r, (i, j)))

    parent: Dict[Tuple[str, int], Tuple[str, int]] = {}
    edge_to_parent: Dict[Tuple[str, int], Cell] = {}
    q: deque[Tuple[str, int]] = deque([start])
    parent[start] = start

    found = False
    while q and not found:
        cur = q.popleft()
        for nxt, edge in adj.get(cur, []):
            if nxt in parent:
                continue
            parent[nxt] = cur
            edge_to_parent[nxt] = edge
            if nxt == target:
                found = True
                break
            q.append(nxt)

    if not found:
        raise RuntimeError("failed to find tree path for entering cell")

    path_cells_reversed: List[Cell] = []
    cur = target
    while cur != start:
        path_cells_reversed.append(edge_to_parent[cur])
        cur = parent[cur]

    path_cells = list(reversed(path_cells_reversed))
    cycle = [entering] + path_cells

    if len(cycle) < 4 or len(cycle) % 2 != 0:
        raise RuntimeError("invalid cycle length constructed")
    return cycle


def pivot_on_cycle(
    x: np.ndarray,
    basis: Set[Cell],
    entering: Cell,
    cycle: Sequence[Cell],
    tol: float,
) -> Tuple[Cell, float]:
    plus_cells = list(cycle[0::2])
    minus_cells = list(cycle[1::2])

    if entering in basis:
        raise RuntimeError("entering cell is already basic")
    if not minus_cells:
        raise RuntimeError("cycle does not contain '-' cells")

    theta = min(float(x[i, j]) for i, j in minus_cells)
    if theta < -tol:
        raise RuntimeError("negative theta encountered")

    leaving_candidates = [
        cell for cell in minus_cells if abs(float(x[cell]) - theta) <= tol
    ]
    if not leaving_candidates:
        raise RuntimeError("no leaving candidate found")
    leaving = min(leaving_candidates)

    basis.add(entering)

    for i, j in plus_cells:
        x[i, j] += theta
    for i, j in minus_cells:
        x[i, j] -= theta
        if abs(x[i, j]) <= tol:
            x[i, j] = 0.0

    basis.remove(leaving)
    return leaving, theta


def transportation_network_simplex(
    cost: np.ndarray,
    supply: np.ndarray,
    demand: np.ndarray,
    max_iter: int = 500,
    tol: float = 1e-10,
) -> Tuple[np.ndarray, Set[Cell], List[HistoryItem]]:
    validate_inputs(cost, supply, demand)

    x, basis = build_initial_solution_northwest(supply=supply, demand=demand)
    history: List[HistoryItem] = []

    for it in range(1, max_iter + 1):
        u, v = compute_potentials(cost=cost, basis=basis)
        entering, min_rc = select_entering_cell(
            cost=cost,
            basis=basis,
            u=u,
            v=v,
            tol=tol,
        )
        objective = float(np.sum(cost * x))

        if entering is None:
            history.append((it, objective, min_rc, None, None, 0.0))
            return x, basis, history

        cycle = find_cycle_cells(basis=basis, entering=entering, m=cost.shape[0], n=cost.shape[1])
        leaving, theta = pivot_on_cycle(
            x=x,
            basis=basis,
            entering=entering,
            cycle=cycle,
            tol=tol,
        )
        history.append((it, objective, min_rc, entering, leaving, theta))

    raise RuntimeError("max_iter reached before optimality")


def check_feasibility(x: np.ndarray, supply: np.ndarray, demand: np.ndarray) -> None:
    if np.any(x < -1e-9):
        raise RuntimeError("flow matrix has negative entries")
    row_sum = np.sum(x, axis=1)
    col_sum = np.sum(x, axis=0)
    if np.max(np.abs(row_sum - supply)) > 1e-8:
        raise RuntimeError("row sums do not match supply")
    if np.max(np.abs(col_sum - demand)) > 1e-8:
        raise RuntimeError("column sums do not match demand")


def solve_with_linprog(
    cost: np.ndarray,
    supply: np.ndarray,
    demand: np.ndarray,
) -> Optional[float]:
    """Optional oracle objective via LP; returns None if SciPy is unavailable."""
    if linprog is None:
        return None

    m, n = cost.shape
    c = cost.reshape(-1)

    num_vars = m * n
    a_eq = []
    b_eq = []

    for i in range(m):
        row = np.zeros(num_vars, dtype=float)
        for j in range(n):
            row[i * n + j] = 1.0
        a_eq.append(row)
        b_eq.append(float(supply[i]))

    for j in range(n):
        row = np.zeros(num_vars, dtype=float)
        for i in range(m):
            row[i * n + j] = 1.0
        a_eq.append(row)
        b_eq.append(float(demand[j]))

    res = linprog(
        c=c,
        A_eq=np.asarray(a_eq),
        b_eq=np.asarray(b_eq),
        bounds=[(0.0, None)] * num_vars,
        method="highs",
    )

    if not res.success:
        raise RuntimeError(f"linprog failed: {res.message}")
    return float(res.fun)


def print_history(history: Sequence[HistoryItem], max_lines: int = 20) -> None:
    print("iter | objective        | min_reduced_cost | entering   | leaving    | theta")
    print("-" * 86)
    for (it, obj, min_rc, entering, leaving, theta) in history[:max_lines]:
        entering_text = "-" if entering is None else f"{entering}"
        leaving_text = "-" if leaving is None else f"{leaving}"
        print(
            f"{it:4d} | {obj:16.8f} | {min_rc:16.8e} | "
            f"{entering_text:10s} | {leaving_text:10s} | {theta:10.6f}"
        )
    if len(history) > max_lines:
        print(f"... ({len(history) - max_lines} more iterations omitted)")


def main() -> None:
    # A fixed, balanced transportation instance (3 suppliers, 4 demand nodes).
    cost = np.array(
        [
            [2.0, 11.0, 10.0, 7.0],
            [7.0, 13.0, 2.0, 10.0],
            [3.0, 2.0, 8.0, 14.0],
        ],
        dtype=float,
    )
    supply = np.array([30.0, 40.0, 20.0], dtype=float)
    demand = np.array([20.0, 30.0, 25.0, 15.0], dtype=float)

    print("=== Network Simplex MVP (Transportation Special Case) ===")
    print("Cost matrix:")
    print(cost)
    print(f"Supply: {supply}")
    print(f"Demand: {demand}")

    x_opt, basis_opt, history = transportation_network_simplex(
        cost=cost,
        supply=supply,
        demand=demand,
        max_iter=500,
        tol=1e-10,
    )

    check_feasibility(x_opt, supply, demand)
    objective = float(np.sum(cost * x_opt))

    print("\n=== Iteration Log ===")
    print_history(history)

    print("\n=== Optimal Flow Matrix x ===")
    print(x_opt)
    print(f"Objective value: {objective:.8f}")
    print(f"Basic cells count: {len(basis_opt)} (expected m+n-1 = {cost.shape[0] + cost.shape[1] - 1})")

    lp_obj = solve_with_linprog(cost=cost, supply=supply, demand=demand)
    if lp_obj is None:
        print("SciPy not available; skipped LP cross-check.")
    else:
        gap = abs(objective - lp_obj)
        print(f"LP cross-check objective: {lp_obj:.8f}")
        print(f"Absolute objective gap: {gap:.8e}")


if __name__ == "__main__":
    main()

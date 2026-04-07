"""Minimal runnable MVP: network simplex for transportation min-cost flow."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np


EPS = 1e-9


@dataclass
class NetworkSimplexResult:
    flow: np.ndarray
    objective: float
    iterations: int
    optimal: bool
    basis_size: int


def _assert_balanced_transport(cost: np.ndarray, supply: np.ndarray, demand: np.ndarray) -> None:
    if cost.ndim != 2:
        raise ValueError("cost must be a 2D matrix")

    m, n = cost.shape
    if supply.shape != (m,):
        raise ValueError(f"supply shape must be ({m},)")
    if demand.shape != (n,):
        raise ValueError(f"demand shape must be ({n},)")

    if np.any(supply < -EPS) or np.any(demand < -EPS):
        raise ValueError("supply and demand must be nonnegative")

    if abs(float(np.sum(supply) - np.sum(demand))) > EPS:
        raise ValueError("transportation model must be balanced: sum(supply) == sum(demand)")


def _northwest_corner_init(supply: np.ndarray, demand: np.ndarray) -> tuple[np.ndarray, set[tuple[int, int]]]:
    """Build an initial BFS and basis tree for the transportation polytope."""
    m = supply.size
    n = demand.size

    rem_s = supply.astype(float).copy()
    rem_d = demand.astype(float).copy()
    flow = np.zeros((m, n), dtype=float)
    basis: set[tuple[int, int]] = set()

    i = 0
    j = 0
    while i < m and j < n:
        shipped = min(rem_s[i], rem_d[j])
        flow[i, j] = shipped
        basis.add((i, j))
        rem_s[i] -= shipped
        rem_d[j] -= shipped

        s_zero = rem_s[i] <= EPS
        d_zero = rem_d[j] <= EPS

        if s_zero and d_zero:
            # Degeneracy handling: keep basis size at m+n-1 by adding one zero-flow basic arc.
            if i < m - 1 and j < n - 1:
                basis.add((i, j + 1))
                i += 1
                j += 1
            elif i == m - 1 and j < n - 1:
                j += 1
            elif j == n - 1 and i < m - 1:
                i += 1
            else:
                break
        elif s_zero:
            i += 1
        else:
            j += 1

    expected_basis_size = m + n - 1
    if len(basis) != expected_basis_size:
        raise RuntimeError(
            f"invalid initial basis size: got {len(basis)}, expected {expected_basis_size}"
        )

    return flow, basis


def _compute_potentials(
    cost: np.ndarray, basis: set[tuple[int, int]], m: int, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Solve u_i + v_j = c_ij on basis arcs."""
    u = np.full(m, np.nan, dtype=float)
    v = np.full(n, np.nan, dtype=float)
    u[0] = 0.0

    changed = True
    while changed:
        changed = False
        for i, j in basis:
            if np.isnan(u[i]) and not np.isnan(v[j]):
                u[i] = cost[i, j] - v[j]
                changed = True
            elif not np.isnan(u[i]) and np.isnan(v[j]):
                v[j] = cost[i, j] - u[i]
                changed = True

    if np.isnan(u).any() or np.isnan(v).any():
        raise RuntimeError("basis graph is disconnected; cannot compute node potentials")

    return u, v


def _choose_entering_arc(
    cost: np.ndarray,
    basis: set[tuple[int, int]],
    u: np.ndarray,
    v: np.ndarray,
    tol: float,
) -> tuple[int, int] | None:
    """Bland-style pricing: choose lexicographically smallest arc with negative reduced cost."""
    m, n = cost.shape
    candidates: list[tuple[int, int]] = []

    for i in range(m):
        for j in range(n):
            if (i, j) in basis:
                continue
            reduced_cost = cost[i, j] - u[i] - v[j]
            if reduced_cost < -tol:
                candidates.append((i, j))

    if not candidates:
        return None

    return min(candidates)


def _basis_path_nodes(
    basis: set[tuple[int, int]], start_row: int, end_col: int
) -> list[tuple[str, int]]:
    """Find unique path in the basis tree from row node R_start to column node C_end."""
    start = ("r", start_row)
    goal = ("c", end_col)

    adj: dict[tuple[str, int], list[tuple[str, int]]] = {}
    for i, j in basis:
        r_node = ("r", i)
        c_node = ("c", j)
        adj.setdefault(r_node, []).append(c_node)
        adj.setdefault(c_node, []).append(r_node)

    q: deque[tuple[str, int]] = deque([start])
    parent: dict[tuple[str, int], tuple[str, int] | None] = {start: None}

    while q:
        cur = q.popleft()
        if cur == goal:
            break
        for nxt in adj.get(cur, []):
            if nxt in parent:
                continue
            parent[nxt] = cur
            q.append(nxt)

    if goal not in parent:
        raise RuntimeError("failed to find cycle path in basis tree")

    path_nodes: list[tuple[str, int]] = []
    cur: tuple[str, int] | None = goal
    while cur is not None:
        path_nodes.append(cur)
        cur = parent[cur]
    path_nodes.reverse()

    return path_nodes


def _path_edges_from_nodes(path_nodes: list[tuple[str, int]]) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    for a, b in zip(path_nodes, path_nodes[1:]):
        if a[0] == "r" and b[0] == "c":
            edges.append((a[1], b[1]))
        elif a[0] == "c" and b[0] == "r":
            edges.append((b[1], a[1]))
        else:
            raise RuntimeError("invalid path: consecutive nodes share same side")
    return edges


def _objective(cost: np.ndarray, flow: np.ndarray) -> float:
    return float(np.sum(cost * flow))


def transportation_network_simplex(
    cost: np.ndarray,
    supply: Iterable[float],
    demand: Iterable[float],
    *,
    max_iter: int = 10_000,
    tol: float = 1e-9,
) -> NetworkSimplexResult:
    """Solve balanced transportation min-cost flow via network simplex pivots."""
    cost = np.asarray(cost, dtype=float)
    supply_arr = np.asarray(list(supply), dtype=float)
    demand_arr = np.asarray(list(demand), dtype=float)

    _assert_balanced_transport(cost, supply_arr, demand_arr)

    m, n = cost.shape
    flow, basis = _northwest_corner_init(supply_arr, demand_arr)

    optimal = False
    iterations = 0

    for it in range(max_iter):
        iterations = it + 1
        u, v = _compute_potentials(cost, basis, m, n)
        entering = _choose_entering_arc(cost, basis, u, v, tol)

        if entering is None:
            optimal = True
            break

        path_nodes = _basis_path_nodes(basis, start_row=entering[0], end_col=entering[1])
        path_edges = _path_edges_from_nodes(path_nodes)

        # Cycle signs: + entering, then path edge signs alternate -, +, -, ...
        minus_edges = path_edges[0::2]
        plus_edges = [entering] + path_edges[1::2]

        theta = min(flow[i, j] for i, j in minus_edges)
        leaving_candidates = sorted(
            (i, j)
            for i, j in minus_edges
            if abs(flow[i, j] - theta) <= tol
        )
        if not leaving_candidates:
            raise RuntimeError("no leaving arc found; pivot is ill-defined")
        leaving = leaving_candidates[0]

        for i, j in plus_edges:
            flow[i, j] += theta
        for i, j in minus_edges:
            flow[i, j] -= theta
            if abs(flow[i, j]) <= tol:
                flow[i, j] = 0.0

        basis.add(entering)
        basis.remove(leaving)
    else:
        iterations = max_iter

    # Feasibility check (row/column marginals).
    row_res = np.max(np.abs(np.sum(flow, axis=1) - supply_arr))
    col_res = np.max(np.abs(np.sum(flow, axis=0) - demand_arr))
    if row_res > 1e-7 or col_res > 1e-7:
        raise RuntimeError(
            f"numerical infeasibility detected: row_res={row_res:.3e}, col_res={col_res:.3e}"
        )

    return NetworkSimplexResult(
        flow=flow,
        objective=_objective(cost, flow),
        iterations=iterations,
        optimal=optimal,
        basis_size=len(basis),
    )


def _run_case(name: str, cost: np.ndarray, supply: list[float], demand: list[float]) -> None:
    result = transportation_network_simplex(cost, supply, demand)
    print(f"\\n=== {name} ===")
    print(f"optimal={result.optimal}, iterations={result.iterations}, basis_size={result.basis_size}")
    print(f"objective={result.objective:.6f}")
    print("flow matrix:")
    print(np.array2string(result.flow, precision=3, suppress_small=True))


def main() -> None:
    # Case 1: non-degenerate medium example.
    cost1 = np.array(
        [
            [8, 6, 10, 9],
            [9, 7, 4, 2],
            [3, 4, 2, 5],
        ],
        dtype=float,
    )
    supply1 = [35, 50, 40]
    demand1 = [45, 20, 30, 30]

    # Case 2: includes degeneracy in the initial basic feasible solution.
    cost2 = np.array(
        [
            [2, 3, 1],
            [5, 4, 8],
            [5, 6, 8],
        ],
        dtype=float,
    )
    supply2 = [20, 30, 25]
    demand2 = [10, 10, 55]

    _run_case("Case-1", cost1, supply1, demand1)
    _run_case("Case-2", cost2, supply2, demand2)


if __name__ == "__main__":
    main()

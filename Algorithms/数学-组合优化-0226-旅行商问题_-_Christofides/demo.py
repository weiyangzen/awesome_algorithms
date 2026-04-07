"""Christofides algorithm MVP for metric TSP.

This script is self-contained and deterministic. It runs several Euclidean TSP
instances, compares Christofides against exact Held-Karp, and validates the
1.5-approximation bound on those metric instances.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Sequence, Tuple

import numpy as np


Edge = Tuple[int, int]


@dataclass
class ChristofidesResult:
    tour: List[int]
    length: float
    mst_weight: float
    matching_weight: float
    odd_vertices: List[int]


@dataclass
class InstanceReport:
    seed: int
    n: int
    christofides: ChristofidesResult
    optimal_tour: List[int]
    optimal_length: float
    nearest_neighbor_tour: List[int]
    nearest_neighbor_length: float


def euclidean_distance_matrix(points: np.ndarray) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    diff = points[:, None, :] - points[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, 0.0)
    return dist


def tour_length(tour: Sequence[int], dist: np.ndarray) -> float:
    if len(tour) < 2:
        return 0.0
    total = 0.0
    for i in range(len(tour) - 1):
        total += dist[tour[i], tour[i + 1]]
    return float(total)


def prim_mst(dist: np.ndarray) -> List[Edge]:
    n = dist.shape[0]
    if n == 0:
        return []

    in_tree = np.zeros(n, dtype=bool)
    min_cost = np.full(n, np.inf)
    parent = np.full(n, -1, dtype=int)

    min_cost[0] = 0.0
    edges: List[Edge] = []

    for _ in range(n):
        candidates = np.where(~in_tree)[0]
        u = int(candidates[np.argmin(min_cost[candidates])])
        in_tree[u] = True

        if parent[u] != -1:
            edges.append((int(parent[u]), u))

        for v in np.where(~in_tree)[0]:
            w = dist[u, v]
            if w < min_cost[v]:
                min_cost[v] = w
                parent[v] = u

    return edges


def odd_degree_vertices(n: int, edges: Sequence[Edge]) -> List[int]:
    deg = np.zeros(n, dtype=int)
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    return [i for i in range(n) if deg[i] % 2 == 1]


def min_weight_perfect_matching_dp(odd_vertices: Sequence[int], dist: np.ndarray) -> Tuple[List[Edge], float]:
    m = len(odd_vertices)
    if m == 0:
        return [], 0.0
    if m % 2 != 0:
        raise ValueError("odd-vertex set size must be even")

    local_dist = np.zeros((m, m), dtype=float)
    for i in range(m):
        for j in range(m):
            local_dist[i, j] = dist[odd_vertices[i], odd_vertices[j]]

    choice: dict[int, Tuple[int, int]] = {}

    @lru_cache(maxsize=None)
    def solve(mask: int) -> float:
        if mask == 0:
            return 0.0

        i = (mask & -mask).bit_length() - 1
        best = float("inf")
        best_pair = (-1, -1)

        rest = mask ^ (1 << i)
        j_mask = rest
        while j_mask:
            j_bit = j_mask & -j_mask
            j = j_bit.bit_length() - 1
            cost = local_dist[i, j] + solve(rest ^ (1 << j))
            if cost < best:
                best = cost
                best_pair = (i, j)
            j_mask ^= j_bit

        choice[mask] = best_pair
        return best

    full_mask = (1 << m) - 1
    best_weight = solve(full_mask)

    matching: List[Edge] = []
    mask = full_mask
    while mask:
        i, j = choice[mask]
        matching.append((int(odd_vertices[i]), int(odd_vertices[j])))
        mask ^= (1 << i)
        mask ^= (1 << j)

    return matching, float(best_weight)


def build_multigraph(n: int, edges_a: Sequence[Edge], edges_b: Sequence[Edge]) -> List[List[int]]:
    adj: List[List[int]] = [[] for _ in range(n)]
    for u, v in list(edges_a) + list(edges_b):
        adj[u].append(v)
        adj[v].append(u)
    return adj


def eulerian_tour(multigraph_adj: List[List[int]], start: int = 0) -> List[int]:
    # Copy adjacency because Hierholzer consumes edges.
    adj = [lst.copy() for lst in multigraph_adj]
    stack = [start]
    tour: List[int] = []

    while stack:
        u = stack[-1]
        if adj[u]:
            v = adj[u].pop()
            adj[v].remove(u)
            stack.append(v)
        else:
            tour.append(stack.pop())

    tour.reverse()
    return tour


def shortcut_to_hamiltonian(euler: Sequence[int], n: int) -> List[int]:
    seen = np.zeros(n, dtype=bool)
    cycle: List[int] = []

    for v in euler:
        if not seen[v]:
            cycle.append(v)
            seen[v] = True

    if len(cycle) != n:
        raise ValueError("shortcutting failed to visit all vertices exactly once")

    cycle.append(cycle[0])
    return cycle


def christofides_tsp(dist: np.ndarray) -> ChristofidesResult:
    n = dist.shape[0]
    if n < 3:
        raise ValueError("Christofides MVP expects n >= 3")

    mst_edges = prim_mst(dist)
    mst_weight = sum(dist[u, v] for u, v in mst_edges)

    odd = odd_degree_vertices(n, mst_edges)
    matching_edges, matching_weight = min_weight_perfect_matching_dp(odd, dist)

    multigraph = build_multigraph(n, mst_edges, matching_edges)
    euler = eulerian_tour(multigraph, start=0)
    tour = shortcut_to_hamiltonian(euler, n)

    return ChristofidesResult(
        tour=tour,
        length=tour_length(tour, dist),
        mst_weight=float(mst_weight),
        matching_weight=float(matching_weight),
        odd_vertices=odd,
    )


def held_karp_exact_tsp(dist: np.ndarray) -> Tuple[List[int], float]:
    """Exact TSP by Held-Karp DP, start/end at vertex 0."""
    n = dist.shape[0]
    if n < 2:
        raise ValueError("n must be >= 2")

    @lru_cache(maxsize=None)
    def dp(mask: int, j: int) -> float:
        # Minimum path cost: 0 -> ... -> j, visiting exactly vertices in mask.
        if mask == (1 << j):
            return dist[0, j]

        prev_mask = mask ^ (1 << j)
        best = float("inf")
        k_mask = prev_mask
        while k_mask:
            k_bit = k_mask & -k_mask
            k = k_bit.bit_length() - 1
            if k != j:
                best = min(best, dp(prev_mask, k) + dist[k, j])
            k_mask ^= k_bit
        return best

    # Subsets over vertices {1..n-1}, represented in n-bit masks.
    full = 0
    for v in range(1, n):
        full |= (1 << v)

    best_cost = float("inf")
    best_last = -1
    for j in range(1, n):
        cost = dp(full, j) + dist[j, 0]
        if cost < best_cost:
            best_cost = cost
            best_last = j

    # Reconstruct route.
    route_rev = [0, best_last]
    mask = full
    j = best_last
    while mask != (1 << j):
        prev_mask = mask ^ (1 << j)
        pick = -1
        best = float("inf")
        k_mask = prev_mask
        while k_mask:
            k_bit = k_mask & -k_mask
            k = k_bit.bit_length() - 1
            val = dp(prev_mask, k) + dist[k, j]
            if val < best:
                best = val
                pick = k
            k_mask ^= k_bit
        route_rev.append(pick)
        mask = prev_mask
        j = pick

    route = [0] + list(reversed(route_rev[1:])) + [0]
    return route, float(best_cost)


def nearest_neighbor_tsp(dist: np.ndarray, start: int = 0) -> Tuple[List[int], float]:
    n = dist.shape[0]
    unvisited = set(range(n))
    unvisited.remove(start)
    tour = [start]
    cur = start

    while unvisited:
        nxt = min(unvisited, key=lambda v: dist[cur, v])
        tour.append(nxt)
        unvisited.remove(nxt)
        cur = nxt

    tour.append(start)
    return tour, tour_length(tour, dist)


def make_euclidean_instance(n: int, seed: int) -> np.ndarray:
    if n < 3:
        raise ValueError("n must be >= 3")
    rng = np.random.default_rng(seed)
    points = rng.uniform(0.0, 100.0, size=(n, 2))
    return euclidean_distance_matrix(points)


def verify_tour_shape(tour: Sequence[int], n: int) -> None:
    if len(tour) != n + 1:
        raise AssertionError(f"tour length expected {n + 1}, got {len(tour)}")
    if tour[0] != tour[-1]:
        raise AssertionError("tour must start and end at same node")
    body = tour[:-1]
    if len(set(body)) != n:
        raise AssertionError("tour must visit each vertex exactly once")


def run_one_instance(n: int, seed: int) -> InstanceReport:
    dist = make_euclidean_instance(n=n, seed=seed)
    christofides = christofides_tsp(dist)
    optimum_tour, optimum_length = held_karp_exact_tsp(dist)
    nn_tour, nn_length = nearest_neighbor_tsp(dist)

    verify_tour_shape(christofides.tour, n)
    verify_tour_shape(optimum_tour, n)

    ratio = christofides.length / optimum_length
    if ratio > 1.5 + 1e-9:
        raise AssertionError(f"approx ratio violated: {ratio:.6f} > 1.5")

    return InstanceReport(
        seed=seed,
        n=n,
        christofides=christofides,
        optimal_tour=optimum_tour,
        optimal_length=optimum_length,
        nearest_neighbor_tour=nn_tour,
        nearest_neighbor_length=nn_length,
    )


def print_report(report: InstanceReport) -> None:
    ch = report.christofides
    ratio = ch.length / report.optimal_length
    nn_ratio = report.nearest_neighbor_length / report.optimal_length

    print("=" * 72)
    print(f"Instance seed={report.seed}, n={report.n}")
    print(f"Odd-degree vertices in MST: {ch.odd_vertices}")
    print(f"MST weight:               {ch.mst_weight:.6f}")
    print(f"Matching weight:          {ch.matching_weight:.6f}")
    print(f"Christofides length:      {ch.length:.6f}")
    print(f"Optimal length (Held-Karp){report.optimal_length:>12.6f}")
    print(f"Nearest-neighbor length:  {report.nearest_neighbor_length:.6f}")
    print(f"Approx ratio (Chr/Opt):   {ratio:.6f}")
    print(f"Baseline ratio (NN/Opt):  {nn_ratio:.6f}")
    print(f"Christofides tour:        {ch.tour}")
    print(f"Optimal tour:             {report.optimal_tour}")


def main() -> None:
    configs = [
        (9, 7),
        (10, 17),
        (10, 29),
    ]

    reports = [run_one_instance(n=n, seed=seed) for n, seed in configs]
    for rep in reports:
        print_report(rep)

    avg_ratio = sum(rep.christofides.length / rep.optimal_length for rep in reports) / len(reports)
    print("=" * 72)
    print(f"All {len(reports)} instances passed feasibility + 1.5 bound checks.")
    print(f"Average Christofides/Optimal ratio: {avg_ratio:.6f}")


if __name__ == "__main__":
    main()

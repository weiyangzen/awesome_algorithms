"""TSP nearest-neighbor heuristic MVP.

Run:
    python3 demo.py
"""

from __future__ import annotations

import math
import time
from itertools import permutations
from typing import List, Sequence, Tuple

import numpy as np


def build_cities(n_cities: int = 20, seed: int = 221) -> np.ndarray:
    """Generate deterministic 2D points as TSP cities."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 100.0, size=(n_cities, 2))


def pairwise_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Compute Euclidean pairwise distance matrix."""
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (n, 2)")
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, 0.0)
    return dist


def tour_length(tour: Sequence[int], dist: np.ndarray) -> float:
    """Compute closed-tour length for a permutation-style tour."""
    idx = np.asarray(tour, dtype=np.int64)
    nxt = np.roll(idx, -1)
    return float(dist[idx, nxt].sum())


def validate_tour(tour: Sequence[int], n: int) -> None:
    """Validate that tour is a full permutation of [0, n)."""
    if len(tour) != n:
        raise ValueError("tour length mismatch")
    if len(set(tour)) != n:
        raise ValueError("tour is not a permutation")
    if min(tour) < 0 or max(tour) >= n:
        raise ValueError("tour node index out of range")


def nearest_neighbor_tour(dist: np.ndarray, start: int = 0) -> List[int]:
    """Construct one greedy nearest-neighbor tour from a fixed start node.

    Tie-break rule: choose the smallest city index among equal distances.
    """
    n = dist.shape[0]
    if dist.ndim != 2 or n != dist.shape[1]:
        raise ValueError("dist must be a square matrix")
    if not (0 <= start < n):
        raise ValueError("start out of range")

    visited = np.zeros(n, dtype=bool)
    visited[start] = True

    tour = [start]
    current = start

    for _ in range(n - 1):
        candidates = np.flatnonzero(~visited)
        cand_dist = dist[current, candidates]
        best_idx = int(np.argmin(cand_dist))
        nxt = int(candidates[best_idx])

        if not math.isfinite(float(dist[current, nxt])):
            raise ValueError("graph has unreachable step for nearest-neighbor expansion")

        tour.append(nxt)
        visited[nxt] = True
        current = nxt

    return tour


def multi_start_nearest_neighbor(dist: np.ndarray) -> Tuple[List[int], float, int]:
    """Run nearest-neighbor from all starts and return the best tour found."""
    n = dist.shape[0]
    best_tour: List[int] | None = None
    best_cost = np.inf
    best_start = -1

    for s in range(n):
        tour = nearest_neighbor_tour(dist, start=s)
        cost = tour_length(tour, dist)
        if cost < best_cost:
            best_tour = tour
            best_cost = cost
            best_start = s

    if best_tour is None:
        raise RuntimeError("failed to construct any nearest-neighbor tour")

    return best_tour, float(best_cost), best_start


def brute_force_tsp(dist: np.ndarray, start: int = 0) -> Tuple[float, List[int]]:
    """Exact TSP by brute force for very small n, used only for sanity comparison."""
    n = dist.shape[0]
    if n > 10:
        raise ValueError("brute_force_tsp is only for n <= 10")

    nodes = [i for i in range(n) if i != start]
    best_cost = np.inf
    best_tour: List[int] | None = None

    for perm in permutations(nodes):
        tour = [start, *perm]
        cost = tour_length(tour, dist)
        if cost < best_cost:
            best_cost = cost
            best_tour = list(tour)

    if best_tour is None:
        raise RuntimeError("no feasible tour found in brute force")

    return float(best_cost), best_tour


def main() -> None:
    n_cities = 20
    coords = build_cities(n_cities=n_cities, seed=221)
    dist = pairwise_distance_matrix(coords)

    t0 = time.perf_counter()
    tour_s0 = nearest_neighbor_tour(dist, start=0)
    cost_s0 = tour_length(tour_s0, dist)
    t1 = time.perf_counter()

    tour_ms, cost_ms, best_start = multi_start_nearest_neighbor(dist)
    t2 = time.perf_counter()

    validate_tour(tour_s0, n_cities)
    validate_tour(tour_ms, n_cities)
    assert cost_ms <= cost_s0 + 1e-12

    # Small-scale exact comparison to show heuristic gap (optional but deterministic).
    n_small = 9
    dist_small = dist[:n_small, :n_small]
    nn_small = nearest_neighbor_tour(dist_small, start=0)
    nn_small_cost = tour_length(nn_small, dist_small)
    opt_small_cost, opt_small_tour = brute_force_tsp(dist_small, start=0)

    print("TSP Nearest-Neighbor MVP")
    print(f"Cities                       : {n_cities}")
    print("=" * 72)
    print(f"NN(start=0) cost             : {cost_s0:10.3f}")
    print(f"Multi-start NN best cost     : {cost_ms:10.3f} (best_start={best_start})")
    print(f"Improvement over start=0 (%) : {(cost_s0 - cost_ms) / cost_s0 * 100:10.3f}")
    print("-" * 72)
    print(f"NN(start=0) time (ms)        : {(t1 - t0) * 1000:10.3f}")
    print(f"Multi-start time (ms)        : {(t2 - t1) * 1000:10.3f}")
    print("=" * 72)
    print("Tour prefix (start=0, first 12 nodes):", tour_s0[:12])
    print("Tour prefix (multi-start, first 12):", tour_ms[:12])
    print("=" * 72)
    print(f"Small exact check n={n_small}")
    print(f"NN cost                      : {nn_small_cost:10.3f}")
    print(f"Optimal cost (brute force)   : {opt_small_cost:10.3f}")
    print(f"Heuristic gap (%)            : {(nn_small_cost - opt_small_cost) / opt_small_cost * 100:10.3f}")
    print(f"Optimal tour                 : {opt_small_tour}")
    print("All checks passed.")


if __name__ == "__main__":
    main()

"""State-compression DP MVP: Held-Karp TSP with path reconstruction.

The script solves fixed demo instances of the Traveling Salesman Problem (TSP)
using bitmask dynamic programming, then cross-checks the optimum with brute
force permutation enumeration on small instances.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TSPResult:
    """Container for one TSP solution."""

    best_cost: float
    tour: List[int]



def validate_distance_matrix(distance: np.ndarray, start: int) -> np.ndarray:
    """Validate and normalize distance matrix input."""
    dist = np.asarray(distance, dtype=np.float64)

    if dist.ndim != 2 or dist.shape[0] != dist.shape[1]:
        raise ValueError(f"distance must be a square matrix, got shape={dist.shape}")

    n = dist.shape[0]
    if n < 2:
        raise ValueError("distance matrix must have at least 2 nodes")

    if not (0 <= start < n):
        raise ValueError(f"start must be in [0, {n - 1}], got {start}")

    if not np.all(np.isfinite(dist)):
        raise ValueError("distance matrix contains non-finite values")

    if np.any(dist < 0):
        raise ValueError("distance matrix must be non-negative")

    if not np.allclose(np.diag(dist), 0.0):
        raise ValueError("distance diagonal must be zero")

    return dist



def held_karp_tsp(distance: np.ndarray, start: int = 0) -> TSPResult:
    """Solve TSP by state-compression DP (Held-Karp).

    DP definition:
        dp[mask, j] = minimum cost of a path that starts at `start`,
                      visits exactly the node set in `mask`, and ends at `j`.

    `mask` is a bitmask over nodes (0..n-1).
    """
    dist = validate_distance_matrix(distance, start)
    n = dist.shape[0]

    if n > 20:
        raise ValueError("MVP implementation limits n <= 20 to avoid huge memory/time")

    start_mask = 1 << start
    full_mask = (1 << n) - 1

    dp = np.full((1 << n, n), np.inf, dtype=np.float64)
    parent = np.full((1 << n, n), -1, dtype=np.int64)
    dp[start_mask, start] = 0.0

    for mask in range(1 << n):
        if (mask & start_mask) == 0:
            continue

        for end in range(n):
            if (mask & (1 << end)) == 0:
                continue
            if end == start and mask != start_mask:
                continue

            if mask == start_mask and end == start:
                continue

            prev_mask = mask ^ (1 << end)
            best_prev_cost = np.inf
            best_prev_node = -1

            for prev in range(n):
                if (prev_mask & (1 << prev)) == 0:
                    continue

                candidate = dp[prev_mask, prev] + dist[prev, end]
                if candidate < best_prev_cost:
                    best_prev_cost = float(candidate)
                    best_prev_node = prev

            dp[mask, end] = best_prev_cost
            parent[mask, end] = best_prev_node

    best_cost = np.inf
    best_end = -1

    for end in range(n):
        if end == start:
            continue

        candidate = dp[full_mask, end] + dist[end, start]
        if candidate < best_cost:
            best_cost = float(candidate)
            best_end = end

    if not np.isfinite(best_cost):
        raise RuntimeError("failed to build a valid Hamiltonian cycle")

    path_reversed: List[int] = []
    mask = full_mask
    end = best_end

    while end != start:
        path_reversed.append(end)
        prev = int(parent[mask, end])
        if prev == -1:
            raise RuntimeError("path reconstruction failed due to missing parent")
        mask ^= 1 << end
        end = prev

    forward_path = [start] + list(reversed(path_reversed))
    tour = forward_path + [start]

    return TSPResult(best_cost=best_cost, tour=tour)



def brute_force_tsp(distance: np.ndarray, start: int = 0) -> TSPResult:
    """Brute-force TSP solver for small instances, used as correctness oracle."""
    dist = validate_distance_matrix(distance, start)
    n = dist.shape[0]

    if n > 10:
        raise ValueError("brute_force_tsp is limited to n <= 10")

    nodes = [i for i in range(n) if i != start]

    best_cost = np.inf
    best_tour: List[int] = []

    for perm in permutations(nodes):
        tour = [start, *perm, start]
        cost = 0.0
        for i in range(len(tour) - 1):
            cost += float(dist[tour[i], tour[i + 1]])

        if cost < best_cost:
            best_cost = cost
            best_tour = list(tour)

    return TSPResult(best_cost=best_cost, tour=best_tour)



def tour_to_dataframe(tour: List[int], distance: np.ndarray) -> pd.DataFrame:
    """Format a tour into a readable table."""
    rows = []
    total = 0.0

    for step in range(len(tour) - 1):
        frm = tour[step]
        to = tour[step + 1]
        edge = float(distance[frm, to])
        total += edge
        rows.append(
            {
                "step": step,
                "from": frm,
                "to": to,
                "edge_cost": round(edge, 4),
                "prefix_cost": round(total, 4),
            }
        )

    return pd.DataFrame(rows)



def run_case(case_name: str, distance: np.ndarray, start: int = 0) -> None:
    """Run one deterministic case and print DP + brute-force comparison."""
    dist = validate_distance_matrix(distance, start)
    n = dist.shape[0]

    dp_result = held_karp_tsp(dist, start=start)
    brute_result = brute_force_tsp(dist, start=start)

    dp_df = tour_to_dataframe(dp_result.tour, dist)

    print(f"=== {case_name} ===")
    print(f"n_nodes = {n}, start = {start}")
    print("distance matrix:")
    print(pd.DataFrame(np.round(dist, 3)).to_string(index=True))

    print("\nDP best tour:")
    print(dp_result.tour)
    print(f"DP best cost       = {dp_result.best_cost:.6f}")

    print("\nBrute-force best tour:")
    print(brute_result.tour)
    print(f"Brute-force cost   = {brute_result.best_cost:.6f}")

    print("\nTour detail (DP result):")
    print(dp_df.to_string(index=False))

    same_cost = bool(np.isclose(dp_result.best_cost, brute_result.best_cost, atol=1e-9))
    print(f"\ndp_matches_bruteforce = {same_cost}")

    if not same_cost:
        raise RuntimeError("Held-Karp DP cost does not match brute-force oracle")



def build_demo_matrix() -> np.ndarray:
    """Create a fixed symmetric Euclidean distance matrix."""
    coords = np.array(
        [
            [0.0, 0.0],
            [2.0, 4.0],
            [5.0, 1.0],
            [6.0, 6.0],
            [8.0, 3.0],
            [1.0, 7.0],
        ],
        dtype=np.float64,
    )

    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, 0.0)
    return dist



def main() -> None:
    # Fixed, deterministic data. No interactive input.
    dist = build_demo_matrix()
    run_case("Case 1: 6-city Euclidean TSP", dist, start=0)


if __name__ == "__main__":
    main()

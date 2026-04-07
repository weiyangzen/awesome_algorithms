"""Minimal runnable MVP for TSP (Traveling Salesman Problem) via Held-Karp DP."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import List, Optional, Sequence

import numpy as np


@dataclass
class HeldKarpResult:
    best_cost: float
    tour: Optional[List[int]]
    start: int
    states_explored: int


def validate_distance_matrix(dist: np.ndarray) -> np.ndarray:
    """Validate and normalize an input distance matrix."""
    if dist.ndim != 2 or dist.shape[0] != dist.shape[1]:
        raise ValueError("distance matrix must be square")
    if dist.shape[0] < 2:
        raise ValueError("distance matrix must have size >= 2")
    if not np.all(np.isfinite(np.diag(dist))):
        raise ValueError("diagonal values must be finite")

    mat = dist.astype(float, copy=True)
    np.fill_diagonal(mat, 0.0)

    if np.any(np.isnan(mat)):
        raise ValueError("distance matrix must not contain NaN")
    return mat


def held_karp_tsp(dist: np.ndarray, start: int = 0) -> HeldKarpResult:
    """
    Solve TSP exactly by Held-Karp dynamic programming.

    Supports asymmetric matrices and optional +inf edges.
    Returns an empty solution (tour=None, cost=inf) when no Hamiltonian tour exists.
    """
    mat = validate_distance_matrix(dist)
    n = mat.shape[0]
    if not (0 <= start < n):
        raise ValueError("start index out of range")

    start_mask = 1 << start
    full_mask = (1 << n) - 1

    dp = np.full((1 << n, n), np.inf, dtype=float)
    parent = np.full((1 << n, n), -1, dtype=int)
    dp[start_mask, start] = 0.0

    states_explored = 0

    for mask in range(1 << n):
        if (mask & start_mask) == 0:
            continue

        for last in range(n):
            base_cost = dp[mask, last]
            if not np.isfinite(base_cost):
                continue

            states_explored += 1

            for nxt in range(n):
                if mask & (1 << nxt):
                    continue
                edge = mat[last, nxt]
                if not np.isfinite(edge):
                    continue

                new_mask = mask | (1 << nxt)
                cand = base_cost + edge
                if cand < dp[new_mask, nxt]:
                    dp[new_mask, nxt] = cand
                    parent[new_mask, nxt] = last

    best_cost = np.inf
    best_end = -1

    for end in range(n):
        if end == start:
            continue
        tour_cost = dp[full_mask, end] + mat[end, start]
        if tour_cost < best_cost:
            best_cost = tour_cost
            best_end = end

    if not np.isfinite(best_cost):
        return HeldKarpResult(
            best_cost=np.inf,
            tour=None,
            start=start,
            states_explored=states_explored,
        )

    reverse_path = [best_end]
    mask = full_mask
    cur = best_end

    while cur != start:
        prev = int(parent[mask, cur])
        if prev < 0:
            # Defensive fallback; in theory should not happen after a finite solution.
            return HeldKarpResult(
                best_cost=np.inf,
                tour=None,
                start=start,
                states_explored=states_explored,
            )
        mask ^= 1 << cur
        cur = prev
        reverse_path.append(cur)

    forward_path = list(reversed(reverse_path))
    tour = forward_path + [start]

    return HeldKarpResult(
        best_cost=float(best_cost),
        tour=tour,
        start=start,
        states_explored=states_explored,
    )


def tour_cost(dist: np.ndarray, tour: Sequence[int]) -> float:
    """Compute total tour length for a cycle path such as [0, 2, 1, 0]."""
    total = 0.0
    for i in range(len(tour) - 1):
        total += float(dist[tour[i], tour[i + 1]])
    return total


def brute_force_tsp(dist: np.ndarray, start: int = 0) -> HeldKarpResult:
    """Small-instance checker for MVP verification."""
    mat = validate_distance_matrix(dist)
    n = mat.shape[0]
    if n > 10:
        raise ValueError("brute_force_tsp is only intended for n <= 10")

    nodes = [i for i in range(n) if i != start]
    best_cost = np.inf
    best_tour: Optional[List[int]] = None

    for perm in permutations(nodes):
        tour = [start, *perm, start]
        cost = tour_cost(mat, tour)
        if cost < best_cost:
            best_cost = cost
            best_tour = list(tour)

    return HeldKarpResult(
        best_cost=float(best_cost),
        tour=best_tour,
        start=start,
        states_explored=0,
    )


def pairwise_euclidean(points: np.ndarray) -> np.ndarray:
    """Construct Euclidean distance matrix from 2D points."""
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must be of shape (n, 2)")
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def format_matrix(mat: np.ndarray) -> str:
    lines: List[str] = []
    for row in mat:
        parts: List[str] = []
        for x in row:
            if np.isinf(x):
                parts.append("  inf")
            else:
                parts.append(f"{x:6.2f}")
        lines.append(" ".join(parts))
    return "\n".join(lines)


def run_case_euclidean() -> None:
    print("=== Case A: Symmetric Euclidean TSP (exact Held-Karp) ===")
    points = np.array(
        [
            [0.0, 0.0],
            [1.5, 4.0],
            [4.0, 3.5],
            [6.0, 1.0],
            [3.0, -1.0],
            [1.0, -2.0],
        ],
        dtype=float,
    )
    dist = pairwise_euclidean(points)

    print("Distance matrix:")
    print(format_matrix(dist))

    hk = held_karp_tsp(dist, start=0)
    bf = brute_force_tsp(dist, start=0)

    print(f"\nHeld-Karp best cost: {hk.best_cost:.6f}")
    print(f"Held-Karp tour: {hk.tour}")
    print(f"DP states explored: {hk.states_explored}")
    print(f"Brute-force best cost: {bf.best_cost:.6f}")
    print(f"Brute-force tour: {bf.tour}")

    ok = np.isclose(hk.best_cost, bf.best_cost)
    print(f"Cost check passed: {ok}")


def run_case_asymmetric() -> None:
    print("\n=== Case B: Asymmetric TSP matrix ===")
    dist = np.array(
        [
            [0.0, 14.0, 4.0, 10.0, 20.0],
            [14.0, 0.0, 7.0, 8.0, 7.0],
            [4.0, 5.0, 0.0, 7.0, 16.0],
            [11.0, 7.0, 9.0, 0.0, 2.0],
            [18.0, 7.0, 17.0, 4.0, 0.0],
        ],
        dtype=float,
    )

    print("Distance matrix:")
    print(format_matrix(dist))

    hk = held_karp_tsp(dist, start=0)
    print(f"\nHeld-Karp best cost: {hk.best_cost:.6f}")
    print(f"Held-Karp tour: {hk.tour}")
    if hk.tour is not None:
        print(f"Tour cost recomputed: {tour_cost(dist, hk.tour):.6f}")


def main() -> None:
    run_case_euclidean()
    run_case_asymmetric()


if __name__ == "__main__":
    main()

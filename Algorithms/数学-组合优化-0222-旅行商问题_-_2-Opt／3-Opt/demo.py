"""TSP local-search MVP: nearest-neighbor + 2-Opt + 3-Opt.

Run:
    python3 demo.py
"""

from __future__ import annotations

import time
from typing import Iterable, List, Sequence, Tuple

import numpy as np


def build_cities(n_cities: int = 24, seed: int = 222) -> np.ndarray:
    """Generate 2D coordinates for TSP cities."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 100.0, size=(n_cities, 2))


def pairwise_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Compute Euclidean pairwise distances."""
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def tour_length(tour: Sequence[int], dist: np.ndarray) -> float:
    """Length of a Hamiltonian cycle represented by a permutation."""
    idx = np.asarray(tour, dtype=np.int64)
    nxt = np.roll(idx, -1)
    return float(dist[idx, nxt].sum())


def nearest_neighbor_tour(dist: np.ndarray, start: int = 0) -> List[int]:
    """Construct a deterministic initial tour by nearest-neighbor heuristic."""
    n = dist.shape[0]
    unvisited = set(range(n))
    unvisited.remove(start)

    tour = [start]
    current = start
    while unvisited:
        nxt = min(unvisited, key=lambda node: dist[current, node])
        unvisited.remove(nxt)
        tour.append(nxt)
        current = nxt
    return tour


def two_opt_pass(tour: Sequence[int], dist: np.ndarray) -> Tuple[List[int], float, bool]:
    """One best-improvement 2-opt pass."""
    n = len(tour)
    best_delta = 0.0
    best_move: Tuple[int, int] | None = None

    for i in range(n - 1):
        a = tour[i]
        b = tour[(i + 1) % n]

        # j+1 is the paired endpoint. Skip adjacent edges and the full wrap edge.
        j_start = i + 2
        j_end = n - 1 if i == 0 else n

        for j in range(j_start, j_end):
            c = tour[j]
            d = tour[(j + 1) % n]
            delta = dist[a, c] + dist[b, d] - dist[a, b] - dist[c, d]
            if delta < best_delta - 1e-12:
                best_delta = delta
                best_move = (i, j)

    if best_move is None:
        return list(tour), 0.0, False

    i, j = best_move
    improved = list(tour[: i + 1]) + list(reversed(tour[i + 1 : j + 1])) + list(tour[j + 1 :])
    return improved, best_delta, True


def two_opt_local_search(
    tour: Sequence[int], dist: np.ndarray, max_rounds: int = 200
) -> Tuple[List[int], float, int]:
    """Iterate 2-opt passes until local optimum."""
    current = list(tour)
    total_delta = 0.0
    rounds = 0

    for _ in range(max_rounds):
        current, delta, improved = two_opt_pass(current, dist)
        if not improved:
            break
        total_delta += delta
        rounds += 1
    return current, total_delta, rounds


def _three_opt_candidates(tour: Sequence[int], i: int, j: int, k: int) -> Iterable[List[int]]:
    """Return reconnection candidates for a 3-cut decomposition.

    Tour is split as A | B | C | D with cut points after i, j, k.
    """
    a = list(tour[: i + 1])
    b = list(tour[i + 1 : j + 1])
    c = list(tour[j + 1 : k + 1])
    d = list(tour[k + 1 :])

    # 7 non-trivial patterns: two are 2-opt-equivalent, five are genuine 3-opt style reconnects.
    yield a + b[::-1] + c + d
    yield a + b + c[::-1] + d
    yield a + c + b + d
    yield a + c[::-1] + b + d
    yield a + c + b[::-1] + d
    yield a + b[::-1] + c[::-1] + d
    yield a + c[::-1] + b[::-1] + d


def three_opt_pass(tour: Sequence[int], dist: np.ndarray) -> Tuple[List[int], float, bool]:
    """One best-improvement 3-opt pass using explicit reconnection enumeration."""
    n = len(tour)
    current_len = tour_length(tour, dist)
    best_len = current_len
    best_tour = list(tour)

    # Ensure three non-empty middle segments by keeping a one-node gap between cuts.
    for i in range(0, n - 5):
        for j in range(i + 2, n - 3):
            for k in range(j + 2, n - 1):
                for cand in _three_opt_candidates(tour, i, j, k):
                    if len(cand) != n or len(set(cand)) != n:
                        continue
                    cand_len = tour_length(cand, dist)
                    if cand_len < best_len - 1e-12:
                        best_len = cand_len
                        best_tour = cand

    if best_len < current_len - 1e-12:
        return best_tour, best_len - current_len, True
    return list(tour), 0.0, False


def three_opt_local_search(
    tour: Sequence[int], dist: np.ndarray, max_rounds: int = 20
) -> Tuple[List[int], float, int]:
    """Iterate 3-opt passes until local optimum or round cap."""
    current = list(tour)
    total_delta = 0.0
    rounds = 0

    for _ in range(max_rounds):
        current, delta, improved = three_opt_pass(current, dist)
        if not improved:
            break
        total_delta += delta
        rounds += 1
    return current, total_delta, rounds


def _validate_tour(tour: Sequence[int], n: int) -> None:
    if len(tour) != n:
        raise ValueError("Tour length mismatch.")
    if len(set(tour)) != n:
        raise ValueError("Tour is not a permutation of all cities.")


def main() -> None:
    n_cities = 24
    coords = build_cities(n_cities=n_cities, seed=222)
    dist = pairwise_distance_matrix(coords)

    t0 = time.perf_counter()
    initial_tour = nearest_neighbor_tour(dist, start=0)
    initial_len = tour_length(initial_tour, dist)
    t1 = time.perf_counter()

    after_2opt_tour, delta_2opt, rounds_2opt = two_opt_local_search(initial_tour, dist)
    after_2opt_len = tour_length(after_2opt_tour, dist)
    t2 = time.perf_counter()

    after_3opt_tour, delta_3opt, rounds_3opt = three_opt_local_search(after_2opt_tour, dist)
    after_3opt_len = tour_length(after_3opt_tour, dist)
    t3 = time.perf_counter()

    _validate_tour(initial_tour, n_cities)
    _validate_tour(after_2opt_tour, n_cities)
    _validate_tour(after_3opt_tour, n_cities)

    # Monotonic non-worsening guarantees for local search phases.
    assert after_2opt_len <= initial_len + 1e-9
    assert after_3opt_len <= after_2opt_len + 1e-9

    print("TSP (2-Opt/3-Opt) MVP")
    print(f"Cities: {n_cities}")
    print("=" * 66)
    print(f"Initial tour length      : {initial_len:10.3f}")
    print(f"After 2-Opt length       : {after_2opt_len:10.3f} (delta {delta_2opt:8.3f}, rounds={rounds_2opt})")
    print(f"After 3-Opt length       : {after_3opt_len:10.3f} (delta {delta_3opt:8.3f}, rounds={rounds_3opt})")
    print("-" * 66)
    print(f"2-Opt improvement (%)    : {(initial_len - after_2opt_len) / initial_len * 100:10.3f}")
    print(f"3-Opt extra improve (%)  : {(after_2opt_len - after_3opt_len) / after_2opt_len * 100:10.3f}")
    print(f"Total improvement (%)    : {(initial_len - after_3opt_len) / initial_len * 100:10.3f}")
    print("=" * 66)
    print(f"Init time (ms)           : {(t1 - t0) * 1000:10.3f}")
    print(f"2-Opt time (ms)          : {(t2 - t1) * 1000:10.3f}")
    print(f"3-Opt time (ms)          : {(t3 - t2) * 1000:10.3f}")
    print("=" * 66)
    print("Final tour prefix (first 12 nodes):", after_3opt_tour[:12])
    print("All checks passed.")


if __name__ == "__main__":
    main()

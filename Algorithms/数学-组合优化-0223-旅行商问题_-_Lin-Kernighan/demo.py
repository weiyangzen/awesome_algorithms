"""Lin-Kernighan style MVP for Euclidean TSP.

This script provides a deterministic, non-interactive demonstration of a
"LK-style variable-depth chained 2-opt" heuristic:
- nearest-neighbor initialization
- candidate lists for restricted edge exchanges
- variable-depth gain-positive move chaining

It is intentionally compact and auditable, and does not call a black-box TSP solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Set, Tuple

import numpy as np


HistoryItem = Tuple[int, float, float, float]


@dataclass
class TSPSolution:
    tour: np.ndarray
    length: float
    history: List[HistoryItem]


def euclidean_distance_matrix(coords: np.ndarray) -> np.ndarray:
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must have shape (n, 2), got {coords.shape}")
    deltas = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(deltas * deltas, axis=2))


def validate_distance_matrix(dist: np.ndarray) -> None:
    if dist.ndim != 2 or dist.shape[0] != dist.shape[1]:
        raise ValueError(f"dist must be square, got shape={dist.shape}")
    if not np.all(np.isfinite(dist)):
        raise ValueError("dist contains non-finite values")
    if np.any(dist < 0):
        raise ValueError("dist contains negative values")
    if not np.allclose(dist, dist.T, atol=1e-10):
        raise ValueError("dist must be symmetric for this MVP")
    if not np.allclose(np.diag(dist), 0.0, atol=1e-10):
        raise ValueError("dist diagonal must be zero")


def validate_tour(tour: np.ndarray, n_cities: int) -> None:
    if tour.ndim != 1 or tour.size != n_cities:
        raise ValueError(f"tour must be 1D of size {n_cities}, got shape={tour.shape}")
    if not np.array_equal(np.sort(tour), np.arange(n_cities)):
        raise ValueError("tour is not a valid permutation of city ids")


def tour_length(tour: np.ndarray, dist: np.ndarray) -> float:
    nxt = np.roll(tour, -1)
    return float(np.sum(dist[tour, nxt]))


def nearest_neighbor_tour(dist: np.ndarray, start: int = 0) -> np.ndarray:
    n = dist.shape[0]
    if not (0 <= start < n):
        raise ValueError(f"start={start} out of range for n={n}")

    visited = np.zeros(n, dtype=bool)
    tour = np.empty(n, dtype=int)
    current = start

    for i in range(n):
        tour[i] = current
        visited[current] = True
        if i == n - 1:
            break

        row = dist[current].copy()
        row[visited] = np.inf
        current = int(np.argmin(row))

    return tour


def build_candidate_lists(dist: np.ndarray, candidate_k: int) -> np.ndarray:
    n = dist.shape[0]
    k = max(2, min(candidate_k, n - 1))
    order = np.argsort(dist, axis=1)
    return order[:, 1 : k + 1]


def canonical_pair(i: int, j: int) -> Tuple[int, int]:
    return (i, j) if i < j else (j, i)


def is_valid_two_opt_indices(i: int, j: int, n: int) -> bool:
    if i == j:
        return False
    if j == i + 1:
        return False
    if i == 0 and j == n - 1:
        return False
    return True


def two_opt_gain(tour: np.ndarray, i: int, j: int, dist: np.ndarray) -> float:
    n = tour.size
    i, j = canonical_pair(i, j)
    if not is_valid_two_opt_indices(i, j, n):
        return -np.inf

    a = tour[i]
    b = tour[(i + 1) % n]
    c = tour[j]
    d = tour[(j + 1) % n]

    removed = dist[a, b] + dist[c, d]
    added = dist[a, c] + dist[b, d]
    return float(removed - added)


def apply_two_opt_inplace(tour: np.ndarray, i: int, j: int) -> None:
    i, j = canonical_pair(i, j)
    tour[i + 1 : j + 1] = tour[i + 1 : j + 1][::-1]


def find_best_candidate_two_opt(
    tour: np.ndarray,
    dist: np.ndarray,
    candidate_lists: np.ndarray,
    active_nodes: Optional[Set[int]] = None,
) -> Tuple[float, Optional[Tuple[int, int]], Optional[Set[int]]]:
    n = tour.size
    pos = np.empty(n, dtype=int)
    pos[tour] = np.arange(n)

    best_gain = 0.0
    best_move: Optional[Tuple[int, int]] = None
    best_touched: Optional[Set[int]] = None

    for i in range(n):
        a = int(tour[i])
        b = int(tour[(i + 1) % n])

        if active_nodes is not None and a not in active_nodes and b not in active_nodes:
            continue

        for c in candidate_lists[a]:
            c = int(c)
            j0 = int(pos[c])
            for j in (j0, (j0 - 1) % n):
                ii, jj = canonical_pair(i, j)
                if not is_valid_two_opt_indices(ii, jj, n):
                    continue

                gain = two_opt_gain(tour=tour, i=ii, j=jj, dist=dist)
                if gain > best_gain + 1e-12:
                    x1 = int(tour[ii])
                    x2 = int(tour[(ii + 1) % n])
                    y1 = int(tour[jj])
                    y2 = int(tour[(jj + 1) % n])
                    best_gain = gain
                    best_move = (ii, jj)
                    best_touched = {x1, x2, y1, y2}

    return best_gain, best_move, best_touched


def lk_style_chain_pass(
    tour: np.ndarray,
    dist: np.ndarray,
    candidate_lists: np.ndarray,
    max_depth: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float]:
    n = tour.size
    anchors = rng.permutation(n)

    best_gain_global = 0.0
    best_tour_global = tour.copy()

    for anchor in anchors:
        work = tour.copy()
        cumulative_gain = 0.0
        best_gain_local = 0.0
        best_tour_local = work.copy()

        active_nodes: Set[int] = {int(work[anchor]), int(work[(anchor + 1) % n])}

        for _ in range(max_depth):
            gain, move, touched = find_best_candidate_two_opt(
                tour=work,
                dist=dist,
                candidate_lists=candidate_lists,
                active_nodes=active_nodes,
            )
            if move is None or gain <= 1e-12:
                break

            i, j = move
            apply_two_opt_inplace(work, i, j)
            cumulative_gain += gain

            if cumulative_gain > best_gain_local + 1e-12:
                best_gain_local = cumulative_gain
                best_tour_local = work.copy()

            if cumulative_gain <= 1e-12:
                break

            active_nodes = touched if touched is not None else active_nodes

        if best_gain_local > best_gain_global + 1e-12:
            best_gain_global = best_gain_local
            best_tour_global = best_tour_local

    return best_tour_global, best_gain_global


def lin_kernighan_style_tsp(
    dist: np.ndarray,
    initial_tour: np.ndarray,
    candidate_k: int = 20,
    max_depth: int = 5,
    max_iters: int = 80,
    seed: int = 2026,
) -> TSPSolution:
    validate_distance_matrix(dist)
    validate_tour(initial_tour, n_cities=dist.shape[0])
    if candidate_k <= 0:
        raise ValueError("candidate_k must be > 0")
    if max_depth <= 0:
        raise ValueError("max_depth must be > 0")
    if max_iters <= 0:
        raise ValueError("max_iters must be > 0")

    rng = np.random.default_rng(seed)
    candidate_lists = build_candidate_lists(dist=dist, candidate_k=candidate_k)

    tour = initial_tour.copy()
    length = tour_length(tour, dist)
    history: List[HistoryItem] = []
    stale_rounds = 0

    for it in range(1, max_iters + 1):
        old_length = length
        new_tour, estimated_gain = lk_style_chain_pass(
            tour=tour,
            dist=dist,
            candidate_lists=candidate_lists,
            max_depth=max_depth,
            rng=rng,
        )

        if estimated_gain > 1e-12:
            new_length = tour_length(new_tour, dist)
            realized_gain = old_length - new_length
            if realized_gain > 1e-12:
                tour = new_tour
                length = new_length
                stale_rounds = 0
                history.append((it, old_length, length, realized_gain))
            else:
                stale_rounds += 1
                history.append((it, old_length, old_length, 0.0))
        else:
            stale_rounds += 1
            history.append((it, old_length, old_length, 0.0))

        if stale_rounds >= 4:
            break

    return TSPSolution(tour=tour, length=length, history=history)


def make_euclidean_instance(n_cities: int = 120, seed: int = 7) -> np.ndarray:
    if n_cities < 8:
        raise ValueError("n_cities must be >= 8 for meaningful LK-style search")
    rng = np.random.default_rng(seed)

    # Two-cluster layout makes nearest-neighbor initialization non-trivial.
    centers = np.array([[0.25, 0.30], [0.72, 0.68]], dtype=float)
    labels = rng.integers(low=0, high=2, size=n_cities)
    coords = centers[labels] + 0.12 * rng.normal(size=(n_cities, 2))
    coords = np.clip(coords, 0.0, 1.0)
    return coords


def print_history(history: Sequence[HistoryItem], max_rows: int = 12) -> None:
    print("iter | old_length   | new_length   | gain")
    print("---------------------------------------------")

    show = min(max_rows, len(history))
    for i in range(show):
        it, old_len, new_len, gain = history[i]
        print(f"{it:4d} | {old_len:11.6f} | {new_len:11.6f} | {gain:9.6f}")

    if len(history) > max_rows:
        omitted = len(history) - max_rows
        it, old_len, new_len, gain = history[-1]
        print(f"... ({omitted} rows omitted)")
        print(f"{it:4d} | {old_len:11.6f} | {new_len:11.6f} | {gain:9.6f}  (last)")


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)

    n_cities = 120
    coord_seed = 7
    solve_seed = 2026

    coords = make_euclidean_instance(n_cities=n_cities, seed=coord_seed)
    dist = euclidean_distance_matrix(coords)

    init_tour = nearest_neighbor_tour(dist=dist, start=0)
    init_length = tour_length(init_tour, dist)

    result = lin_kernighan_style_tsp(
        dist=dist,
        initial_tour=init_tour,
        candidate_k=24,
        max_depth=6,
        max_iters=80,
        seed=solve_seed,
    )

    improvement = init_length - result.length
    improve_ratio = 100.0 * improvement / init_length

    print("=== TSP | Lin-Kernighan style MVP ===")
    print(f"cities: {n_cities}")
    print(f"instance seed: {coord_seed}")
    print(f"solver seed: {solve_seed}")
    print(f"initial length: {init_length:.6f}")
    print(f"final length:   {result.length:.6f}")
    print(f"improvement:    {improvement:.6f} ({improve_ratio:.2f}%)")
    print(f"iterations:     {len(result.history)}")
    print()

    print_history(result.history, max_rows=12)
    print()

    preview = np.concatenate([result.tour[:15], result.tour[:1]])
    print("tour prefix (return-to-start shown at end):")
    print(np.array2string(preview, separator=", "))


if __name__ == "__main__":
    main()

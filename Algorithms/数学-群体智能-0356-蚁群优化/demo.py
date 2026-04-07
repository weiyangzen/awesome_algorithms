"""Ant Colony Optimization (ACO) minimal runnable MVP.

This demo solves a symmetric Euclidean Traveling Salesman Problem (TSP)
instance using the Ant System style workflow:
1) probabilistic tour construction,
2) pheromone evaporation,
3) pheromone deposition,
4) iterative best-solution tracking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ACOConfig:
    """Hyperparameters for Ant Colony Optimization."""

    alpha: float = 1.1
    beta: float = 4.2
    evaporation_rate: float = 0.45
    q: float = 80.0
    n_ants: int = 30
    n_iterations: int = 160
    elite_weight: float = 2.5
    seed: int = 123


Tour = List[int]


def euclidean_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Build pairwise Euclidean distances; diagonal is inf to forbid self-loops."""
    deltas = coords[:, None, :] - coords[None, :, :]
    distance = np.linalg.norm(deltas, axis=2)
    np.fill_diagonal(distance, np.inf)
    return distance


def cycle_length(tour: Sequence[int], distance: np.ndarray) -> float:
    route = np.asarray(tour, dtype=int)
    nxt = np.roll(route, -1)
    return float(distance[route, nxt].sum())


def nearest_neighbor_tour(distance: np.ndarray, start: int) -> Tour:
    """Simple constructive baseline."""
    n = distance.shape[0]
    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    tour = [start]

    for _ in range(n - 1):
        cur = tour[-1]
        candidates = np.where(~visited)[0]
        nxt = int(candidates[np.argmin(distance[cur, candidates])])
        visited[nxt] = True
        tour.append(nxt)

    return tour


def construct_ant_tour(
    rng: np.random.Generator,
    pheromone: np.ndarray,
    heuristic: np.ndarray,
    config: ACOConfig,
    start: int,
) -> Tour:
    """Construct one ant's route using roulette-wheel transition probabilities."""
    n = pheromone.shape[0]
    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    tour = [start]

    for _ in range(n - 1):
        cur = tour[-1]
        candidates = np.where(~visited)[0]

        tau = np.power(pheromone[cur, candidates], config.alpha)
        eta = np.power(heuristic[cur, candidates], config.beta)
        weight = tau * eta

        total = float(weight.sum())
        if total <= 0.0 or not np.isfinite(total):
            probs = np.full(candidates.shape[0], 1.0 / candidates.shape[0], dtype=float)
        else:
            probs = weight / total

        nxt = int(rng.choice(candidates, p=probs))
        visited[nxt] = True
        tour.append(nxt)

    return tour


def deposit_pheromone(pheromone: np.ndarray, tour: Sequence[int], amount: float) -> None:
    """Deposit pheromone on an undirected Hamiltonian cycle."""
    route = np.asarray(tour, dtype=int)
    nxt = np.roll(route, -1)
    for u, v in zip(route, nxt):
        pheromone[u, v] += amount
        pheromone[v, u] += amount


def run_aco_tsp(coords: np.ndarray, config: ACOConfig) -> Tuple[Tour, float, List[float], np.ndarray]:
    """Run ACO on Euclidean TSP points."""
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (n, 2)")

    n_cities = coords.shape[0]
    if n_cities < 4:
        raise ValueError("Need at least 4 cities for a meaningful TSP instance.")

    rng = np.random.default_rng(config.seed)
    distance = euclidean_distance_matrix(coords)

    heuristic = 1.0 / np.where(np.isfinite(distance), distance, 1.0)
    np.fill_diagonal(heuristic, 0.0)

    nn_lengths = [cycle_length(nearest_neighbor_tour(distance, s), distance) for s in range(n_cities)]
    tau0 = 1.0 / (n_cities * min(nn_lengths))
    pheromone = np.full((n_cities, n_cities), tau0, dtype=float)
    np.fill_diagonal(pheromone, 0.0)

    best_tour: Tour | None = None
    best_length = np.inf
    history: List[float] = []

    for _ in range(config.n_iterations):
        iteration_solutions: List[Tuple[Tour, float]] = []

        for _ant in range(config.n_ants):
            start = int(rng.integers(0, n_cities))
            tour = construct_ant_tour(rng, pheromone, heuristic, config, start)
            length = cycle_length(tour, distance)
            iteration_solutions.append((tour, length))

            if length < best_length:
                best_length = length
                best_tour = list(tour)

        pheromone *= 1.0 - config.evaporation_rate

        for tour, length in iteration_solutions:
            deposit_pheromone(pheromone, tour, config.q / length)

        if best_tour is not None and config.elite_weight > 0.0:
            deposit_pheromone(pheromone, best_tour, config.elite_weight * config.q / best_length)

        np.fill_diagonal(pheromone, 0.0)
        history.append(best_length)

    if best_tour is None:
        raise RuntimeError("ACO failed to produce a valid tour.")

    return best_tour, best_length, history, distance


def random_baseline(distance: np.ndarray, n_samples: int = 1200, seed: int = 99) -> float:
    """Best tour among random permutations, used only as a weak baseline."""
    rng = np.random.default_rng(seed)
    n = distance.shape[0]
    best = np.inf

    for _ in range(n_samples):
        tour = rng.permutation(n)
        length = float(distance[tour, np.roll(tour, -1)].sum())
        if length < best:
            best = length

    return best


def validate_tour(tour: Sequence[int], n_cities: int) -> bool:
    return len(tour) == n_cities and sorted(tour) == list(range(n_cities))


def main() -> None:
    rng = np.random.default_rng(2026)
    coords = rng.uniform(0.0, 100.0, size=(22, 2))

    config = ACOConfig()
    best_tour, best_length, history, distance = run_aco_tsp(coords, config)

    nn_best = min(cycle_length(nearest_neighbor_tour(distance, s), distance) for s in range(coords.shape[0]))
    random_best = random_baseline(distance, n_samples=1200, seed=99)

    assert validate_tour(best_tour, coords.shape[0]), "Invalid tour: city coverage check failed."
    assert all(history[i] <= history[i - 1] + 1e-12 for i in range(1, len(history))), "Best history must be non-increasing."

    print("Ant Colony Optimization (TSP) MVP")
    print(f"cities={coords.shape[0]}, ants={config.n_ants}, iterations={config.n_iterations}")
    print(f"nearest_neighbor_best={nn_best:.3f}")
    print(f"random_baseline_best={random_best:.3f}")
    print(f"aco_best={best_length:.3f}")

    gap_vs_nn = (best_length - nn_best) / nn_best * 100.0
    gap_vs_random = (best_length - random_best) / random_best * 100.0
    print(f"gap_vs_nearest_neighbor={gap_vs_nn:+.2f}%")
    print(f"gap_vs_random_baseline={gap_vs_random:+.2f}%")
    print("best_tour_prefix=", best_tour[:10], "...")
    print("history_tail=", [round(x, 3) for x in history[-5:]])


if __name__ == "__main__":
    main()

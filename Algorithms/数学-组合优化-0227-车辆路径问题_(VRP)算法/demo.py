"""CVRP MVP using Clarke-Wright Savings + 2-opt local search.

This script is self-contained and deterministic. It builds a synthetic CVRP
instance, constructs routes with a transparent savings heuristic, improves each
route by 2-opt, validates feasibility, and prints a compact summary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Route = List[int]


@dataclass
class CVRPInstance:
    coords: np.ndarray  # shape: (n_customers + 1, 2), index 0 is depot
    demands: np.ndarray  # shape: (n_customers + 1,), demands[0] == 0
    capacity: int
    max_vehicles: Optional[int] = None

    @property
    def n_customers(self) -> int:
        return int(self.demands.shape[0] - 1)


@dataclass
class CVRPSolution:
    routes: List[Route]
    total_distance: float


def euclidean_distance_matrix(coords: np.ndarray) -> np.ndarray:
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must be shape (n, 2), got {coords.shape}")
    if not np.all(np.isfinite(coords)):
        raise ValueError("coords contains non-finite values")

    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    return dist.astype(float)


def route_load(route: Route, demands: np.ndarray) -> int:
    # route format: [0, ..., 0]
    return int(np.sum(demands[np.array(route[1:-1], dtype=int)]))


def route_distance(route: Route, dist: np.ndarray) -> float:
    idx_from = np.array(route[:-1], dtype=int)
    idx_to = np.array(route[1:], dtype=int)
    return float(np.sum(dist[idx_from, idx_to]))


def total_distance(routes: Sequence[Route], dist: np.ndarray) -> float:
    return float(sum(route_distance(r, dist) for r in routes))


def validate_solution(instance: CVRPInstance, routes: Sequence[Route], dist: np.ndarray) -> None:
    n = instance.n_customers

    seen: List[int] = []
    for route in routes:
        if len(route) < 3:
            raise ValueError(f"invalid route length: {route}")
        if route[0] != 0 or route[-1] != 0:
            raise ValueError(f"route must start and end at depot 0: {route}")

        for node in route:
            if node < 0 or node > n:
                raise ValueError(f"node out of range: {node}")

        load = route_load(route, instance.demands)
        if load > instance.capacity:
            raise ValueError(f"capacity violated: load={load}, cap={instance.capacity}, route={route}")

        seen.extend(route[1:-1])

        rd = route_distance(route, dist)
        if not np.isfinite(rd):
            raise ValueError("route distance is non-finite")

    seen_sorted = sorted(seen)
    expected = list(range(1, n + 1))
    if seen_sorted != expected:
        raise ValueError(
            "customer coverage violation: "
            f"seen={seen_sorted[:10]}{'...' if len(seen_sorted) > 10 else ''}, expected=1..{n}"
        )

    if instance.max_vehicles is not None and len(routes) > instance.max_vehicles:
        raise ValueError(
            f"vehicle count violated: used={len(routes)}, limit={instance.max_vehicles}"
        )


def try_merge(route_i: List[int], route_j: List[int], i: int, j: int) -> Optional[List[int]]:
    """Try to merge two customer sequences according to endpoint compatibility.

    route_i / route_j do not contain depot.
    """
    i_at_start = route_i[0] == i
    i_at_end = route_i[-1] == i
    j_at_start = route_j[0] == j
    j_at_end = route_j[-1] == j

    if not (i_at_start or i_at_end):
        return None
    if not (j_at_start or j_at_end):
        return None

    if i_at_end and j_at_start:
        return route_i + route_j
    if i_at_start and j_at_end:
        return route_j + route_i
    if i_at_start and j_at_start:
        return list(reversed(route_i)) + route_j
    if i_at_end and j_at_end:
        return route_i + list(reversed(route_j))

    return None


def clarke_wright_savings(instance: CVRPInstance, dist: np.ndarray) -> CVRPSolution:
    n = instance.n_customers
    if n <= 0:
        raise ValueError("instance must contain at least one customer")

    routes: Dict[int, List[int]] = {i: [i] for i in range(1, n + 1)}
    loads: Dict[int, int] = {i: int(instance.demands[i]) for i in range(1, n + 1)}
    customer_to_route: Dict[int, int] = {i: i for i in range(1, n + 1)}
    next_route_id = n + 1

    savings: List[Tuple[float, int, int]] = []
    for i in range(1, n):
        for j in range(i + 1, n + 1):
            s = float(dist[0, i] + dist[0, j] - dist[i, j])
            savings.append((s, i, j))

    savings.sort(key=lambda x: x[0], reverse=True)

    for _, i, j in savings:
        ri = customer_to_route.get(i)
        rj = customer_to_route.get(j)

        if ri is None or rj is None or ri == rj:
            continue

        seq_i = routes[ri]
        seq_j = routes[rj]

        if loads[ri] + loads[rj] > instance.capacity:
            continue

        merged = try_merge(seq_i, seq_j, i, j)
        if merged is None:
            continue

        new_id = next_route_id
        next_route_id += 1

        new_load = loads[ri] + loads[rj]
        routes[new_id] = merged
        loads[new_id] = new_load
        for c in merged:
            customer_to_route[c] = new_id

        del routes[ri]
        del routes[rj]
        del loads[ri]
        del loads[rj]

    final_routes = [[0] + seq + [0] for seq in routes.values()]
    final_routes.sort(key=lambda r: tuple(r))

    if instance.max_vehicles is not None and len(final_routes) > instance.max_vehicles:
        raise ValueError(
            f"infeasible under current capacity/limit: got {len(final_routes)} routes, "
            f"limit={instance.max_vehicles}"
        )

    return CVRPSolution(routes=final_routes, total_distance=total_distance(final_routes, dist))


def two_opt_route(route: Route, dist: np.ndarray) -> Route:
    if len(route) <= 4:
        return route[:]

    best = route[:]
    improved = True

    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for k in range(i + 1, len(best) - 1):
                a, b = best[i - 1], best[i]
                c, d = best[k], best[k + 1]
                delta = (dist[a, c] + dist[b, d]) - (dist[a, b] + dist[c, d])
                if delta < -1e-12:
                    best = best[:i] + list(reversed(best[i : k + 1])) + best[k + 1 :]
                    improved = True
                    break
            if improved:
                break

    return best


def improve_with_two_opt(routes: Sequence[Route], dist: np.ndarray) -> CVRPSolution:
    improved_routes = [two_opt_route(r, dist) for r in routes]
    return CVRPSolution(
        routes=improved_routes,
        total_distance=total_distance(improved_routes, dist),
    )


def trivial_single_customer_routes(instance: CVRPInstance, dist: np.ndarray) -> CVRPSolution:
    routes = [[0, i, 0] for i in range(1, instance.n_customers + 1)]
    return CVRPSolution(routes=routes, total_distance=total_distance(routes, dist))


def generate_demo_instance(
    seed: int = 17,
    n_customers: int = 20,
    area_size: float = 100.0,
    demand_low: int = 1,
    demand_high: int = 9,
    target_vehicles: int = 5,
    max_vehicles: int = 6,
) -> CVRPInstance:
    if n_customers <= 0:
        raise ValueError("n_customers must be > 0")

    rng = np.random.default_rng(seed)

    depot = np.array([[area_size * 0.5, area_size * 0.5]], dtype=float)
    customers = rng.uniform(0.0, area_size, size=(n_customers, 2)).astype(float)
    coords = np.vstack([depot, customers])

    demands = np.zeros(n_customers + 1, dtype=int)
    demands[1:] = rng.integers(demand_low, demand_high + 1, size=n_customers)

    total_dem = int(np.sum(demands[1:]))

    # Pick a capacity that tends to need around target_vehicles routes.
    capacity = int(np.ceil(total_dem / float(target_vehicles)))
    capacity = max(capacity, int(np.max(demands[1:])))

    # Ensure fleet-limit feasibility by construction.
    if capacity * max_vehicles < total_dem:
        capacity = int(np.ceil(total_dem / float(max_vehicles)))

    return CVRPInstance(
        coords=coords,
        demands=demands,
        capacity=capacity,
        max_vehicles=max_vehicles,
    )


def print_solution(label: str, sol: CVRPSolution, instance: CVRPInstance, dist: np.ndarray) -> None:
    print(f"\n[{label}]")
    print(f"routes: {len(sol.routes)}")
    print(f"total distance: {sol.total_distance:.3f}")
    for rid, route in enumerate(sol.routes, start=1):
        load = route_load(route, instance.demands)
        rdist = route_distance(route, dist)
        print(f"  route {rid:02d}: load={load:3d}, dist={rdist:8.3f}, path={route}")


def main() -> None:
    instance = generate_demo_instance()
    dist = euclidean_distance_matrix(instance.coords)

    print("=== CVRP Demo: Clarke-Wright Savings + 2-opt ===")
    print(f"customers: {instance.n_customers}")
    print(f"total demand: {int(np.sum(instance.demands[1:]))}")
    print(f"vehicle capacity: {instance.capacity}")
    print(f"vehicle limit: {instance.max_vehicles}")

    baseline = trivial_single_customer_routes(instance, dist)
    savings_sol = clarke_wright_savings(instance, dist)
    improved_sol = improve_with_two_opt(savings_sol.routes, dist)

    validate_solution(instance, savings_sol.routes, dist)
    validate_solution(instance, improved_sol.routes, dist)

    print_solution("Baseline (single-customer routes)", baseline, instance, dist)
    print_solution("After Clarke-Wright savings", savings_sol, instance, dist)
    print_solution("After per-route 2-opt", improved_sol, instance, dist)

    gain_vs_baseline = 100.0 * (baseline.total_distance - improved_sol.total_distance) / baseline.total_distance
    gain_vs_savings = 100.0 * (savings_sol.total_distance - improved_sol.total_distance) / max(
        savings_sol.total_distance,
        1e-12,
    )

    print("\n=== Summary ===")
    print(f"distance reduction vs baseline: {gain_vs_baseline:.2f}%")
    print(f"distance reduction from 2-opt stage: {gain_vs_savings:.2f}%")
    print("feasibility checks: PASS")


if __name__ == "__main__":
    main()

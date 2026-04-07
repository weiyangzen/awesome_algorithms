"""Minimal runnable MVP for uncapacitated facility location (UFL).

The script implements a transparent algorithmic pipeline instead of using
black-box solvers:
1) deterministic synthetic instance generation,
2) greedy add-only initialization,
3) local search with add/drop/swap moves,
4) exact brute-force enumeration for a quality reference on small facility sets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

Array = np.ndarray


@dataclass
class FacilityLocationInstance:
    facility_xy: Array  # (m, 2)
    customer_xy: Array  # (n, 2)
    demand: Array  # (n,)
    open_cost: Array  # (m,)
    service_cost: Array  # (m, n), already multiplied by demand


@dataclass
class FacilityLocationSolution:
    open_mask: Array  # bool, shape (m,)
    assignment: Array  # int, shape (n,)
    opening_cost: float
    shipping_cost: float
    total_cost: float


@dataclass
class MoveRecord:
    iteration: int
    move_type: str
    detail: str
    delta: float
    total_cost: float


def ensure_2d(name: str, x: Array) -> None:
    if x.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape={x.shape}.")
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty, got shape={x.shape}.")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contains non-finite values.")


def ensure_1d(name: str, x: Array) -> None:
    if x.ndim != 1 or x.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D vector, got shape={x.shape}.")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contains non-finite values.")


def pairwise_euclidean(a: Array, b: Array) -> Array:
    """Return matrix D where D[i, j] = ||a_i - b_j||_2."""
    ensure_2d("a", a)
    ensure_2d("b", b)
    if a.shape[1] != b.shape[1]:
        raise ValueError("a and b must have the same feature dimension.")

    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def build_service_cost(facility_xy: Array, customer_xy: Array, demand: Array) -> Array:
    """Build shipping matrix C[i,j] = demand[j] * distance(i,j)."""
    ensure_2d("facility_xy", facility_xy)
    ensure_2d("customer_xy", customer_xy)
    ensure_1d("demand", demand)

    n_customers = customer_xy.shape[0]
    if demand.size != n_customers:
        raise ValueError("demand length must equal number of customers.")
    if np.any(demand <= 0):
        raise ValueError("all demand values must be positive.")

    distance = pairwise_euclidean(facility_xy, customer_xy)
    return distance * demand[None, :]


def evaluate_solution(open_mask: Array, instance: FacilityLocationInstance) -> FacilityLocationSolution:
    """Compute assignment and objective for a chosen open facility set."""
    if open_mask.dtype != np.bool_:
        open_mask = open_mask.astype(bool)
    if open_mask.ndim != 1 or open_mask.size != instance.open_cost.size:
        raise ValueError("open_mask shape mismatch with facility count.")

    open_idx = np.flatnonzero(open_mask)
    if open_idx.size == 0:
        raise ValueError("at least one facility must be open.")

    shipping_slice = instance.service_cost[open_idx, :]  # (k, n)
    best_local = np.argmin(shipping_slice, axis=0)
    assignment = open_idx[best_local]

    shipping_cost = float(np.sum(shipping_slice[best_local, np.arange(shipping_slice.shape[1])]))
    opening_cost = float(np.sum(instance.open_cost[open_mask]))
    total_cost = opening_cost + shipping_cost

    return FacilityLocationSolution(
        open_mask=open_mask.copy(),
        assignment=assignment,
        opening_cost=opening_cost,
        shipping_cost=shipping_cost,
        total_cost=total_cost,
    )


def greedy_add_initialization(instance: FacilityLocationInstance) -> Tuple[FacilityLocationSolution, List[MoveRecord]]:
    """Start from the best single facility, then keep best improving add moves."""
    m = instance.open_cost.size
    if m == 0:
        raise ValueError("instance has no facilities.")

    best_single: FacilityLocationSolution | None = None
    for i in range(m):
        mask = np.zeros(m, dtype=bool)
        mask[i] = True
        candidate = evaluate_solution(mask, instance)
        if best_single is None or candidate.total_cost < best_single.total_cost:
            best_single = candidate

    assert best_single is not None
    current = best_single
    history: List[MoveRecord] = []

    for it in range(1, m + 1):
        current_open = current.open_mask
        closed_idx = np.flatnonzero(~current_open)

        best_candidate = current
        best_added = -1
        best_delta = 0.0

        for add_idx in closed_idx:
            new_mask = current_open.copy()
            new_mask[add_idx] = True
            candidate = evaluate_solution(new_mask, instance)
            delta = candidate.total_cost - current.total_cost
            if delta < best_delta - 1e-12:
                best_delta = delta
                best_candidate = candidate
                best_added = int(add_idx)

        if best_added < 0:
            break

        current = best_candidate
        history.append(
            MoveRecord(
                iteration=it,
                move_type="add",
                detail=f"open facility {best_added}",
                delta=best_delta,
                total_cost=current.total_cost,
            )
        )

    return current, history


def local_search(
    instance: FacilityLocationInstance,
    start: FacilityLocationSolution,
    max_iter: int = 80,
) -> Tuple[FacilityLocationSolution, List[MoveRecord]]:
    """Best-improvement local search over add/drop/swap neighborhood."""
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")

    current = start
    m = instance.open_cost.size
    history: List[MoveRecord] = []

    for it in range(1, max_iter + 1):
        open_idx = np.flatnonzero(current.open_mask)
        closed_idx = np.flatnonzero(~current.open_mask)

        best_candidate: FacilityLocationSolution | None = None
        best_move_type = ""
        best_detail = ""
        best_delta = 0.0

        # Add moves: open one currently closed facility.
        for add_idx in closed_idx:
            new_mask = current.open_mask.copy()
            new_mask[add_idx] = True
            candidate = evaluate_solution(new_mask, instance)
            delta = candidate.total_cost - current.total_cost
            if delta < best_delta - 1e-12:
                best_candidate = candidate
                best_move_type = "add"
                best_detail = f"open {int(add_idx)}"
                best_delta = delta

        # Drop moves: close one open facility (while keeping at least one open).
        if open_idx.size > 1:
            for drop_idx in open_idx:
                new_mask = current.open_mask.copy()
                new_mask[drop_idx] = False
                candidate = evaluate_solution(new_mask, instance)
                delta = candidate.total_cost - current.total_cost
                if delta < best_delta - 1e-12:
                    best_candidate = candidate
                    best_move_type = "drop"
                    best_detail = f"close {int(drop_idx)}"
                    best_delta = delta

        # Swap moves: close one open and open one closed facility.
        for drop_idx in open_idx:
            for add_idx in closed_idx:
                new_mask = current.open_mask.copy()
                new_mask[drop_idx] = False
                new_mask[add_idx] = True
                candidate = evaluate_solution(new_mask, instance)
                delta = candidate.total_cost - current.total_cost
                if delta < best_delta - 1e-12:
                    best_candidate = candidate
                    best_move_type = "swap"
                    best_detail = f"close {int(drop_idx)}, open {int(add_idx)}"
                    best_delta = delta

        if best_candidate is None:
            break

        current = best_candidate
        history.append(
            MoveRecord(
                iteration=it,
                move_type=best_move_type,
                detail=best_detail,
                delta=best_delta,
                total_cost=current.total_cost,
            )
        )

    return current, history


def brute_force_optimal(instance: FacilityLocationInstance, max_facilities: int = 20) -> FacilityLocationSolution:
    """Exact solver by enumerating all non-empty subsets (small m only)."""
    m = instance.open_cost.size
    if m == 0:
        raise ValueError("instance has no facilities.")
    if m > max_facilities:
        raise ValueError(
            f"brute-force disabled for m={m}; increase max_facilities or use a MIP solver."
        )

    best: FacilityLocationSolution | None = None
    ids = np.arange(m)

    for subset_bits in range(1, 1 << m):
        mask = ((subset_bits >> ids) & 1).astype(bool)
        candidate = evaluate_solution(mask, instance)
        if best is None or candidate.total_cost < best.total_cost:
            best = candidate

    assert best is not None
    return best


def generate_instance(seed: int, n_facilities: int, n_customers: int) -> FacilityLocationInstance:
    """Deterministically generate a clustered spatial UFL instance."""
    if n_facilities <= 1:
        raise ValueError("n_facilities must be >= 2.")
    if n_customers <= 0:
        raise ValueError("n_customers must be positive.")

    rng = np.random.default_rng(seed)

    facility_xy = rng.uniform(0.0, 100.0, size=(n_facilities, 2))

    center_count = 3
    centers = rng.uniform(15.0, 85.0, size=(center_count, 2))
    cluster_pick = rng.integers(0, center_count, size=n_customers)
    customer_xy = centers[cluster_pick] + rng.normal(0.0, 10.0, size=(n_customers, 2))
    customer_xy = np.clip(customer_xy, 0.0, 100.0)

    demand = rng.integers(1, 8, size=n_customers).astype(float)

    mean_distance = np.mean(pairwise_euclidean(facility_xy, customer_xy), axis=1)
    open_cost = 28.0 + 0.35 * mean_distance + rng.uniform(0.0, 10.0, size=n_facilities)

    service_cost = build_service_cost(facility_xy, customer_xy, demand)
    return FacilityLocationInstance(
        facility_xy=facility_xy,
        customer_xy=customer_xy,
        demand=demand,
        open_cost=open_cost,
        service_cost=service_cost,
    )


def summarize_moves(moves: Sequence[MoveRecord], limit_head: int = 8, limit_tail: int = 3) -> None:
    print("iter | move  | detail                  | delta        | total")
    print("-----+-------+-------------------------+--------------+--------------")

    if not moves:
        print("(no improving move)")
        return

    head = list(moves[:limit_head])
    tail = list(moves[-limit_tail:]) if len(moves) > (limit_head + limit_tail) else []

    for rec in head:
        print(
            f"{rec.iteration:4d} | {rec.move_type:<5} | {rec.detail:<23} "
            f"| {rec.delta: .6e} | {rec.total_cost: .6e}"
        )

    if tail:
        print("  ...")
        for rec in tail:
            print(
                f"{rec.iteration:4d} | {rec.move_type:<5} | {rec.detail:<23} "
                f"| {rec.delta: .6e} | {rec.total_cost: .6e}"
            )


def run_case(case_name: str, seed: int, n_facilities: int, n_customers: int) -> Tuple[float, float]:
    print(f"\n=== Case: {case_name} ===")
    instance = generate_instance(seed=seed, n_facilities=n_facilities, n_customers=n_customers)

    init_solution, init_moves = greedy_add_initialization(instance)
    final_solution, ls_moves = local_search(instance, init_solution, max_iter=80)
    exact_solution = brute_force_optimal(instance, max_facilities=20)

    open_count_init = int(np.sum(init_solution.open_mask))
    open_count_final = int(np.sum(final_solution.open_mask))
    open_count_exact = int(np.sum(exact_solution.open_mask))

    gap = 100.0 * (final_solution.total_cost - exact_solution.total_cost) / exact_solution.total_cost

    print(f"Facilities: {n_facilities}, Customers: {n_customers}")
    print(f"Initial (greedy-add) total cost : {init_solution.total_cost:.6f} (open={open_count_init})")
    print(f"Final   (local-search) total    : {final_solution.total_cost:.6f} (open={open_count_final})")
    print(f"Exact   (brute-force) total     : {exact_solution.total_cost:.6f} (open={open_count_exact})")
    print(f"Relative gap to exact           : {gap:.4f}%")

    print("\nGreedy-add move trace:")
    summarize_moves(init_moves)

    print("\nLocal-search move trace:")
    summarize_moves(ls_moves)

    improved = final_solution.total_cost <= init_solution.total_cost + 1e-9
    print(f"\nQuality checks: improved_vs_greedy={improved}, gap<=10%={gap <= 10.0}")

    return final_solution.total_cost, gap


def main() -> None:
    cases = [
        ("clustered-demand-small", 7, 10, 42),
        ("clustered-demand-medium", 19, 12, 56),
    ]

    final_costs: List[float] = []
    gaps: List[float] = []

    for case_name, seed, m, n in cases:
        final_cost, gap = run_case(case_name, seed, m, n)
        final_costs.append(final_cost)
        gaps.append(gap)

    print("\n=== Summary ===")
    print(f"Average final total cost: {float(np.mean(final_costs)):.6f}")
    print(f"Average exact gap (%):   {float(np.mean(gaps)):.4f}")
    print("Done.")


if __name__ == "__main__":
    main()

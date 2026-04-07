"""Greedy MVP for CS-0062: 分数背包问题.

Run:
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence

import numpy as np

EPS = 1e-9


@dataclass(frozen=True)
class Item:
    """One candidate item in the fractional knapsack."""

    name: str
    weight: float
    value: float


@dataclass(frozen=True)
class SelectedPortion:
    """Chosen fraction for one item."""

    name: str
    fraction: float
    taken_weight: float
    gained_value: float
    density: float


@dataclass(frozen=True)
class FixedCase:
    """Deterministic case for regression checks."""

    name: str
    items: list[Item]
    capacity: float
    expected_value: float


def _normalize_capacity(capacity: float) -> float:
    value = float(capacity)
    if value < 0:
        raise ValueError(f"capacity must be >= 0, got {value}")
    return value


def _normalize_items(items: Iterable[Item]) -> list[Item]:
    normalized: list[Item] = []
    for idx, item in enumerate(items):
        weight = float(item.weight)
        value = float(item.value)
        if weight < 0:
            raise ValueError(f"item[{idx}] has negative weight: {weight}")
        if value < 0:
            raise ValueError(f"item[{idx}] has negative value: {value}")
        normalized.append(Item(name=str(item.name), weight=weight, value=value))
    return normalized


def _density(item: Item) -> float:
    if item.weight == 0:
        return float("inf") if item.value > 0 else 0.0
    return item.value / item.weight


def fractional_knapsack_greedy(
    items: Iterable[Item], capacity: float
) -> tuple[float, list[SelectedPortion]]:
    """Return optimal value and picked portions using greedy density order."""
    cap = _normalize_capacity(capacity)
    values = _normalize_items(items)

    ordered = sorted(
        values,
        key=lambda it: (_density(it), -it.weight, it.name),
        reverse=True,
    )

    remaining = cap
    total_value = 0.0
    chosen: list[SelectedPortion] = []

    for item in ordered:
        if item.weight == 0:
            if item.value > 0:
                chosen.append(
                    SelectedPortion(
                        name=item.name,
                        fraction=1.0,
                        taken_weight=0.0,
                        gained_value=item.value,
                        density=float("inf"),
                    )
                )
                total_value += item.value
            continue

        if remaining <= EPS:
            break

        taken_weight = min(item.weight, remaining)
        fraction = taken_weight / item.weight
        if fraction <= EPS:
            continue

        gained_value = item.value * fraction
        chosen.append(
            SelectedPortion(
                name=item.name,
                fraction=fraction,
                taken_weight=taken_weight,
                gained_value=gained_value,
                density=_density(item),
            )
        )
        total_value += gained_value
        remaining -= taken_weight

    return total_value, chosen


def exact_fractional_by_enumeration(items: Iterable[Item], capacity: float) -> float:
    """Exact solver for small n via subset enumeration + one fractional pivot.

    Idea: with one capacity constraint and box bounds, an optimal basic solution has
    all variables at 0/1 except at most one fractional variable.
    """
    cap = _normalize_capacity(capacity)
    values = _normalize_items(items)
    n = len(values)

    if n == 0 or cap <= EPS:
        # zero-weight positive-value items are still collectible even if cap == 0
        free_gain = sum(it.value for it in values if it.weight == 0 and it.value > 0)
        return free_gain

    best = 0.0

    for pivot in range(-1, n):
        other_indices = [idx for idx in range(n) if idx != pivot]
        m = len(other_indices)

        for r in range(m + 1):
            for picked_pos in combinations(range(m), r):
                picked_set = set(picked_pos)
                total_w = 0.0
                total_v = 0.0

                feasible = True
                for pos, item_idx in enumerate(other_indices):
                    if pos not in picked_set:
                        continue
                    item = values[item_idx]
                    total_w += item.weight
                    total_v += item.value
                    if total_w - cap > EPS:
                        feasible = False
                        break

                if not feasible:
                    continue

                candidate = total_v
                if pivot != -1:
                    p_item = values[pivot]
                    if p_item.weight == 0:
                        if p_item.value > 0:
                            candidate += p_item.value
                    else:
                        remain = cap - total_w
                        if remain > EPS:
                            frac = min(1.0, remain / p_item.weight)
                            candidate += p_item.value * frac

                if candidate > best:
                    best = candidate

    return best


def items_from_numpy(weights: np.ndarray, values: np.ndarray) -> list[Item]:
    """Build item list from two 1D numpy arrays."""
    w = np.asarray(weights, dtype=float)
    v = np.asarray(values, dtype=float)
    if w.ndim != 1 or v.ndim != 1:
        raise ValueError(f"weights and values must be 1D, got {w.shape}, {v.shape}")
    if w.shape[0] != v.shape[0]:
        raise ValueError("weights and values must have the same length")

    return [Item(name=f"np_{i}", weight=float(w[i]), value=float(v[i])) for i in range(w.size)]


def assert_consistency(items: Sequence[Item], capacity: float) -> None:
    greedy_value, _ = fractional_knapsack_greedy(items, capacity)
    exact_value = exact_fractional_by_enumeration(items, capacity)
    assert abs(greedy_value - exact_value) <= 1e-7, (
        f"greedy != exact, capacity={capacity}, greedy={greedy_value}, exact={exact_value}"
    )


def run_fixed_cases() -> None:
    cases = [
        FixedCase(
            name="classic textbook",
            items=[
                Item("A", 10, 60),
                Item("B", 20, 100),
                Item("C", 30, 120),
            ],
            capacity=50,
            expected_value=240.0,
        ),
        FixedCase(
            name="all fit",
            items=[Item("A", 2, 20), Item("B", 3, 18), Item("C", 1, 6)],
            capacity=10,
            expected_value=44.0,
        ),
        FixedCase(
            name="partial at first item",
            items=[Item("A", 5, 50), Item("B", 10, 60)],
            capacity=2,
            expected_value=20.0,
        ),
        FixedCase(
            name="zero capacity with free item",
            items=[Item("free", 0, 5), Item("A", 4, 40)],
            capacity=0,
            expected_value=5.0,
        ),
        FixedCase(
            name="zero weight positive value",
            items=[Item("free", 0, 7), Item("A", 3, 12), Item("B", 2, 4)],
            capacity=3,
            expected_value=19.0,
        ),
    ]

    print("=== Fixed Cases ===")
    for idx, case in enumerate(cases, start=1):
        got, chosen = fractional_knapsack_greedy(case.items, case.capacity)
        assert abs(got - case.expected_value) <= 1e-8, (
            f"Case {case.name} failed: expected={case.expected_value}, got={got}"
        )
        assert_consistency(case.items, case.capacity)
        print(
            f"[{idx}] {case.name}: capacity={case.capacity}, value={got:.6f}, picks={len(chosen)}"
        )


def run_random_verification() -> None:
    rng = np.random.default_rng(62)
    total = 200

    for _ in range(total):
        n = int(rng.integers(1, 9))
        weights = rng.integers(0, 12, size=n)
        values = rng.integers(0, 50, size=n)
        items = [
            Item(name=f"r{i}", weight=float(weights[i]), value=float(values[i]))
            for i in range(n)
        ]
        capacity = float(rng.integers(0, 25))
        assert_consistency(items, capacity)

    print(f"\nRandom verification passed: {total} cases.")


def run_numpy_case() -> None:
    weights = np.array([4, 8, 1, 0, 6], dtype=float)
    values = np.array([20, 30, 6, 3, 18], dtype=float)
    items = items_from_numpy(weights, values)
    capacity = 10.0

    best, chosen = fractional_knapsack_greedy(items, capacity)

    print("\n=== Numpy Case ===")
    print(f"weights: {weights}")
    print(f"values : {values}")
    print(f"capacity: {capacity}")
    print(f"best value: {best:.6f}")
    for row in chosen:
        print(
            "  "
            f"{row.name}: fraction={row.fraction:.6f}, "
            f"taken_weight={row.taken_weight:.6f}, gained_value={row.gained_value:.6f}"
        )


def main() -> None:
    run_fixed_cases()
    run_random_verification()
    run_numpy_case()
    print("\nAll checks passed for CS-0062 (分数背包问题).")


if __name__ == "__main__":
    main()

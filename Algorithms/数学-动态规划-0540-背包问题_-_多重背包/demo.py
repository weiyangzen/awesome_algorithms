"""Multi-knapsack MVP: binary-splitting DP + naive bounded-DP cross-check."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ItemType:
    weight: int
    value: int
    count: int


@dataclass(frozen=True)
class SplitPack:
    type_index: int
    units: int
    weight: int
    value: int


@dataclass
class MultiKnapsackResult:
    max_value: int
    total_weight: int
    picks: List[int]


def validate_items(items: Sequence[ItemType], capacity: int) -> None:
    if capacity < 0:
        raise ValueError(f"Capacity must be >= 0, got {capacity}.")
    for idx, item in enumerate(items):
        if item.weight <= 0:
            raise ValueError(f"Item[{idx}] has non-positive weight: {item.weight}.")
        if item.value < 0:
            raise ValueError(f"Item[{idx}] has negative value: {item.value}.")
        if item.count < 0:
            raise ValueError(f"Item[{idx}] has negative count: {item.count}.")


def split_items_binary(items: Sequence[ItemType]) -> List[SplitPack]:
    packs: List[SplitPack] = []
    for idx, item in enumerate(items):
        remaining = item.count
        k = 1
        while remaining > 0:
            take_units = min(k, remaining)
            packs.append(
                SplitPack(
                    type_index=idx,
                    units=take_units,
                    weight=take_units * item.weight,
                    value=take_units * item.value,
                )
            )
            remaining -= take_units
            k <<= 1
    return packs


def solve_multi_knapsack_binary_split(
    items: Sequence[ItemType], capacity: int
) -> MultiKnapsackResult:
    validate_items(items, capacity)
    n = len(items)
    if n == 0 or capacity == 0:
        return MultiKnapsackResult(max_value=0, total_weight=0, picks=[0] * n)

    packs = split_items_binary(items)
    m = len(packs)

    dp = np.zeros((m + 1, capacity + 1), dtype=np.int64)
    take = np.zeros((m + 1, capacity + 1), dtype=np.bool_)

    for i in range(1, m + 1):
        pack = packs[i - 1]
        w = pack.weight
        v = pack.value
        for cap in range(capacity + 1):
            no_take = int(dp[i - 1, cap])
            best = no_take
            if cap >= w:
                with_take = int(dp[i - 1, cap - w]) + v
                if with_take > best:
                    best = with_take
                    take[i, cap] = True
            dp[i, cap] = best

    picks = [0] * n
    cap = capacity
    for i in range(m, 0, -1):
        if bool(take[i, cap]):
            pack = packs[i - 1]
            picks[pack.type_index] += pack.units
            cap -= pack.weight

    total_weight = sum(items[i].weight * picks[i] for i in range(n))
    max_value = int(dp[m, capacity])
    return MultiKnapsackResult(max_value=max_value, total_weight=total_weight, picks=picks)


def solve_multi_knapsack_naive(
    items: Sequence[ItemType], capacity: int
) -> MultiKnapsackResult:
    validate_items(items, capacity)
    n = len(items)
    if n == 0 or capacity == 0:
        return MultiKnapsackResult(max_value=0, total_weight=0, picks=[0] * n)

    dp = np.zeros((n + 1, capacity + 1), dtype=np.int64)
    choose = np.zeros((n + 1, capacity + 1), dtype=np.int64)

    for i in range(1, n + 1):
        item = items[i - 1]
        for cap in range(capacity + 1):
            best_value = int(dp[i - 1, cap])
            best_take = 0
            max_units = min(item.count, cap // item.weight)
            for units in range(1, max_units + 1):
                cand = int(dp[i - 1, cap - units * item.weight]) + units * item.value
                if cand > best_value:
                    best_value = cand
                    best_take = units
            dp[i, cap] = best_value
            choose[i, cap] = best_take

    picks = [0] * n
    cap = capacity
    for i in range(n, 0, -1):
        taken = int(choose[i, cap])
        picks[i - 1] = taken
        cap -= taken * items[i - 1].weight

    total_weight = sum(items[i].weight * picks[i] for i in range(n))
    max_value = int(dp[n, capacity])
    return MultiKnapsackResult(max_value=max_value, total_weight=total_weight, picks=picks)


def evaluate_solution(
    items: Sequence[ItemType], picks: Sequence[int], capacity: int
) -> Tuple[int, int, bool]:
    total_weight = sum(item.weight * pick for item, pick in zip(items, picks))
    total_value = sum(item.value * pick for item, pick in zip(items, picks))
    feasible = total_weight <= capacity and all(
        0 <= pick <= item.count for item, pick in zip(items, picks)
    )
    return total_weight, total_value, feasible


def run_case(name: str, items: Sequence[ItemType], capacity: int) -> None:
    fast = solve_multi_knapsack_binary_split(items, capacity)
    slow = solve_multi_knapsack_naive(items, capacity)

    fast_weight, fast_value, fast_feasible = evaluate_solution(items, fast.picks, capacity)
    slow_weight, slow_value, slow_feasible = evaluate_solution(items, slow.picks, capacity)

    print(f"=== {name} ===")
    print(f"Capacity: {capacity}")
    print("Items (weight, value, count):")
    for idx, item in enumerate(items):
        print(f"  - #{idx}: ({item.weight}, {item.value}, {item.count})")
    print(
        "Fast  (binary split): "
        f"value={fast.max_value}, weight={fast.total_weight}, picks={fast.picks}"
    )
    print(
        "Naive (bounded DP):  "
        f"value={slow.max_value}, weight={slow.total_weight}, picks={slow.picks}"
    )
    print(
        "Checks: "
        f"value_equal={fast.max_value == slow.max_value}, "
        f"fast_feasible={fast_feasible}, slow_feasible={slow_feasible}, "
        f"fast_value_match={fast.max_value == fast_value}, "
        f"slow_value_match={slow.max_value == slow_value}, "
        f"fast_weight_match={fast.total_weight == fast_weight}, "
        f"slow_weight_match={slow.total_weight == slow_weight}"
    )
    print()

    if fast.max_value != slow.max_value:
        raise AssertionError(
            f"Value mismatch in case '{name}': fast={fast.max_value}, slow={slow.max_value}"
        )
    if not (fast_feasible and slow_feasible):
        raise AssertionError(f"Infeasible solution found in case '{name}'.")
    if fast.max_value != fast_value or slow.max_value != slow_value:
        raise AssertionError(f"Value re-check failed in case '{name}'.")
    if fast.total_weight != fast_weight or slow.total_weight != slow_weight:
        raise AssertionError(f"Weight re-check failed in case '{name}'.")


def main() -> None:
    cases = [
        (
            "Case 1: classic",
            [
                ItemType(weight=2, value=6, count=3),
                ItemType(weight=3, value=10, count=2),
                ItemType(weight=5, value=12, count=1),
            ],
            10,
        ),
        (
            "Case 2: sparse counts",
            [
                ItemType(weight=4, value=7, count=1),
                ItemType(weight=6, value=12, count=4),
                ItemType(weight=3, value=5, count=0),
            ],
            18,
        ),
        (
            "Case 3: zero capacity",
            [
                ItemType(weight=1, value=2, count=10),
                ItemType(weight=7, value=20, count=1),
            ],
            0,
        ),
        (
            "Case 4: mixed scale",
            [
                ItemType(weight=2, value=3, count=5),
                ItemType(weight=5, value=9, count=3),
                ItemType(weight=7, value=13, count=2),
                ItemType(weight=9, value=18, count=2),
            ],
            25,
        ),
        ("Case 5: empty items", [], 15),
    ]

    for name, items, capacity in cases:
        run_case(name, items, capacity)


if __name__ == "__main__":
    main()

"""Complete Knapsack (Unbounded Knapsack) minimal runnable MVP.

This script solves one fixed demo instance with dynamic programming,
reconstructs an item-count solution, and verifies optimality via
small-scale brute force enumeration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Item:
    """Item type in complete knapsack: each item can be used unlimited times."""

    name: str
    weight: int
    value: int


def validate_inputs(items: Sequence[Item], capacity: int) -> None:
    """Validate problem input before running DP."""
    if capacity < 0:
        raise ValueError("capacity must be >= 0")
    if not items:
        raise ValueError("items must not be empty")

    for idx, item in enumerate(items):
        if item.weight <= 0:
            raise ValueError(f"item[{idx}] weight must be > 0")
        if item.value < 0:
            raise ValueError(f"item[{idx}] value must be >= 0")


def solve_complete_knapsack(items: Sequence[Item], capacity: int) -> tuple[int, List[int], np.ndarray]:
    """Solve complete knapsack with 1D DP and reconstruct one optimal plan.

    Recurrence (unbounded):
        dp[c] = max(dp[c], dp[c - w_i] + v_i)  for c from w_i..capacity

    Returns:
        best_value: optimal total value under capacity.
        counts: multiplicity of each item in one optimal solution.
        dp: full DP value table from 0..capacity.
    """
    validate_inputs(items, capacity)

    dp = np.zeros(capacity + 1, dtype=np.int64)
    parent_item = np.full(capacity + 1, -1, dtype=np.int64)
    parent_capacity = np.full(capacity + 1, -1, dtype=np.int64)

    for i, item in enumerate(items):
        for c in range(item.weight, capacity + 1):
            candidate = int(dp[c - item.weight] + item.value)
            if candidate > int(dp[c]):
                dp[c] = candidate
                parent_item[c] = i
                parent_capacity[c] = c - item.weight

    counts = [0] * len(items)
    c = capacity
    while c > 0 and parent_item[c] != -1:
        i = int(parent_item[c])
        counts[i] += 1
        c = int(parent_capacity[c])

    return int(dp[capacity]), counts, dp


def brute_force_check(items: Sequence[Item], capacity: int) -> int:
    """Compute optimum by exhaustive count enumeration for small demos.

    This is intentionally simple and only used for correctness checking.
    """

    def dfs(idx: int, remaining: int) -> int:
        if idx == len(items):
            return 0

        item = items[idx]
        max_k = remaining // item.weight
        best = 0
        for k in range(max_k + 1):
            candidate = k * item.value + dfs(idx + 1, remaining - k * item.weight)
            if candidate > best:
                best = candidate
        return best

    return dfs(0, capacity)


def solution_summary(items: Sequence[Item], counts: Sequence[int]) -> pd.DataFrame:
    """Create a compact table for the reconstructed item multiplicities."""
    rows = []
    for item, count in zip(items, counts):
        rows.append(
            {
                "item": item.name,
                "weight": item.weight,
                "value": item.value,
                "count": int(count),
                "total_weight": int(count * item.weight),
                "total_value": int(count * item.value),
            }
        )

    df = pd.DataFrame(rows)
    return df


def main() -> None:
    # Fixed, deterministic demo instance; no interactive input needed.
    items = [
        Item("A", weight=2, value=20),
        Item("B", weight=3, value=30),
        Item("C", weight=4, value=45),
        Item("D", weight=7, value=80),
    ]
    capacity = 17

    best_value, counts, dp = solve_complete_knapsack(items, capacity)
    brute_force_value = brute_force_check(items, capacity)

    plan_df = solution_summary(items, counts)
    used_weight = int(plan_df["total_weight"].sum())
    used_value = int(plan_df["total_value"].sum())

    print("=== Complete Knapsack Demo ===")
    print(f"capacity = {capacity}")
    print("items:")
    print(pd.DataFrame([item.__dict__ for item in items]).to_string(index=False))

    print("\nDP value table (capacity -> best value):")
    dp_table = pd.DataFrame({"capacity": np.arange(capacity + 1), "best_value": dp})
    print(dp_table.to_string(index=False))

    print("\nReconstructed one optimal plan:")
    print(plan_df.to_string(index=False))

    print("\nSummary:")
    print(f"best_value_by_dp      = {best_value}")
    print(f"best_value_by_plan    = {used_value}")
    print(f"best_value_bruteforce = {brute_force_value}")
    print(f"used_weight           = {used_weight}")
    print(f"feasible              = {used_weight <= capacity}")
    print(f"dp_matches_bruteforce = {best_value == brute_force_value}")

    if used_value != best_value:
        raise RuntimeError("reconstruction value does not match DP optimum")
    if brute_force_value != best_value:
        raise RuntimeError("DP result does not match brute-force optimum")


if __name__ == "__main__":
    main()

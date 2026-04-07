"""Partition Equal Subset Sum MVP with DP trace and cross-checks.

Run:
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from random import Random
from typing import Sequence

import numpy as np


@dataclass
class PartitionResult:
    can_partition: bool
    target: int
    subset_a_indices: list[int]
    subset_b_indices: list[int]
    subset_a_values: list[int]
    subset_b_values: list[int]


def to_1d_nonnegative_int_array(values: Sequence[int] | np.ndarray) -> np.ndarray:
    """Validate and normalize input into a 1D non-negative integer array."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"Input must be a 1D sequence, got shape={arr.shape}.")
    if arr.size > 0 and not np.all(np.isfinite(arr)):
        raise ValueError("Input contains non-finite values (nan or inf).")

    rounded = np.rint(arr)
    if arr.size > 0 and not np.allclose(arr, rounded):
        raise ValueError("All values must be integers.")

    ints = rounded.astype(np.int64)
    if ints.size > 0 and np.any(ints < 0):
        raise ValueError("All values must be non-negative.")
    return ints


def partition_equal_subset_dp_trace(values: Sequence[int] | np.ndarray) -> PartitionResult:
    """2D DP feasibility + trace back one concrete partition plan."""
    nums = to_1d_nonnegative_int_array(values)
    n = int(nums.size)
    total = int(np.sum(nums))

    if total % 2 == 1:
        return PartitionResult(
            can_partition=False,
            target=total // 2,
            subset_a_indices=[],
            subset_b_indices=[],
            subset_a_values=[],
            subset_b_values=[],
        )

    target = total // 2
    dp = np.zeros((n + 1, target + 1), dtype=bool)
    dp[:, 0] = True

    for i in range(1, n + 1):
        num = int(nums[i - 1])
        dp[i] = dp[i - 1]
        if num <= target:
            dp[i, num:] = dp[i, num:] | dp[i - 1, : target - num + 1]

    if not bool(dp[n, target]):
        return PartitionResult(
            can_partition=False,
            target=target,
            subset_a_indices=[],
            subset_b_indices=[],
            subset_a_values=[],
            subset_b_values=[],
        )

    chosen: list[int] = []
    s = target
    for i in range(n, 0, -1):
        num = int(nums[i - 1])

        # Tie-breaker: prefer "not take" when both choices are feasible.
        if bool(dp[i - 1, s]):
            continue

        if s >= num and bool(dp[i - 1, s - num]):
            chosen.append(i - 1)
            s -= num
            continue

        raise AssertionError("Backtracking failed: inconsistent DP table.")

    if s != 0:
        raise AssertionError("Backtracking failed: target residue is not zero.")

    chosen.reverse()
    chosen_set = set(chosen)
    other = [idx for idx in range(n) if idx not in chosen_set]

    subset_a_values = [int(nums[idx]) for idx in chosen]
    subset_b_values = [int(nums[idx]) for idx in other]

    return PartitionResult(
        can_partition=True,
        target=target,
        subset_a_indices=chosen,
        subset_b_indices=other,
        subset_a_values=subset_a_values,
        subset_b_values=subset_b_values,
    )


def can_partition_dp_optimized(values: Sequence[int] | np.ndarray) -> bool:
    """1D DP feasibility check with O(target) space."""
    nums = to_1d_nonnegative_int_array(values)
    total = int(np.sum(nums))

    if total % 2 == 1:
        return False

    target = total // 2
    dp = np.zeros(target + 1, dtype=bool)
    dp[0] = True

    for num in nums:
        v = int(num)
        if v > target:
            continue
        for s in range(target, v - 1, -1):
            dp[s] = bool(dp[s] or dp[s - v])

    return bool(dp[target])


def can_partition_bruteforce(values: Sequence[int] | np.ndarray) -> bool:
    """Exact solver for small arrays, used for correctness regression."""
    nums = to_1d_nonnegative_int_array(values)
    n = int(nums.size)
    total = int(np.sum(nums))

    if total % 2 == 1:
        return False

    target = total // 2

    @lru_cache(maxsize=None)
    def dfs(i: int, s: int) -> bool:
        if s == target:
            return True
        if i >= n or s > target:
            return False

        return dfs(i + 1, s) or dfs(i + 1, s + int(nums[i]))

    return dfs(0, 0)


def validate_partition_result(values: Sequence[int] | np.ndarray, result: PartitionResult) -> bool:
    nums = to_1d_nonnegative_int_array(values)
    n = int(nums.size)

    a_idx = result.subset_a_indices
    b_idx = result.subset_b_indices

    if not result.can_partition:
        return a_idx == [] and b_idx == []

    all_idx = a_idx + b_idx
    if len(all_idx) != n:
        return False

    if len(set(all_idx)) != n:
        return False

    if sorted(all_idx) != list(range(n)):
        return False

    if any(a_idx[i] >= a_idx[i + 1] for i in range(len(a_idx) - 1)):
        return False
    if any(b_idx[i] >= b_idx[i + 1] for i in range(len(b_idx) - 1)):
        return False

    sum_a = int(sum(result.subset_a_values))
    sum_b = int(sum(result.subset_b_values))

    if sum_a != result.target or sum_b != result.target:
        return False

    if sum_a + sum_b != int(np.sum(nums)):
        return False

    return True


def run_case(name: str, values: Sequence[int]) -> None:
    traced = partition_equal_subset_dp_trace(values)
    optimized = can_partition_dp_optimized(values)
    brute = can_partition_bruteforce(values) if len(values) <= 24 else None

    valid_partition = validate_partition_result(values, traced)
    balanced_sum = (
        sum(traced.subset_a_values) == sum(traced.subset_b_values)
        if traced.can_partition
        else True
    )

    print(f"=== {name} ===")
    print(f"nums            = {list(values)}")
    print(
        "dp_trace        -> "
        f"can_partition={traced.can_partition}, target={traced.target}, "
        f"subset_a_indices={traced.subset_a_indices}, subset_b_indices={traced.subset_b_indices}, "
        f"subset_a_values={traced.subset_a_values}, subset_b_values={traced.subset_b_values}"
    )
    print(f"dp_optimized    -> can_partition={optimized}")
    if brute is not None:
        print(f"bruteforce      -> can_partition={brute}")

    print(
        "checks          -> "
        f"valid_partition={valid_partition}, balanced_sum={balanced_sum}, "
        f"optimized_match={optimized == traced.can_partition}, "
        f"bruteforce_match={None if brute is None else brute == traced.can_partition}"
    )
    print()

    if not valid_partition:
        raise AssertionError(f"Invalid partition result in case '{name}'.")
    if not balanced_sum:
        raise AssertionError(f"Unbalanced subset sums in case '{name}'.")
    if optimized != traced.can_partition:
        raise AssertionError(f"Optimized DP mismatch in case '{name}'.")
    if brute is not None and brute != traced.can_partition:
        raise AssertionError(f"Bruteforce mismatch in case '{name}'.")


def randomized_regression(seed: int = 2026, rounds: int = 220) -> None:
    """Randomized consistency test among 2D DP trace, 1D DP, and brute-force."""
    rng = Random(seed)

    for _ in range(rounds):
        n = rng.randint(0, 14)
        values = [rng.randint(0, 20) for _ in range(n)]

        traced = partition_equal_subset_dp_trace(values)
        optimized = can_partition_dp_optimized(values)
        brute = can_partition_bruteforce(values)

        assert validate_partition_result(values, traced)
        assert optimized == traced.can_partition
        assert brute == traced.can_partition

    print(
        "randomized regression passed: "
        f"seed={seed}, rounds={rounds}, n_range=[0,14], value_range=[0,20]"
    )


def main() -> None:
    cases = {
        "Case 1: classic true": [1, 5, 11, 5],
        "Case 2: classic false": [1, 2, 3, 5],
        "Case 3: even total true": [2, 2, 1, 1],
        "Case 4: odd total false": [2, 2, 2, 1],
        "Case 5: contains zeros": [0, 0, 0, 0],
        "Case 6: single element": [2],
        "Case 7: empty": [],
        "Case 8: mixed true": [3, 3, 3, 4, 5],
    }

    for name, values in cases.items():
        run_case(name, values)

    randomized_regression()


if __name__ == "__main__":
    main()

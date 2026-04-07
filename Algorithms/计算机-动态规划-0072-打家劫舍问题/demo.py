"""House Robber MVP with DP reconstruction and cross-checks.

Run:
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from random import Random
from typing import Iterable, Sequence

import numpy as np


@dataclass
class RobberyResult:
    total: int
    indices: list[int]
    values: list[int]


def to_1d_nonnegative_int_array(values: Sequence[int] | np.ndarray) -> np.ndarray:
    """Validate and normalize input into a 1D non-negative integer array."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"Input must be a 1D sequence, got shape={arr.shape}.")
    if arr.size > 0 and not np.all(np.isfinite(arr)):
        raise ValueError("Input contains non-finite values (nan or inf).")

    rounded = np.rint(arr)
    if arr.size > 0 and not np.allclose(arr, rounded):
        raise ValueError("House amounts must be integers.")

    ints = rounded.astype(np.int64)
    if ints.size > 0 and np.any(ints < 0):
        raise ValueError("House amounts must be non-negative.")
    return ints


def house_robber_dp_trace(values: Sequence[int] | np.ndarray) -> RobberyResult:
    """O(n) DP with path reconstruction.

    State:
        dp[i] = best total using first i houses (house indices 0..i-1)
    Transition:
        dp[i] = max(dp[i-1], dp[i-2] + nums[i-1])
    """
    nums = to_1d_nonnegative_int_array(values)
    n = int(nums.size)
    if n == 0:
        return RobberyResult(total=0, indices=[], values=[])

    dp = [0] * (n + 1)

    for i in range(1, n + 1):
        take = int(nums[i - 1]) + (dp[i - 2] if i >= 2 else 0)
        skip = dp[i - 1]
        dp[i] = max(skip, take)

    # Reconstruct one optimal plan.
    chosen: list[int] = []
    i = n
    while i >= 1:
        take = int(nums[i - 1]) + (dp[i - 2] if i >= 2 else 0)
        skip = dp[i - 1]

        # Tie-breaker: prefer skip for deterministic output.
        if skip >= take:
            i -= 1
        else:
            chosen.append(i - 1)
            i -= 2

    chosen.reverse()
    chosen_values = [int(nums[idx]) for idx in chosen]
    return RobberyResult(total=int(dp[n]), indices=chosen, values=chosen_values)


def house_robber_dp_optimized_value(values: Sequence[int] | np.ndarray) -> int:
    """O(n) time, O(1) space DP returning only the optimal value."""
    nums = to_1d_nonnegative_int_array(values)

    prev2 = 0  # dp[i-2]
    prev1 = 0  # dp[i-1]
    for amount in nums:
        cur = max(prev1, prev2 + int(amount))
        prev2, prev1 = prev1, cur
    return int(prev1)


def house_robber_bruteforce(values: Sequence[int] | np.ndarray) -> RobberyResult:
    """Exact solver for small n, used for correctness cross-check."""
    nums = to_1d_nonnegative_int_array(values)
    n = int(nums.size)

    @lru_cache(maxsize=None)
    def dfs(i: int, prev_taken: bool) -> tuple[int, tuple[int, ...]]:
        if i >= n:
            return 0, ()

        skip_total, skip_indices = dfs(i + 1, False)
        best_total, best_indices = skip_total, skip_indices

        if not prev_taken:
            take_total_next, take_indices_next = dfs(i + 1, True)
            take_total = int(nums[i]) + take_total_next
            take_indices = (i,) + take_indices_next

            # Prefer larger total. If equal, prefer lexicographically smaller indices.
            if (take_total > best_total) or (
                take_total == best_total and take_indices < best_indices
            ):
                best_total, best_indices = take_total, take_indices

        return best_total, best_indices

    total, indices_tuple = dfs(0, False)
    indices = list(indices_tuple)
    chosen_values = [int(nums[idx]) for idx in indices]
    return RobberyResult(total=int(total), indices=indices, values=chosen_values)


def is_valid_plan(values: Sequence[int] | np.ndarray, indices: Iterable[int]) -> bool:
    nums = to_1d_nonnegative_int_array(values)
    idx_list = list(indices)

    if any(not 0 <= idx < len(nums) for idx in idx_list):
        return False

    if any(idx_list[i] >= idx_list[i + 1] for i in range(len(idx_list) - 1)):
        return False

    if any(idx_list[i + 1] - idx_list[i] == 1 for i in range(len(idx_list) - 1)):
        return False

    return True


def run_case(name: str, values: Sequence[int]) -> None:
    traced = house_robber_dp_trace(values)
    optimized_total = house_robber_dp_optimized_value(values)
    brute = house_robber_bruteforce(values) if len(values) <= 20 else None

    valid = is_valid_plan(values, traced.indices)
    traced_sum = int(sum(traced.values))

    print(f"=== {name} ===")
    print(f"houses = {list(values)}")
    print(
        "dp_trace      -> "
        f"total={traced.total}, indices={traced.indices}, chosen_values={traced.values}"
    )
    print(f"dp_optimized  -> total={optimized_total}")

    if brute is not None:
        print(
            "bruteforce    -> "
            f"total={brute.total}, indices={brute.indices}, chosen_values={brute.values}"
        )

    print(
        "checks        -> "
        f"valid_plan={valid}, traced_sum_match={traced_sum == traced.total}, "
        f"optimized_match={optimized_total == traced.total}, "
        f"bruteforce_match={None if brute is None else brute.total == traced.total}"
    )
    print()

    if not valid:
        raise AssertionError(f"Invalid plan in case '{name}'.")
    if traced_sum != traced.total:
        raise AssertionError(f"Sum mismatch in case '{name}'.")
    if optimized_total != traced.total:
        raise AssertionError(f"Optimized DP mismatch in case '{name}'.")
    if brute is not None and brute.total != traced.total:
        raise AssertionError(f"Bruteforce mismatch in case '{name}'.")


def randomized_regression(seed: int = 2026, rounds: int = 200) -> None:
    """Randomized consistency test for small arrays."""
    rng = Random(seed)

    for _ in range(rounds):
        n = rng.randint(0, 14)
        values = [rng.randint(0, 30) for _ in range(n)]

        traced = house_robber_dp_trace(values)
        optimized_total = house_robber_dp_optimized_value(values)
        brute = house_robber_bruteforce(values)

        assert is_valid_plan(values, traced.indices)
        assert sum(traced.values) == traced.total
        assert optimized_total == traced.total
        assert brute.total == traced.total

    print(
        "randomized regression passed: "
        f"seed={seed}, rounds={rounds}, n_range=[0,14], value_range=[0,30]"
    )


def main() -> None:
    cases = {
        "Case 1: classic": [1, 2, 3, 1],
        "Case 2: common": [2, 7, 9, 3, 1],
        "Case 3: zeros": [0, 0, 0, 0],
        "Case 4: single": [9],
        "Case 5: empty": [],
        "Case 6: mixed": [6, 1, 2, 10, 4, 2, 7],
        "Case 7: tie behavior": [2, 2, 2, 2],
    }

    for name, values in cases.items():
        run_case(name, values)

    randomized_regression()


if __name__ == "__main__":
    main()

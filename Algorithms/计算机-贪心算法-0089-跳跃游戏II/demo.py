"""Greedy MVP for CS-0069: 跳跃游戏II.

Run:
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class FixedCase:
    """Deterministic case for Jump Game II minimal jumps."""

    name: str
    nums: list[int]
    expected: int | None


def _normalize_nums(nums: Iterable[int]) -> list[int]:
    """Convert input to validated non-negative integer list."""
    normalized: list[int] = []
    for idx, value in enumerate(nums):
        jump = int(value)
        if jump < 0:
            raise ValueError(f"Jump length at index {idx} must be non-negative, got {jump}.")
        normalized.append(jump)
    return normalized


def min_jumps_greedy(nums: Iterable[int]) -> int | None:
    """Return minimum jumps using O(n) greedy frontier expansion.

    Returns:
        int: minimal jumps when the last index is reachable.
        None: when the last index is unreachable.
    """
    values = _normalize_nums(nums)
    n = len(values)

    if n <= 1:
        return 0

    jumps = 0
    current_end = 0
    farthest = 0

    for i in range(n - 1):
        if i > farthest:
            return None

        farthest = max(farthest, i + values[i])

        # Finish one BFS-like layer: commit one jump and move to next frontier.
        if i == current_end:
            jumps += 1
            current_end = farthest
            if current_end >= n - 1:
                return jumps

    return jumps if current_end >= n - 1 else None


def min_jumps_dp(nums: Iterable[int]) -> int | None:
    """Reference O(n^2) dynamic programming for minimal jumps."""
    values = _normalize_nums(nums)
    n = len(values)
    if n <= 1:
        return 0

    inf = 10**9
    dp = [inf] * n
    dp[0] = 0

    for i in range(n):
        if dp[i] == inf:
            continue
        furthest = min(n - 1, i + values[i])
        for nxt in range(i + 1, furthest + 1):
            candidate = dp[i] + 1
            if candidate < dp[nxt]:
                dp[nxt] = candidate

    return None if dp[-1] == inf else dp[-1]


def min_jumps_bruteforce(nums: Iterable[int]) -> int | None:
    """Exact solver for small arrays via DFS + memoization."""
    values = _normalize_nums(nums)
    n = len(values)

    if n <= 1:
        return 0

    inf = 10**9

    @lru_cache(maxsize=None)
    def dfs(i: int) -> int:
        if i >= n - 1:
            return 0

        max_step = values[i]
        if max_step == 0:
            return inf

        best = inf
        furthest = min(n - 1, i + max_step)
        for nxt in range(i + 1, furthest + 1):
            best = min(best, 1 + dfs(nxt))
        return best

    ans = dfs(0)
    return None if ans >= inf else ans


def min_jumps_numpy(nums_array: np.ndarray) -> int | None:
    """Bridge function: accept 1D numpy array and run greedy solver."""
    arr = np.asarray(nums_array)
    if arr.ndim != 1:
        raise ValueError(f"Expected a 1D numpy array, got shape {arr.shape}")
    return min_jumps_greedy(arr.tolist())


def assert_consistency(nums: Sequence[int]) -> None:
    """Check greedy answer against DP and brute-force baselines."""
    greedy = min_jumps_greedy(nums)
    dp = min_jumps_dp(nums)
    assert greedy == dp, f"Greedy != DP for nums={nums}: greedy={greedy}, dp={dp}"

    if len(nums) <= 12:
        brute = min_jumps_bruteforce(nums)
        assert greedy == brute, (
            f"Greedy != brute-force for nums={nums}: greedy={greedy}, brute={brute}"
        )


def run_fixed_cases() -> None:
    cases = [
        FixedCase(name="example reachable", nums=[2, 3, 1, 1, 4], expected=2),
        FixedCase(name="reachable with zero", nums=[2, 3, 0, 1, 4], expected=2),
        FixedCase(name="single element", nums=[0], expected=0),
        FixedCase(name="unit steps", nums=[1, 1, 1, 1], expected=3),
        FixedCase(name="unreachable trap", nums=[3, 2, 1, 0, 4], expected=None),
        FixedCase(name="short unreachable", nums=[1, 0, 1], expected=None),
        FixedCase(name="empty input", nums=[], expected=0),
    ]

    print("=== Fixed Cases ===")
    for i, case in enumerate(cases, start=1):
        got = min_jumps_greedy(case.nums)
        assert got == case.expected, (
            f"Case {case.name} failed: expected={case.expected}, got={got}"
        )
        assert_consistency(case.nums)
        print(f"[{i}] {case.name}: nums={case.nums} -> min_jumps={got}")


def run_random_verification() -> None:
    rng = np.random.default_rng(2026)
    total = 300

    for _ in range(total):
        n = int(rng.integers(0, 13))
        nums = rng.integers(0, 8, size=n).tolist()
        assert_consistency(nums)

    print(f"\nRandom verification passed: {total} cases.")


def run_numpy_case() -> None:
    rng = np.random.default_rng(89)
    nums_array = rng.integers(0, 6, size=10)
    jumps = min_jumps_numpy(nums_array)

    print("\n=== Numpy Case ===")
    print(f"nums array: {nums_array}")
    print(f"greedy min jumps: {jumps}")


def main() -> None:
    run_fixed_cases()
    run_random_verification()
    run_numpy_case()
    print("\nAll checks passed for CS-0069 (跳跃游戏II).")


if __name__ == "__main__":
    main()

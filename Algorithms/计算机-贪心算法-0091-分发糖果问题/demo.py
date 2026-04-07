"""Greedy candy-distribution MVP for CS-0071.

Run:
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FixedCase:
    """Deterministic case for candy distribution."""

    name: str
    ratings: list[int]
    expected_total: int


def validate_ratings(ratings: Iterable[Real]) -> list[int]:
    """Validate input ratings and convert to a normalized int list."""
    normalized: list[int] = []
    for idx, value in enumerate(ratings):
        if not isinstance(value, Real):
            raise TypeError(f"ratings[{idx}] is not numeric: {value!r}")
        if not np.isfinite(float(value)):
            raise ValueError(f"ratings[{idx}] is not finite: {value!r}")
        normalized.append(int(value))
    return normalized


def candy_greedy(ratings: Sequence[int]) -> tuple[int, list[int]]:
    """Two-pass greedy solution: O(n) time, O(n) space."""
    arr = validate_ratings(ratings)
    n = len(arr)
    if n == 0:
        return 0, []

    candies = [1] * n

    for i in range(1, n):
        if arr[i] > arr[i - 1]:
            candies[i] = candies[i - 1] + 1

    for i in range(n - 2, -1, -1):
        if arr[i] > arr[i + 1]:
            candies[i] = max(candies[i], candies[i + 1] + 1)

    return int(sum(candies)), candies


def assert_distribution_valid(ratings: Sequence[int], candies: Sequence[int]) -> None:
    """Check all candy constraints for a candidate distribution."""
    assert len(ratings) == len(candies), "ratings/candies length mismatch"
    for i, c in enumerate(candies):
        assert c >= 1, f"candies[{i}] must be >= 1, got {c}"

    for i in range(len(ratings) - 1):
        if ratings[i] > ratings[i + 1]:
            assert candies[i] > candies[i + 1], (
                f"constraint failed at ({i},{i+1}): "
                f"ratings {ratings[i]}>{ratings[i+1]} but candies {candies[i]}<={candies[i+1]}"
            )
        if ratings[i] < ratings[i + 1]:
            assert candies[i] < candies[i + 1], (
                f"constraint failed at ({i},{i+1}): "
                f"ratings {ratings[i]}<{ratings[i+1]} but candies {candies[i]}>={candies[i+1]}"
            )


def candy_exact_small(ratings: Sequence[int], max_n: int = 10) -> int:
    """Exact solver for small n using branch-and-bound enumeration.

    This is only for verification in MVP tests, not the production algorithm.
    """
    arr = validate_ratings(ratings)
    n = len(arr)
    if n == 0:
        return 0
    if n > max_n:
        raise ValueError(f"Exact search supports n <= {max_n}, got {n}")

    greedy_total, _ = candy_greedy(arr)
    upper = n
    best = greedy_total
    chosen = [0] * n

    def dfs(i: int, partial_sum: int) -> None:
        nonlocal best
        if partial_sum + (n - i) >= best:
            return
        if i == n:
            best = min(best, partial_sum)
            return

        low, high = 1, upper
        if i > 0:
            prev = chosen[i - 1]
            if arr[i] > arr[i - 1]:
                low = max(low, prev + 1)
            elif arr[i] < arr[i - 1]:
                high = min(high, prev - 1)

        if i < n - 1:
            if arr[i] > arr[i + 1]:
                low = max(low, 2)
            elif arr[i] < arr[i + 1]:
                high = min(high, upper - 1)

        if low > high:
            return

        for candy in range(low, high + 1):
            chosen[i] = candy
            dfs(i + 1, partial_sum + candy)

    dfs(0, 0)
    return int(best)


def _distribution_table(ratings: Sequence[int], candies: Sequence[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "index": list(range(len(ratings))),
            "rating": list(ratings),
            "candy": list(candies),
        }
    )


def run_fixed_cases() -> None:
    cases = [
        FixedCase(name="leetcode_sample_1", ratings=[1, 0, 2], expected_total=5),
        FixedCase(name="leetcode_sample_2", ratings=[1, 2, 2], expected_total=4),
        FixedCase(name="strictly_increasing", ratings=[1, 2, 3, 4], expected_total=10),
        FixedCase(name="strictly_decreasing", ratings=[5, 4, 3, 2, 1], expected_total=15),
        FixedCase(name="all_equal", ratings=[3, 3, 3, 3], expected_total=4),
        FixedCase(name="mixed_with_negative", ratings=[-1, 0, -1, 2, 2], expected_total=7),
    ]

    print("=== Fixed Cases ===")
    for idx, case in enumerate(cases, start=1):
        total, candies = candy_greedy(case.ratings)
        assert_distribution_valid(case.ratings, candies)
        exact_total = candy_exact_small(case.ratings)

        assert total == case.expected_total, (
            f"Case {case.name} expected total={case.expected_total}, got={total}"
        )
        assert total == exact_total, (
            f"Case {case.name} mismatch exact optimum: greedy={total}, exact={exact_total}"
        )

        print(f"[{idx}] {case.name}")
        print(f"ratings={case.ratings}")
        print(f"candies={candies}, total={total}, exact_total={exact_total}")
        print(_distribution_table(case.ratings, candies).to_string(index=False))
        print()


def run_random_small_verification(num_cases: int = 6) -> None:
    """Generate random small cases and compare greedy result with exact optimum."""
    rng = np.random.default_rng(2026)
    print("=== Random Small Verification ===")
    for k in range(1, num_cases + 1):
        n = int(rng.integers(1, 8))
        ratings = rng.integers(-3, 6, size=n).tolist()
        total, candies = candy_greedy(ratings)
        exact_total = candy_exact_small(ratings)

        assert_distribution_valid(ratings, candies)
        assert total == exact_total, (
            f"Random case {k} mismatch exact optimum: ratings={ratings}, "
            f"greedy={total}, exact={exact_total}"
        )

        print(f"[{k}] ratings={ratings} -> candies={candies}, total={total}")


def main() -> None:
    run_fixed_cases()
    run_random_small_verification()
    print("\nAll checks passed for CS-0071 (分发糖果问题).")


if __name__ == "__main__":
    main()

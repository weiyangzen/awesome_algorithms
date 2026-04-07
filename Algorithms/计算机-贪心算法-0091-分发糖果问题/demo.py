"""CS-0071 分发糖果问题：贪心算法最小可运行 MVP。

运行:
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class FixedCase:
    """Deterministic test case for the candy problem."""

    name: str
    ratings: list[int]
    expected_total: int
    expected_distribution: list[int] | None = None


def _normalize_ratings(ratings: Iterable[int]) -> list[int]:
    """Normalize ratings into a plain integer list with basic validation."""
    normalized: list[int] = []
    for idx, value in enumerate(ratings):
        if isinstance(value, bool):
            raise TypeError(f"ratings[{idx}] must be an int, got bool")
        normalized.append(int(value))
    return normalized


def _normalize_distribution(candies: Iterable[int]) -> list[int]:
    """Normalize candy distribution and require positive integers."""
    normalized: list[int] = []
    for idx, value in enumerate(candies):
        if isinstance(value, bool):
            raise TypeError(f"candies[{idx}] must be an int, got bool")
        candy = int(value)
        if candy < 1:
            raise ValueError(f"candies[{idx}] must be >= 1, got {candy}")
        normalized.append(candy)
    return normalized


def is_distribution_valid(ratings: Iterable[int], candies: Iterable[int]) -> bool:
    """Check whether a candy distribution satisfies all problem constraints."""
    r = _normalize_ratings(ratings)
    c = _normalize_distribution(candies)

    if len(r) != len(c):
        return False

    for i in range(len(r) - 1):
        if r[i] < r[i + 1] and not (c[i] < c[i + 1]):
            return False
        if r[i] > r[i + 1] and not (c[i] > c[i + 1]):
            return False

    return True


def candy_distribution_greedy(ratings: Iterable[int]) -> list[int]:
    """Two-pass greedy construction of a minimum feasible candy distribution."""
    values = _normalize_ratings(ratings)
    n = len(values)

    if n == 0:
        return []

    candies = [1] * n

    # Pass 1: satisfy left-neighbor constraints.
    for i in range(1, n):
        if values[i] > values[i - 1]:
            candies[i] = candies[i - 1] + 1

    # Pass 2: satisfy right-neighbor constraints without breaking pass-1.
    for i in range(n - 2, -1, -1):
        if values[i] > values[i + 1]:
            candies[i] = max(candies[i], candies[i + 1] + 1)

    return candies


def min_candies_greedy(ratings: Iterable[int]) -> int:
    """Return the minimal candy count computed by the greedy method."""
    return sum(candy_distribution_greedy(ratings))


def min_candies_bruteforce(ratings: Sequence[int], max_n: int = 8) -> tuple[int, list[int]]:
    """Exact baseline via DFS enumeration (small n only)."""
    values = _normalize_ratings(ratings)
    n = len(values)

    if n == 0:
        return 0, []
    if n > max_n:
        raise ValueError(f"Bruteforce is limited to n <= {max_n}, got n={n}")

    greedy_dist = candy_distribution_greedy(values)
    best_sum = sum(greedy_dist)
    best_dist = greedy_dist.copy()

    current = [0] * n

    def dfs(i: int, partial_sum: int) -> None:
        nonlocal best_sum, best_dist

        if partial_sum >= best_sum:
            return

        if i == n:
            if is_distribution_valid(values, current) and partial_sum < best_sum:
                best_sum = partial_sum
                best_dist = current.copy()
            return

        low = 1
        high = min(best_sum - partial_sum - (n - i - 1), n)

        if i > 0:
            if values[i] > values[i - 1]:
                low = max(low, current[i - 1] + 1)
            elif values[i] < values[i - 1]:
                high = min(high, current[i - 1] - 1)

        if low > high:
            return

        for candy in range(low, high + 1):
            current[i] = candy
            dfs(i + 1, partial_sum + candy)
            current[i] = 0

    dfs(0, 0)
    return best_sum, best_dist


def _run_fixed_cases() -> None:
    print("=== Fixed Cases ===")
    cases = [
        FixedCase(
            name="leetcode canonical 1",
            ratings=[1, 0, 2],
            expected_total=5,
            expected_distribution=[2, 1, 2],
        ),
        FixedCase(
            name="leetcode canonical 2",
            ratings=[1, 2, 2],
            expected_total=4,
        ),
        FixedCase(
            name="strictly increasing",
            ratings=[1, 2, 3, 4],
            expected_total=10,
            expected_distribution=[1, 2, 3, 4],
        ),
        FixedCase(
            name="strictly decreasing",
            ratings=[4, 3, 2, 1],
            expected_total=10,
            expected_distribution=[4, 3, 2, 1],
        ),
        FixedCase(
            name="plateau with peaks",
            ratings=[1, 2, 87, 87, 87, 2, 1],
            expected_total=13,
        ),
        FixedCase(
            name="single child",
            ratings=[9],
            expected_total=1,
            expected_distribution=[1],
        ),
        FixedCase(
            name="empty",
            ratings=[],
            expected_total=0,
            expected_distribution=[],
        ),
    ]

    for i, case in enumerate(cases, start=1):
        distribution = candy_distribution_greedy(case.ratings)
        total = sum(distribution)

        assert total == case.expected_total, (
            f"Case '{case.name}' total mismatch: expected={case.expected_total}, got={total}"
        )
        assert is_distribution_valid(case.ratings, distribution), (
            f"Case '{case.name}' produced invalid distribution: {distribution}"
        )

        if case.expected_distribution is not None:
            assert distribution == case.expected_distribution, (
                f"Case '{case.name}' distribution mismatch: "
                f"expected={case.expected_distribution}, got={distribution}"
            )

        if len(case.ratings) <= 8:
            brute_total, brute_distribution = min_candies_bruteforce(case.ratings)
            assert total == brute_total, (
                f"Case '{case.name}' mismatch with bruteforce: greedy={total}, brute={brute_total}"
            )
            assert is_distribution_valid(case.ratings, brute_distribution)

        print(f"[{i}] {case.name}: ratings={case.ratings}, candies={distribution}, total={total}")


def _run_random_regression(seed: int = 71, rounds: int = 50) -> None:
    """Randomized checks against bruteforce on small inputs and validity on larger ones."""
    print("\n=== Random Regression ===")
    rng = np.random.default_rng(seed)

    checked_small = 0
    checked_large = 0

    for n in range(1, 13):
        for _ in range(rounds):
            ratings = rng.integers(low=-5, high=20, size=n).tolist()
            greedy_distribution = candy_distribution_greedy(ratings)
            greedy_total = sum(greedy_distribution)

            assert is_distribution_valid(ratings, greedy_distribution), (
                f"Invalid greedy distribution for ratings={ratings}: {greedy_distribution}"
            )
            assert greedy_total == min_candies_greedy(ratings)

            if n <= 8:
                brute_total, brute_distribution = min_candies_bruteforce(ratings)
                assert greedy_total == brute_total, (
                    f"Random mismatch n={n}, ratings={ratings}: "
                    f"greedy={greedy_total}, brute={brute_total}"
                )
                assert is_distribution_valid(ratings, brute_distribution)
                checked_small += 1
            else:
                checked_large += 1

    numpy_case = np.array([3, 3, 1, 2, 2, 4, 1])
    numpy_distribution = candy_distribution_greedy(numpy_case)
    assert is_distribution_valid(numpy_case.tolist(), numpy_distribution)

    print(
        f"small_cases_with_bruteforce={checked_small}, "
        f"large_cases_validated={checked_large}, seed={seed}"
    )


def _run_perf_snapshot(seed: int = 2026, n: int = 200_000) -> None:
    """Simple performance snapshot on a long random rating sequence."""
    print("\n=== Performance Snapshot ===")
    rng = np.random.default_rng(seed)
    ratings = rng.integers(low=0, high=10_000, size=n)

    t0 = perf_counter()
    distribution = candy_distribution_greedy(ratings)
    total = sum(distribution)
    t1 = perf_counter()

    assert len(distribution) == n
    assert total >= n
    assert is_distribution_valid(ratings.tolist(), distribution)

    print(f"n={n}, total={total}, greedy_time={t1 - t0:.6f}s")


def main() -> None:
    _run_fixed_cases()
    _run_random_regression()
    _run_perf_snapshot()
    print("\nAll checks passed for CS-0071 (分发糖果问题).")


if __name__ == "__main__":
    main()

"""Greedy MVP for CS-0070: 买卖股票的最佳时机II.

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
    """Deterministic test case for stock-trading profit."""

    name: str
    prices: list[int]
    expected: int


def _normalize_prices(prices: Iterable[int]) -> list[int]:
    """Convert input to a validated list of non-negative integer prices."""
    normalized: list[int] = []
    for idx, price in enumerate(prices):
        value = int(price)
        if value < 0:
            raise ValueError(f"Price at index {idx} must be non-negative, got {value}.")
        normalized.append(value)
    return normalized


def max_profit_greedy(prices: Iterable[int]) -> int:
    """Compute max profit using greedy positive-delta accumulation."""
    values = _normalize_prices(prices)
    profit = 0
    for prev_price, curr_price in zip(values, values[1:]):
        if curr_price > prev_price:
            profit += curr_price - prev_price
    return profit


def max_profit_dp(prices: Iterable[int]) -> int:
    """Reference solution using two-state dynamic programming."""
    values = _normalize_prices(prices)
    if not values:
        return 0

    cash = 0
    hold = -values[0]
    for price in values[1:]:
        next_cash = max(cash, hold + price)
        next_hold = max(hold, cash - price)
        cash, hold = next_cash, next_hold

    return cash


def max_profit_bruteforce(prices: Iterable[int]) -> int:
    """Exhaustive search for small arrays, used for verification only."""
    values = _normalize_prices(prices)
    n = len(values)
    neg_inf = -10**15

    @lru_cache(maxsize=None)
    def dfs(day: int, holding: int) -> int:
        if day == n:
            return 0 if holding == 0 else neg_inf

        stay = dfs(day + 1, holding)
        price = values[day]
        if holding:
            sell = price + dfs(day + 1, 0)
            return max(stay, sell)

        buy = -price + dfs(day + 1, 1)
        return max(stay, buy)

    return dfs(0, 0)


def max_profit_numpy(prices_array: np.ndarray) -> int:
    """Bridge function: accept 1D numpy array and run greedy solver."""
    arr = np.asarray(prices_array)
    if arr.ndim != 1:
        raise ValueError(f"Expected a 1D numpy array, got shape {arr.shape}")
    return max_profit_greedy(arr.tolist())


def assert_consistency(prices: Sequence[int]) -> None:
    """Check greedy answer against DP and brute force baselines."""
    greedy = max_profit_greedy(prices)
    dp = max_profit_dp(prices)
    assert greedy == dp, f"Greedy != DP for prices={prices}: greedy={greedy}, dp={dp}"

    if len(prices) <= 12:
        brute = max_profit_bruteforce(prices)
        assert greedy == brute, (
            f"Greedy != brute force for prices={prices}: greedy={greedy}, brute={brute}"
        )


def run_fixed_cases() -> None:
    cases = [
        FixedCase(name="example 1", prices=[7, 1, 5, 3, 6, 4], expected=7),
        FixedCase(name="monotonic up", prices=[1, 2, 3, 4, 5], expected=4),
        FixedCase(name="monotonic down", prices=[7, 6, 4, 3, 1], expected=0),
        FixedCase(name="single day", prices=[5], expected=0),
        FixedCase(name="zigzag", prices=[2, 1, 2, 0, 1], expected=2),
        FixedCase(name="empty input", prices=[], expected=0),
    ]

    print("=== Fixed Cases ===")
    for i, case in enumerate(cases, start=1):
        got = max_profit_greedy(case.prices)
        assert got == case.expected, (
            f"Case {case.name} failed: expected={case.expected}, got={got}"
        )
        assert_consistency(case.prices)
        print(f"[{i}] {case.name}: prices={case.prices} -> profit={got}")


def run_random_verification() -> None:
    rng = np.random.default_rng(2026)
    total = 300

    for _ in range(total):
        n = int(rng.integers(1, 10))
        prices = rng.integers(0, 20, size=n).tolist()
        assert_consistency(prices)

    print(f"\nRandom verification passed: {total} cases.")


def run_numpy_case() -> None:
    rng = np.random.default_rng(90)
    prices_array = rng.integers(0, 15, size=10)
    profit = max_profit_numpy(prices_array)

    print("\n=== Numpy Case ===")
    print(f"prices array: {prices_array}")
    print(f"greedy profit: {profit}")


def main() -> None:
    run_fixed_cases()
    run_random_verification()
    run_numpy_case()
    print("\nAll checks passed for CS-0070 (买卖股票的最佳时机II).")


if __name__ == "__main__":
    main()

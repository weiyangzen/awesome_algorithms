"""Coin Change Problem MVP.

Run:
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from random import Random

import numpy as np


@dataclass
class CoinChangeResult:
    min_coins: int | None
    combination: list[int]
    ways: int


def normalize_coins(coins: list[int]) -> list[int]:
    """Validate and normalize coin denominations."""
    if not isinstance(coins, list):
        raise TypeError(f"coins must be list[int], got {type(coins).__name__}")
    if not coins:
        raise ValueError("coins must not be empty")
    if any((not isinstance(c, int) or c <= 0) for c in coins):
        raise ValueError("each coin must be a positive integer")
    return sorted(set(coins))


def validate_amount(amount: int) -> int:
    if not isinstance(amount, int):
        raise TypeError(f"amount must be int, got {type(amount).__name__}")
    if amount < 0:
        raise ValueError("amount must be non-negative")
    return amount


def min_coins_dp(coins: list[int], amount: int) -> tuple[int | None, list[int]]:
    """Return minimum coin count and one optimal combination.

    State:
        dp[a] = minimum number of coins needed to make amount a
    Transition:
        dp[a] = min(dp[a], dp[a - coin] + 1)
    """
    c = normalize_coins(coins)
    target = validate_amount(amount)

    inf = target + 1
    dp = np.full(target + 1, inf, dtype=np.int64)
    pick = np.full(target + 1, -1, dtype=np.int64)
    dp[0] = 0

    for coin in c:
        for a in range(coin, target + 1):
            candidate = int(dp[a - coin]) + 1
            if candidate < int(dp[a]):
                dp[a] = candidate
                pick[a] = coin

    if int(dp[target]) >= inf:
        return None, []

    combination: list[int] = []
    cur = target
    while cur > 0:
        coin = int(pick[cur])
        if coin <= 0:
            raise RuntimeError("reconstruction failed: pick table is inconsistent")
        combination.append(coin)
        cur -= coin

    return int(dp[target]), combination


def count_ways_dp(coins: list[int], amount: int) -> int:
    """Count combinations (order-independent) to make target amount.

    State:
        ways[a] = number of combinations to form amount a
    Transition:
        ways[a] += ways[a - coin]
    """
    c = normalize_coins(coins)
    target = validate_amount(amount)

    ways = np.zeros(target + 1, dtype=np.int64)
    ways[0] = 1

    for coin in c:
        for a in range(coin, target + 1):
            ways[a] += ways[a - coin]

    return int(ways[target])


def solve_coin_change(coins: list[int], amount: int) -> CoinChangeResult:
    min_count, combo = min_coins_dp(coins, amount)
    ways = count_ways_dp(coins, amount)
    return CoinChangeResult(min_coins=min_count, combination=combo, ways=ways)


def brute_force_min_and_ways(coins: list[int], amount: int) -> tuple[int | None, int]:
    """Exact solver for small instances (validation only)."""
    c = tuple(normalize_coins(coins))
    target = validate_amount(amount)
    inf = 10**9

    @lru_cache(maxsize=None)
    def dfs(idx: int, remain: int) -> tuple[int, int]:
        if remain == 0:
            return 0, 1
        if idx == len(c):
            return inf, 0

        coin = c[idx]
        best = inf
        ways = 0
        max_k = remain // coin
        for k in range(max_k + 1):
            sub_best, sub_ways = dfs(idx + 1, remain - k * coin)
            if sub_ways == 0:
                continue
            ways += sub_ways
            if sub_best < inf:
                best = min(best, sub_best + k)
        return best, ways

    min_count, ways = dfs(0, target)
    return (None if min_count >= inf else min_count), ways


def run_case(
    name: str,
    coins: list[int],
    amount: int,
    expected_min: int | None = None,
    expected_ways: int | None = None,
) -> None:
    result = solve_coin_change(coins, amount)
    brute_min, brute_ways = brute_force_min_and_ways(coins, amount)
    coin_set = set(normalize_coins(coins))

    if result.min_coins is None:
        combo_sum_ok = result.combination == []
        combo_len_ok = result.combination == []
    else:
        combo_sum_ok = sum(result.combination) == amount
        combo_len_ok = len(result.combination) == result.min_coins
    combo_coin_ok = all(coin in coin_set for coin in result.combination)

    print(f"=== {name} ===")
    print(f"coins               = {normalize_coins(coins)}")
    print(f"amount              = {amount}")
    print(
        "dp_result           -> "
        f"min_coins={result.min_coins}, combination={result.combination}, ways={result.ways}"
    )
    print(f"bruteforce_check    -> min_coins={brute_min}, ways={brute_ways}")
    print(
        "checks              -> "
        f"sum_match={combo_sum_ok}, len_match={combo_len_ok}, "
        f"coin_domain_ok={combo_coin_ok}, min_match={result.min_coins == brute_min}, "
        f"ways_match={result.ways == brute_ways}"
    )
    print()

    if expected_min is not None and result.min_coins != expected_min:
        raise AssertionError(
            f"expected_min mismatch in {name}: {result.min_coins} != {expected_min}"
        )
    if expected_ways is not None and result.ways != expected_ways:
        raise AssertionError(
            f"expected_ways mismatch in {name}: {result.ways} != {expected_ways}"
        )

    if not combo_sum_ok or not combo_len_ok or not combo_coin_ok:
        raise AssertionError(f"invalid combination reconstruction in {name}")
    if result.min_coins != brute_min:
        raise AssertionError(f"min_coins cross-check failed in {name}")
    if result.ways != brute_ways:
        raise AssertionError(f"ways cross-check failed in {name}")


def randomized_regression(seed: int = 2026, rounds: int = 200) -> None:
    rng = Random(seed)
    for _ in range(rounds):
        denom_count = rng.randint(1, 5)
        coins = sorted(rng.sample(range(1, 10), denom_count))
        amount = rng.randint(0, 30)

        result = solve_coin_change(coins, amount)
        brute_min, brute_ways = brute_force_min_and_ways(coins, amount)

        assert result.min_coins == brute_min
        assert result.ways == brute_ways
        if result.min_coins is not None:
            assert len(result.combination) == result.min_coins
            assert sum(result.combination) == amount

    print(
        "randomized regression passed: "
        f"seed={seed}, rounds={rounds}, coin_count=[1,5], amount=[0,30]"
    )


def main() -> None:
    cases = [
        ("Case 1: classic", [1, 2, 5], 11, 3, 11),
        ("Case 2: impossible", [2], 3, None, 0),
        ("Case 3: medium", [1, 3, 4], 6, 2, 4),
        ("Case 4: unsorted input", [2, 5, 10, 1], 27, 4, None),
        ("Case 5: zero amount", [7, 10], 0, 0, 1),
    ]

    for name, coins, amount, expected_min, expected_ways in cases:
        run_case(name, coins, amount, expected_min, expected_ways)

    randomized_regression()

    print("All coin-change checks passed.")


if __name__ == "__main__":
    main()

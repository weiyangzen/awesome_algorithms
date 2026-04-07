"""戳气球问题 MVP：区间 DP + 递归基线 + 小规模暴力对拍。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np


@dataclass
class BurstBalloonsResult:
    n: int
    max_coins: int
    burst_order: list[int]


def to_balloon_array(nums: Sequence[int] | np.ndarray) -> np.ndarray:
    arr = np.asarray(nums, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"nums must be a 1D sequence, got shape={arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("nums contains non-finite values")
    if np.any(arr < 0):
        raise ValueError("nums must be non-negative")
    rounded = np.rint(arr)
    if not np.allclose(arr, rounded):
        raise ValueError("nums must contain integer-like values in this MVP")
    return rounded.astype(np.int64)


def burst_balloons_dp(nums: Sequence[int] | np.ndarray) -> BurstBalloonsResult:
    """区间 DP（按最后戳破建模），返回最大收益与一组最优顺序。"""
    values = to_balloon_array(nums)
    n = int(values.size)
    if n == 0:
        return BurstBalloonsResult(n=0, max_coins=0, burst_order=[])

    arr = np.concatenate((np.array([1], dtype=np.int64), values, np.array([1], dtype=np.int64)))
    m = int(arr.size)

    dp = np.zeros((m, m), dtype=np.int64)
    choice = np.full((m, m), -1, dtype=np.int64)

    for gap in range(2, m):
        for left in range(0, m - gap):
            right = left + gap
            best_val = -1
            best_k = -1
            for k in range(left + 1, right):
                candidate = (
                    int(dp[left, k])
                    + int(dp[k, right])
                    + int(arr[left] * arr[k] * arr[right])
                )
                if candidate > best_val:
                    best_val = candidate
                    best_k = k
            dp[left, right] = max(0, best_val)
            choice[left, right] = best_k

    order: list[int] = []

    def rebuild(left: int, right: int) -> None:
        if right <= left + 1:
            return
        k = int(choice[left, right])
        if k <= left or k >= right:
            raise RuntimeError(
                f"invalid split when rebuilding: left={left}, right={right}, k={k}"
            )
        rebuild(left, k)
        rebuild(k, right)
        order.append(k - 1)

    rebuild(0, m - 1)

    return BurstBalloonsResult(
        n=n,
        max_coins=int(dp[0, m - 1]),
        burst_order=order,
    )


def burst_balloons_top_down(nums: Sequence[int] | np.ndarray) -> int:
    """记忆化递归基线，只返回最大收益。"""
    values = to_balloon_array(nums)
    n = int(values.size)
    if n == 0:
        return 0

    arr = np.concatenate((np.array([1], dtype=np.int64), values, np.array([1], dtype=np.int64)))
    m = int(arr.size)

    @lru_cache(maxsize=None)
    def solve(left: int, right: int) -> int:
        if right <= left + 1:
            return 0
        best = 0
        for k in range(left + 1, right):
            candidate = solve(left, k) + solve(k, right) + int(arr[left] * arr[k] * arr[right])
            if candidate > best:
                best = candidate
        return best

    return solve(0, m - 1)


def burst_balloons_bruteforce(nums: Sequence[int] | np.ndarray) -> int:
    """暴力枚举戳破顺序，仅用于小规模校验。"""
    values = to_balloon_array(nums)

    @lru_cache(maxsize=None)
    def solve(state: tuple[int, ...]) -> int:
        size = len(state)
        if size == 0:
            return 0
        best = 0
        for i in range(size):
            left = 1 if i == 0 else state[i - 1]
            right = 1 if i == size - 1 else state[i + 1]
            gain = left * state[i] * right
            nxt = state[:i] + state[i + 1 :]
            candidate = gain + solve(nxt)
            if candidate > best:
                best = candidate
        return best

    return solve(tuple(int(x) for x in values.tolist()))


def simulate_order_gain(nums: Sequence[int] | np.ndarray, order: Sequence[int]) -> int:
    """按给定原始下标顺序回放戳气球过程，返回总收益。"""
    values = to_balloon_array(nums)
    n = int(values.size)
    if n == 0:
        if len(order) != 0:
            raise ValueError("order must be empty when nums is empty")
        return 0
    if len(order) != n:
        raise ValueError(f"order length mismatch: got {len(order)}, expected {n}")

    alive = list(range(n))
    popped = set()
    total = 0

    for idx in order:
        if idx < 0 or idx >= n:
            raise ValueError(f"order contains invalid index: {idx}")
        if idx in popped:
            raise ValueError(f"order contains duplicate index: {idx}")

        pos = alive.index(idx)
        left_val = 1 if pos == 0 else int(values[alive[pos - 1]])
        right_val = 1 if pos == len(alive) - 1 else int(values[alive[pos + 1]])
        total += left_val * int(values[idx]) * right_val

        alive.pop(pos)
        popped.add(idx)

    return int(total)


def run_case(
    name: str,
    nums: Sequence[int],
    expected: int | None = None,
    brute_force_limit: int = 8,
) -> None:
    result = burst_balloons_dp(nums)
    top_down = burst_balloons_top_down(nums)
    brute_force = None
    if len(nums) <= brute_force_limit:
        brute_force = burst_balloons_bruteforce(nums)

    simulated = simulate_order_gain(nums, result.burst_order)

    print(f"=== {name} ===")
    print(f"nums={list(nums)}")
    print(f"max_coins={result.max_coins}, n={result.n}")
    print(f"order={result.burst_order}")
    print(
        "cross-check => "
        f"simulate_gain={simulated}, "
        f"top_down={top_down}, "
        f"brute_force={('N/A' if brute_force is None else brute_force)}\n"
    )

    if result.max_coins != simulated:
        raise AssertionError("DP result and simulated order gain mismatch")
    if result.max_coins != top_down:
        raise AssertionError("DP result and top-down baseline mismatch")
    if brute_force is not None and result.max_coins != brute_force:
        raise AssertionError("DP result and brute-force baseline mismatch")
    if expected is not None and result.max_coins != expected:
        raise AssertionError(
            f"unexpected result: got {result.max_coins}, expected {expected}"
        )


def randomized_cross_check(
    trials: int = 250,
    max_n: int = 7,
    max_value: int = 9,
    seed: int = 2026,
) -> None:
    rng = np.random.default_rng(seed)
    for _ in range(trials):
        n = int(rng.integers(0, max_n + 1))
        nums = rng.integers(0, max_value + 1, size=n).astype(np.int64)

        result = burst_balloons_dp(nums)
        top_down = burst_balloons_top_down(nums)
        brute = burst_balloons_bruteforce(nums)
        simulated = simulate_order_gain(nums, result.burst_order)

        if result.max_coins != top_down:
            raise AssertionError("random check failed: DP vs top-down mismatch")
        if result.max_coins != brute:
            raise AssertionError("random check failed: DP vs brute-force mismatch")
        if result.max_coins != simulated:
            raise AssertionError("random check failed: DP vs simulated order mismatch")

    print(
        f"Randomized cross-check passed: {trials} trials "
        f"(max_n={max_n}, max_value={max_value}, seed={seed})."
    )


def main() -> None:
    run_case("Case 1: classic", [3, 1, 5, 8], expected=167)
    run_case("Case 2: two balloons", [1, 5], expected=10)
    run_case("Case 3: single balloon", [9], expected=9)
    run_case("Case 4: empty", [], expected=0)
    run_case("Case 5: with zero", [2, 0, 4])

    randomized_cross_check(trials=250, max_n=7, max_value=9, seed=2026)


if __name__ == "__main__":
    main()

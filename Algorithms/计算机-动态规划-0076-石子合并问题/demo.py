"""石子合并问题 MVP：区间 DP + 记忆化基线 + 小规模暴力对拍。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np


@dataclass
class MergeOperation:
    left: tuple[int, int]
    right: tuple[int, int]
    merged: tuple[int, int]
    merge_cost: float


@dataclass
class StoneMergeResult:
    n: int
    min_cost: float
    merge_plan: list[MergeOperation]


def to_weight_array(weights: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(weights, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"weights must be a 1D sequence, got shape={arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("weights contains non-finite values")
    if np.any(arr < 0):
        raise ValueError("weights must be non-negative in this MVP")
    return arr


def prefix_sums(weights: np.ndarray) -> np.ndarray:
    prefix = np.zeros(weights.size + 1, dtype=float)
    prefix[1:] = np.cumsum(weights)
    return prefix


def range_sum(prefix: np.ndarray, i: int, j: int) -> float:
    return float(prefix[j + 1] - prefix[i])


def stone_merge_min_cost(weights: Sequence[float] | np.ndarray) -> StoneMergeResult:
    """自底向上区间 DP，返回最小代价和一组最优合并步骤。"""
    w = to_weight_array(weights)
    n = int(w.size)
    if n <= 1:
        return StoneMergeResult(n=n, min_cost=0.0, merge_plan=[])

    prefix = prefix_sums(w)
    dp = np.full((n, n), np.inf, dtype=float)
    split = np.full((n, n), -1, dtype=int)

    for i in range(n):
        dp[i, i] = 0.0

    for length in range(2, n + 1):
        for i in range(0, n - length + 1):
            j = i + length - 1
            total = range_sum(prefix, i, j)
            best = np.inf
            best_k = -1
            for k in range(i, j):
                candidate = float(dp[i, k] + dp[k + 1, j] + total)
                if candidate < best:
                    best = candidate
                    best_k = k
            dp[i, j] = best
            split[i, j] = best_k

    merge_plan: list[MergeOperation] = []

    def rebuild(i: int, j: int) -> tuple[int, int]:
        if i == j:
            return (i, j)
        k = int(split[i, j])
        if k < i or k >= j:
            raise RuntimeError(f"invalid split when rebuilding: i={i}, j={j}, k={k}")
        left = rebuild(i, k)
        right = rebuild(k + 1, j)
        merged = (i, j)
        merge_plan.append(
            MergeOperation(
                left=left,
                right=right,
                merged=merged,
                merge_cost=range_sum(prefix, i, j),
            )
        )
        return merged

    rebuild(0, n - 1)
    return StoneMergeResult(n=n, min_cost=float(dp[0, n - 1]), merge_plan=merge_plan)


def stone_merge_min_cost_top_down(weights: Sequence[float] | np.ndarray) -> float:
    """记忆化递归基线，只返回最小代价。"""
    w = to_weight_array(weights)
    n = int(w.size)
    if n <= 1:
        return 0.0

    prefix = prefix_sums(w)

    @lru_cache(maxsize=None)
    def solve(i: int, j: int) -> float:
        if i >= j:
            return 0.0
        total = range_sum(prefix, i, j)
        ans = np.inf
        for k in range(i, j):
            candidate = solve(i, k) + solve(k + 1, j) + total
            if candidate < ans:
                ans = candidate
        return float(ans)

    return solve(0, n - 1)


def brute_force_min_cost_adjacent(weights: Sequence[float] | np.ndarray) -> float:
    """暴力枚举相邻合并顺序，仅用于小规模正确性校验。"""
    w = to_weight_array(weights)

    @lru_cache(maxsize=None)
    def solve(state: tuple[float, ...]) -> float:
        m = len(state)
        if m <= 1:
            return 0.0
        best = np.inf
        for i in range(m - 1):
            merged_weight = state[i] + state[i + 1]
            nxt = state[:i] + (merged_weight,) + state[i + 2 :]
            candidate = merged_weight + solve(nxt)
            if candidate < best:
                best = candidate
        return float(best)

    return solve(tuple(float(x) for x in w.tolist()))


def format_plan(plan: Sequence[MergeOperation]) -> str:
    if not plan:
        return "[]"
    parts = []
    for op in plan:
        parts.append(
            f"{op.left}+{op.right}->{op.merged}(cost={op.merge_cost:.0f})"
        )
    return "[" + ", ".join(parts) + "]"


def run_case(
    name: str,
    weights: Sequence[float],
    expected: float | None = None,
    brute_force_limit: int = 8,
) -> None:
    result = stone_merge_min_cost(weights)
    baseline = stone_merge_min_cost_top_down(weights)
    brute_force = None
    if len(weights) <= brute_force_limit:
        brute_force = brute_force_min_cost_adjacent(weights)

    print(f"=== {name} ===")
    print(f"weights={list(weights)}")
    print(f"min_cost={result.min_cost:.2f}, n={result.n}")
    print(f"plan={format_plan(result.merge_plan)}")
    print(
        "cross-check => "
        f"top_down={baseline:.2f}, "
        f"brute_force={('N/A' if brute_force is None else f'{brute_force:.2f}')}\n"
    )

    if abs(result.min_cost - baseline) > 1e-9:
        raise AssertionError("bottom-up DP and top-down baseline mismatch")
    if brute_force is not None and abs(result.min_cost - brute_force) > 1e-9:
        raise AssertionError("bottom-up DP and brute-force baseline mismatch")
    if expected is not None and abs(result.min_cost - expected) > 1e-9:
        raise AssertionError(
            f"unexpected min_cost: got {result.min_cost}, expected {expected}"
        )


def randomized_cross_check(
    trials: int = 200,
    max_n: int = 7,
    max_weight: int = 20,
    seed: int = 2026,
) -> None:
    rng = np.random.default_rng(seed)
    for _ in range(trials):
        n = int(rng.integers(0, max_n + 1))
        weights = rng.integers(0, max_weight + 1, size=n).astype(float)

        result = stone_merge_min_cost(weights)
        baseline = stone_merge_min_cost_top_down(weights)
        brute = brute_force_min_cost_adjacent(weights)

        if abs(result.min_cost - baseline) > 1e-9:
            raise AssertionError("random check failed: DP vs top-down mismatch")
        if abs(result.min_cost - brute) > 1e-9:
            raise AssertionError("random check failed: DP vs brute-force mismatch")

    print(
        f"Randomized cross-check passed: {trials} trials "
        f"(max_n={max_n}, max_weight={max_weight}, seed={seed})."
    )


def main() -> None:
    run_case("Case 1: classic", [4, 1, 1, 4], expected=18.0)
    run_case("Case 2: three piles", [3, 5, 1], expected=15.0)
    run_case("Case 3: single pile", [10], expected=0.0)
    run_case("Case 4: empty", [], expected=0.0)
    run_case("Case 5: general", [1, 3, 5, 2])

    randomized_cross_check(trials=200, max_n=7, max_weight=20, seed=2026)


if __name__ == "__main__":
    main()

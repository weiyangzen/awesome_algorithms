"""最大子数组和（Kadane）最小可运行 MVP。

实现内容：
1) Kadane 线性时间动态规划主算法；
2) 暴力基线（前缀和 + 双层枚举）用于交叉校验；
3) 固定样例 + 随机对拍（无交互输入）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class MaxSubarrayResult:
    max_sum: float
    start_index: int
    end_index: int
    subarray: list[float]


def to_1d_float_array(nums: Sequence[float] | np.ndarray) -> np.ndarray:
    """将输入转换为一维 float64 数组，并做合法性校验。"""
    arr = np.asarray(nums, dtype=np.float64)

    if arr.ndim != 1:
        raise ValueError(f"nums must be 1D, got ndim={arr.ndim}")
    if arr.size == 0:
        raise ValueError("nums must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("nums contains NaN or Inf")

    return arr


def _is_better_interval(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    """比较两个同和区间是否更优。

    规则：起点更小优先；若起点相同，长度更短优先。
    返回 True 表示 A 区间优于 B 区间。
    """
    if start_a != start_b:
        return start_a < start_b
    return (end_a - start_a) < (end_b - start_b)


def kadane_max_subarray(nums: Sequence[float] | np.ndarray) -> MaxSubarrayResult:
    """Kadane 算法，时间 O(n)，空间 O(1)。"""
    arr = to_1d_float_array(nums)

    current_sum = float(arr[0])
    best_sum = float(arr[0])

    current_start = 0
    best_start = 0
    best_end = 0

    for i in range(1, arr.size):
        x = float(arr[i])

        # 若累计和为负，继续扩展只会更差，直接从当前位置重启。
        if current_sum < 0.0:
            current_sum = x
            current_start = i
        else:
            current_sum += x

        if current_sum > best_sum:
            best_sum = current_sum
            best_start = current_start
            best_end = i
        elif np.isclose(current_sum, best_sum):
            if _is_better_interval(current_start, i, best_start, best_end):
                best_start = current_start
                best_end = i

    return MaxSubarrayResult(
        max_sum=best_sum,
        start_index=best_start,
        end_index=best_end,
        subarray=arr[best_start : best_end + 1].tolist(),
    )


def bruteforce_max_subarray(nums: Sequence[float] | np.ndarray) -> MaxSubarrayResult:
    """前缀和 + 双层枚举，时间 O(n^2)，用于校验主算法。"""
    arr = to_1d_float_array(nums)
    n = arr.size

    prefix = np.zeros(n + 1, dtype=np.float64)
    prefix[1:] = np.cumsum(arr)

    best_sum = -np.inf
    best_start = 0
    best_end = 0

    for start in range(n):
        for end in range(start, n):
            sub_sum = float(prefix[end + 1] - prefix[start])
            if sub_sum > best_sum:
                best_sum = sub_sum
                best_start = start
                best_end = end
            elif np.isclose(sub_sum, best_sum):
                if _is_better_interval(start, end, best_start, best_end):
                    best_start = start
                    best_end = end

    return MaxSubarrayResult(
        max_sum=float(best_sum),
        start_index=best_start,
        end_index=best_end,
        subarray=arr[best_start : best_end + 1].tolist(),
    )


def run_case(nums: Sequence[float] | np.ndarray, expected_sum: float | None = None) -> MaxSubarrayResult:
    fast = kadane_max_subarray(nums)
    slow = bruteforce_max_subarray(nums)

    if not np.isclose(fast.max_sum, slow.max_sum):
        raise AssertionError(
            f"max_sum mismatch: kadane={fast.max_sum}, bruteforce={slow.max_sum}, nums={list(nums)}"
        )

    if (fast.start_index, fast.end_index) != (slow.start_index, slow.end_index):
        raise AssertionError(
            "interval mismatch: "
            f"kadane=({fast.start_index},{fast.end_index}), "
            f"bruteforce=({slow.start_index},{slow.end_index}), nums={list(nums)}"
        )

    if expected_sum is not None and not np.isclose(fast.max_sum, expected_sum):
        raise AssertionError(
            f"unexpected max_sum: got {fast.max_sum}, expected {expected_sum}, nums={list(nums)}"
        )

    print(f"nums={list(nums)}")
    print(
        f"max_sum={fast.max_sum:.6g}, range=[{fast.start_index}, {fast.end_index}], "
        f"subarray={fast.subarray}, length={len(nums)}"
    )
    print("cross-check: kadane == bruteforce -> True\n")

    return fast


def randomized_cross_check(trials: int = 300, seed: int = 2026) -> None:
    rng = np.random.default_rng(seed)

    for _ in range(trials):
        n = int(rng.integers(1, 31))
        arr = rng.integers(-20, 21, size=n, dtype=np.int32).astype(np.float64)

        fast = kadane_max_subarray(arr)
        slow = bruteforce_max_subarray(arr)

        if not np.isclose(fast.max_sum, slow.max_sum):
            raise AssertionError(
                f"random mismatch(sum): kadane={fast.max_sum}, bruteforce={slow.max_sum}, arr={arr.tolist()}"
            )

        if (fast.start_index, fast.end_index) != (slow.start_index, slow.end_index):
            raise AssertionError(
                "random mismatch(interval): "
                f"kadane=({fast.start_index},{fast.end_index}), "
                f"bruteforce=({slow.start_index},{slow.end_index}), arr={arr.tolist()}"
            )

    print(f"Randomized cross-check passed: {trials} trials (seed={seed}).")


def main() -> None:
    print("Maximum Subarray Sum MVP (Kadane Dynamic Programming)")

    cases: list[tuple[list[int], float]] = [
        ([-2, 1, -3, 4, -1, 2, 1, -5, 4], 6.0),
        ([1, 2, 3, 4], 10.0),
        ([-8, -3, -6, -2, -5, -4], -2.0),
        ([0, -1, 0, 2, -2, 3], 3.0),
        ([7], 7.0),
    ]

    results: list[MaxSubarrayResult] = []
    for nums, expected in cases:
        results.append(run_case(nums, expected_sum=expected))

    randomized_cross_check(trials=300, seed=2026)

    max_of_max = max(item.max_sum for item in results)
    min_of_max = min(item.max_sum for item in results)

    print("\n=== summary ===")
    for idx, item in enumerate(results, start=1):
        print(
            f"case#{idx}: max_sum={item.max_sum:.6g}, "
            f"range=[{item.start_index}, {item.end_index}], subarray={item.subarray}"
        )

    global_ok = np.isclose(max_of_max, 10.0) and np.isclose(min_of_max, -2.0)
    print(f"global checks pass: {bool(global_ok)}")
    print(f"aggregate stats: max_of_max={max_of_max:.6g}, min_of_max={min_of_max:.6g}")


if __name__ == "__main__":
    main()

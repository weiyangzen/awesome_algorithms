"""CS-0086: 寻找峰值元素（分治）最小可运行 MVP。

实现目标：
1) 用分治/二分在 O(log n) 时间内返回任意一个峰值下标；
2) 提供 O(n) 线性扫描版本作为基线；
3) 运行脚本时自动执行固定样例、随机回归和小基准，无交互输入。
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass
class DivideConquerStats:
    recursive_calls: int = 0
    comparisons: int = 0
    max_depth: int = 0


@dataclass
class PeakResult:
    index: int
    value: int
    stats: DivideConquerStats


def _to_1d_int_array(nums: Iterable[int]) -> np.ndarray:
    arr = np.asarray(list(nums), dtype=np.int64)
    if arr.ndim != 1:
        raise ValueError("Input must be a 1-D sequence.")
    if arr.size == 0:
        raise ValueError("Input array must be non-empty.")
    return arr


def is_peak(nums: Sequence[int], idx: int) -> bool:
    """检查 idx 是否满足严格峰值定义。"""
    n = len(nums)
    if not (0 <= idx < n):
        return False

    value = nums[idx]
    left = nums[idx - 1] if idx > 0 else float("-inf")
    right = nums[idx + 1] if idx < n - 1 else float("-inf")
    return value > left and value > right


def find_peak_element_divide_conquer(nums: Iterable[int]) -> PeakResult:
    """分治查找任意峰值，返回峰值下标和值以及统计信息。

    说明：该实现遵循经典设定（相邻元素不等）时的严格峰值语义。
    """
    arr = _to_1d_int_array(nums)
    stats = DivideConquerStats()

    def solve(left: int, right: int, depth: int) -> int:
        stats.recursive_calls += 1
        stats.max_depth = max(stats.max_depth, depth)

        if left == right:
            return left

        mid = (left + right) // 2
        stats.comparisons += 1

        if arr[mid] > arr[mid + 1]:
            return solve(left, mid, depth + 1)
        return solve(mid + 1, right, depth + 1)

    peak_idx = solve(0, arr.size - 1, 1)
    return PeakResult(index=int(peak_idx), value=int(arr[peak_idx]), stats=stats)


def find_peak_element_linear(nums: Iterable[int]) -> int:
    """线性基线：返回第一个满足严格峰值定义的下标。"""
    arr = _to_1d_int_array(nums)
    n = arr.size

    for i in range(n):
        left = arr[i - 1] if i > 0 else float("-inf")
        right = arr[i + 1] if i < n - 1 else float("-inf")
        if arr[i] > left and arr[i] > right:
            return int(i)

    raise RuntimeError("No strict peak found. Check whether adjacent values are distinct.")


def _random_distinct_array(rng: np.random.Generator, size: int) -> np.ndarray:
    """生成相邻必不等的数据（通过唯一值排列）。"""
    if size <= 0:
        return np.array([], dtype=np.int64)
    return rng.permutation(size * 8).astype(np.int64)[:size]


def run_fixed_cases() -> None:
    cases = [
        np.array([1], dtype=np.int64),
        np.array([1, 2, 3, 1], dtype=np.int64),
        np.array([1, 2, 1, 3, 5, 6, 4], dtype=np.int64),
        np.array([10, 9, 8, 7], dtype=np.int64),
        np.array([1, 3, 2, 4, 1], dtype=np.int64),
    ]

    print("[Fixed cases]")
    for arr in cases:
        fast = find_peak_element_divide_conquer(arr)
        slow_idx = find_peak_element_linear(arr)

        if not is_peak(arr, fast.index):
            raise AssertionError(f"Divide-conquer index is not a peak: {arr.tolist()} -> {fast.index}")
        if not is_peak(arr, slow_idx):
            raise AssertionError(f"Linear index is not a peak: {arr.tolist()} -> {slow_idx}")

        print(
            f"nums={arr.tolist()}, fast_idx={fast.index}, fast_val={fast.value}, "
            f"slow_idx={slow_idx}, calls={fast.stats.recursive_calls}, comps={fast.stats.comparisons}"
        )


def run_random_regression() -> None:
    rng = np.random.default_rng(86)
    total = 0

    for n in [1, 2, 3, 4, 8, 16, 32, 64, 128, 256]:
        for _ in range(30):
            arr = _random_distinct_array(rng, n)
            fast = find_peak_element_divide_conquer(arr)
            slow_idx = find_peak_element_linear(arr)

            if not is_peak(arr, fast.index):
                raise AssertionError(
                    f"Random check failed (fast): n={n}, arr={arr.tolist()}, idx={fast.index}"
                )
            if not is_peak(arr, slow_idx):
                raise AssertionError(
                    f"Random check failed (slow): n={n}, arr={arr.tolist()}, idx={slow_idx}"
                )

            total += 1

    print(f"Random cross-check cases: {total}")


def run_benchmark() -> None:
    rng = np.random.default_rng(2026)
    sizes = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    repeats = 40
    rows: list[dict[str, float]] = []

    for n in sizes:
        batch = [_random_distinct_array(rng, n) for _ in range(repeats)]

        t0 = perf_counter()
        for arr in batch:
            _ = find_peak_element_divide_conquer(arr)
        t1 = perf_counter()

        for arr in batch:
            _ = find_peak_element_linear(arr)
        t2 = perf_counter()

        rows.append(
            {
                "n": n,
                "divide_conquer_ms": (t1 - t0) * 1000.0 / repeats,
                "linear_ms": (t2 - t1) * 1000.0 / repeats,
                "speedup_linear_over_dc": ((t2 - t1) / (t1 - t0)) if (t1 - t0) > 0 else float("nan"),
            }
        )

    df = pd.DataFrame(rows)
    pd.set_option("display.float_format", "{:.4f}".format)

    print("\n[Benchmark] average time per case (ms)")
    print(df.to_string(index=False))


def main() -> None:
    run_fixed_cases()
    run_random_regression()
    run_benchmark()
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()

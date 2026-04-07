"""斜率优化 DP 的最小可运行示例。

模型:
    dp[i] = min_{0<=j<i} { dp[j] + (S[i]-S[j])^2 + C }
其中 S 是输入非负序列的前缀和。

当 S[i] 单调不降时，可把转移改写为直线集合最小值查询，
并用单调队列维护下凸壳，把复杂度从 O(n^2) 降到 O(n)。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Line:
    """表示一条直线 y = m*x + b，并记录来自哪个 j 状态。"""

    m: int
    b: int
    idx: int


def validate_inputs(values: Sequence[int], penalty: int) -> np.ndarray:
    """校验并标准化输入。

    约束:
    - values 必须是一维整数序列;
    - values 需全部非负，以保证前缀和单调，从而可用单调 CHT;
    - penalty 必须是非负整数。
    """
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError("values must be a 1D sequence")
    if arr.size == 0:
        return arr.astype(np.int64)
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError("values must contain integers")
    if np.any(arr < 0):
        raise ValueError("values must be non-negative for monotonic CHT")
    if penalty < 0:
        raise ValueError("penalty must be non-negative")
    return arr.astype(np.int64, copy=False)


def line_value(line: Line, x: int) -> int:
    """计算直线在 x 处的取值。"""
    return line.m * x + line.b


def is_redundant(l1: Line, l2: Line, l3: Line) -> bool:
    """判断 l2 是否被 l1 与 l3 永久支配。

    条件对应交点顺序失效:
        intersection(l1,l2) >= intersection(l2,l3)

    使用整数交叉乘法，避免浮点误差。
    假设斜率按非增顺序插入。
    """
    left = (l2.b - l1.b) * (l2.m - l3.m)
    right = (l3.b - l2.b) * (l1.m - l2.m)
    return left >= right


def slope_optimized_dp(values: Sequence[int], penalty: int) -> tuple[list[int], list[int], np.ndarray]:
    """单调队列版斜率优化 DP。

    返回:
    - dp: 最优值数组, 长度 n+1
    - prev: 最优转移来源 j, 长度 n+1
    - prefix: 前缀和 S, 长度 n+1
    """
    arr = validate_inputs(values, penalty)
    n = int(arr.size)

    prefix = np.zeros(n + 1, dtype=np.int64)
    if n > 0:
        prefix[1:] = np.cumsum(arr, dtype=np.int64)

    dp = [0] * (n + 1)
    prev = [-1] * (n + 1)

    hull: deque[Line] = deque()
    hull.append(Line(m=0, b=0, idx=0))

    for i in range(1, n + 1):
        x = int(prefix[i])

        while len(hull) >= 2 and line_value(hull[0], x) >= line_value(hull[1], x):
            hull.popleft()

        best = hull[0]
        dp[i] = x * x + penalty + line_value(best, x)
        prev[i] = best.idx

        new_line = Line(m=-2 * x, b=dp[i] + x * x, idx=i)

        while hull and hull[-1].m == new_line.m:
            if hull[-1].b <= new_line.b:
                new_line = None  # type: ignore[assignment]
                break
            hull.pop()

        if new_line is not None:
            while len(hull) >= 2 and is_redundant(hull[-2], hull[-1], new_line):
                hull.pop()
            hull.append(new_line)

    return dp, prev, prefix


def brute_force_dp(values: Sequence[int], penalty: int) -> tuple[list[int], list[int], np.ndarray]:
    """O(n^2) DP, 作为真值对照。"""
    arr = validate_inputs(values, penalty)
    n = int(arr.size)

    prefix = np.zeros(n + 1, dtype=np.int64)
    if n > 0:
        prefix[1:] = np.cumsum(arr, dtype=np.int64)

    dp = [0] * (n + 1)
    prev = [-1] * (n + 1)

    for i in range(1, n + 1):
        best_cost = None
        best_j = -1
        si = int(prefix[i])
        for j in range(i):
            diff = si - int(prefix[j])
            candidate = dp[j] + diff * diff + penalty
            if best_cost is None or candidate < best_cost:
                best_cost = candidate
                best_j = j
        dp[i] = int(best_cost)
        prev[i] = best_j

    return dp, prev, prefix


def reconstruct_segments(prev: Sequence[int]) -> list[tuple[int, int]]:
    """由 prev 回溯分段，返回 1-based 闭区间列表。"""
    segments: list[tuple[int, int]] = []
    i = len(prev) - 1
    while i > 0:
        j = prev[i]
        if j < 0 or j >= i:
            raise RuntimeError(f"invalid prev chain at i={i}, j={j}")
        segments.append((j + 1, i))
        i = j
    segments.reverse()
    return segments


def segment_cost(values: Sequence[int], penalty: int, segments: Sequence[tuple[int, int]]) -> int:
    """计算给定分段的总代价（用于校验多最优解场景）。"""
    arr = np.asarray(values, dtype=np.int64)
    total = 0
    for l, r in segments:
        seg_sum = int(np.sum(arr[l - 1 : r], dtype=np.int64))
        total += seg_sum * seg_sum + penalty
    return total


def run_case(name: str, values: Sequence[int], penalty: int) -> None:
    """运行单个测试案例并输出结果。"""
    dp_cht, prev_cht, _ = slope_optimized_dp(values, penalty)
    dp_bf, prev_bf, _ = brute_force_dp(values, penalty)

    if dp_cht[-1] != dp_bf[-1]:
        raise AssertionError(
            f"Case {name} mismatch: cht={dp_cht[-1]} vs brute={dp_bf[-1]}"
        )

    segments_cht = reconstruct_segments(prev_cht)
    segments_bf = reconstruct_segments(prev_bf)
    cht_cost_by_segments = segment_cost(values, penalty, segments_cht)
    bf_cost_by_segments = segment_cost(values, penalty, segments_bf)

    if cht_cost_by_segments != dp_cht[-1]:
        raise AssertionError(
            f"Case {name} invalid reconstruction: seg_cost={cht_cost_by_segments}, dp={dp_cht[-1]}"
        )

    print(f"=== {name} ===")
    print(f"values={list(values)}")
    print(f"penalty={penalty}")
    print(f"best_cost_cht={dp_cht[-1]}")
    print(f"best_cost_bruteforce={dp_bf[-1]}")
    print(f"segments_1_based={segments_cht}")
    print(f"segments_cost_check={cht_cost_by_segments == bf_cost_by_segments == dp_cht[-1]}")
    print(f"dp_tail={dp_cht[max(0, len(dp_cht) - 6):]}")
    print()


def main() -> None:
    rng = np.random.default_rng(20260407)

    cases: list[tuple[str, list[int], int]] = [
        ("Small hand-check", [2, 1, 3, 2, 4], 5),
        ("Zeros included", [0, 4, 0, 1, 0, 3, 2], 7),
        ("Deterministic random", rng.integers(0, 6, size=14).tolist(), 9),
        ("Longer stress-lite", rng.integers(0, 10, size=28).tolist(), 12),
    ]

    for name, values, penalty in cases:
        run_case(name=name, values=values, penalty=penalty)

    print("All cases passed.")


if __name__ == "__main__":
    main()

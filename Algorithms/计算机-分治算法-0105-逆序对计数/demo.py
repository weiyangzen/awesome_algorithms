"""CS-0084: 逆序对计数（分治 + 归并）的最小可运行 MVP。

实现目标：
1) 用归并分治在 O(n log n) 时间统计逆序对。
2) 提供 O(n^2) 朴素版本用于对拍验证。
3) 运行脚本时自动执行固定样例和随机测试，无交互输入。
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def _merge_count(arr: list[int], buffer: list[int], left: int, mid: int, right: int) -> int:
    """归并两个有序区间并返回跨区逆序对数量。"""
    i = left
    j = mid + 1
    k = left
    cross_count = 0

    while i <= mid and j <= right:
        # 使用 <=，避免把相等元素误计为逆序对。
        if arr[i] <= arr[j]:
            buffer[k] = arr[i]
            i += 1
        else:
            buffer[k] = arr[j]
            j += 1
            cross_count += mid - i + 1
        k += 1

    while i <= mid:
        buffer[k] = arr[i]
        i += 1
        k += 1

    while j <= right:
        buffer[k] = arr[j]
        j += 1
        k += 1

    arr[left : right + 1] = buffer[left : right + 1]
    return cross_count


def _sort_count(arr: list[int], buffer: list[int], left: int, right: int) -> int:
    """递归分治：返回 arr[left:right+1] 的逆序对数量。"""
    if left >= right:
        return 0

    mid = (left + right) // 2
    left_count = _sort_count(arr, buffer, left, mid)
    right_count = _sort_count(arr, buffer, mid + 1, right)
    cross_count = _merge_count(arr, buffer, left, mid, right)
    return left_count + right_count + cross_count


def count_inversions_divide_conquer(nums: Iterable[int]) -> int:
    """分治法统计逆序对，时间复杂度 O(n log n)。"""
    arr = [int(x) for x in nums]
    n = len(arr)
    if n <= 1:
        return 0

    buffer = [0] * n
    return _sort_count(arr, buffer, 0, n - 1)


def count_inversions_naive(nums: Iterable[int]) -> int:
    """朴素双循环统计逆序对，时间复杂度 O(n^2)。"""
    arr = [int(x) for x in nums]
    n = len(arr)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if arr[i] > arr[j]:
                count += 1
    return count


def main() -> None:
    example = np.array([2, 4, 1, 3, 5], dtype=np.int64)
    expected = 3

    fast = count_inversions_divide_conquer(example)
    slow = count_inversions_naive(example)

    print("Example array:", example.tolist())
    print("Divide-and-conquer inversion count:", fast)
    print("Naive inversion count:", slow)

    if fast != expected:
        raise AssertionError(f"Expected {expected}, got {fast}")
    if fast != slow:
        raise AssertionError("Fast result does not match naive result on example.")

    # 随机对拍：覆盖重复值、负数与不同长度。
    rng = np.random.default_rng(84)
    for n in [0, 1, 2, 3, 5, 8, 16, 32, 64]:
        for _ in range(20):
            data = rng.integers(-20, 21, size=n, dtype=np.int64)
            lhs = count_inversions_divide_conquer(data)
            rhs = count_inversions_naive(data)
            if lhs != rhs:
                raise AssertionError(
                    f"Mismatch for n={n}, data={data.tolist()}, fast={lhs}, naive={rhs}"
                )

    print("Random cross-check cases: 180")
    print("All checks passed.")


if __name__ == "__main__":
    main()

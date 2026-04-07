"""Quick Sort MVP (divide-and-conquer) with basic verification and benchmark."""

from __future__ import annotations

import random
import time

import numpy as np


def _partition(arr: list[int], left: int, right: int, pivot_idx: int) -> int:
    """Lomuto partition with explicit pivot index."""
    arr[pivot_idx], arr[right] = arr[right], arr[pivot_idx]
    pivot = arr[right]
    store_index = left

    for j in range(left, right):
        if arr[j] < pivot:
            arr[store_index], arr[j] = arr[j], arr[store_index]
            store_index += 1

    arr[store_index], arr[right] = arr[right], arr[store_index]
    return store_index


def _quicksort_inplace(arr: list[int], left: int, right: int) -> None:
    """In-place quicksort; recurse on smaller side first to reduce stack depth."""
    while left < right:
        pivot_idx = random.randint(left, right)
        mid = _partition(arr, left, right, pivot_idx)

        left_len = mid - left
        right_len = right - mid

        if left_len < right_len:
            _quicksort_inplace(arr, left, mid - 1)
            left = mid + 1
        else:
            _quicksort_inplace(arr, mid + 1, right)
            right = mid - 1


def quicksort(values: list[int]) -> list[int]:
    """Return a sorted copy of values (ascending)."""
    arr = list(values)
    if len(arr) < 2:
        return arr
    _quicksort_inplace(arr, 0, len(arr) - 1)
    return arr


def _validate_correctness() -> None:
    cases = [
        [],
        [42],
        [1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1],
        [3, 1, 2, 3, 3, 0, -1],
        [9, -3, 5, 2, 6, 8, -6, 1, 3],
    ]

    for case in cases:
        assert quicksort(case) == sorted(case)

    for _ in range(30):
        size = random.randint(0, 200)
        arr = [random.randint(-10_000, 10_000) for _ in range(size)]
        assert quicksort(arr) == sorted(arr)


def _benchmark_once(size: int = 20_000) -> tuple[float, float, float]:
    arr = [random.randint(-1_000_000, 1_000_000) for _ in range(size)]
    arr_np = np.array(arr)

    t0 = time.perf_counter()
    res_qs = quicksort(arr)
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    res_py = sorted(arr)
    t3 = time.perf_counter()

    t4 = time.perf_counter()
    res_np = np.sort(arr_np)
    t5 = time.perf_counter()

    assert res_qs == res_py == res_np.tolist()
    return (t1 - t0, t3 - t2, t5 - t4)


def main() -> None:
    random.seed(20260407)
    np.random.seed(20260407)

    sample = [8, 3, 1, 7, 0, 10, 2, 6, 4, 5, 9, 3]
    print(f"input : {sample}")
    print(f"sorted: {quicksort(sample)}")

    _validate_correctness()
    print("Correctness checks passed.")

    t_qs, t_py, t_np = _benchmark_once()
    print("Timing (size=20000):")
    print(f"  quicksort   : {t_qs:.6f}s")
    print(f"  built-in sort: {t_py:.6f}s")
    print(f"  numpy.sort  : {t_np:.6f}s")


if __name__ == "__main__":
    main()

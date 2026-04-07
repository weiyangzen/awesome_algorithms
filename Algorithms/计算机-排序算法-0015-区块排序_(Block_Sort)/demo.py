"""Minimal runnable MVP for 区块排序 (Block Sort).

This demo implements a transparent block-based stable sort:
1) sort fixed-size blocks locally,
2) iteratively merge neighboring sorted runs.

No interactive input is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, List, Sequence, Tuple, TypeVar

import numpy as np

T = TypeVar("T")


@dataclass(frozen=True)
class BlockSortConfig:
    n_items: int = 2000
    value_low: int = -200
    value_high: int = 200
    block_size: int = 16
    seed: int = 42

    def validate(self) -> None:
        if self.n_items <= 0:
            raise ValueError("n_items must be positive")
        if self.value_low >= self.value_high:
            raise ValueError("value_low must be < value_high")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")


def insertion_sort_range(arr: List[T], left: int, right: int, key: Callable[[T], object]) -> None:
    """Sort arr[left:right] in place using insertion sort (stable)."""
    for i in range(left + 1, right):
        cur = arr[i]
        cur_key = key(cur)
        j = i - 1
        while j >= left and key(arr[j]) > cur_key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = cur


def merge_ranges(
    arr: List[T],
    left: int,
    mid: int,
    right: int,
    temp: List[T | None],
    key: Callable[[T], object],
) -> None:
    """Merge two sorted ranges arr[left:mid] and arr[mid:right] stably."""
    i = left
    j = mid
    k = left

    while i < mid and j < right:
        # Stability: use <= so equal keys from left run stay before right run.
        if key(arr[i]) <= key(arr[j]):
            temp[k] = arr[i]
            i += 1
        else:
            temp[k] = arr[j]
            j += 1
        k += 1

    while i < mid:
        temp[k] = arr[i]
        i += 1
        k += 1

    while j < right:
        temp[k] = arr[j]
        j += 1
        k += 1

    for idx in range(left, right):
        arr[idx] = temp[idx]  # type: ignore[assignment]


def block_sort(arr: Sequence[T], block_size: int, key: Callable[[T], object] = lambda x: x) -> List[T]:
    """Return a stably sorted copy of arr using block-local sort + iterative merges."""
    n = len(arr)
    if n <= 1:
        return list(arr)
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    out = list(arr)
    temp: List[T | None] = [None] * n

    # Phase 1: sort each block independently.
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        insertion_sort_range(out, start, end, key)

    # Phase 2: iteratively merge neighboring sorted runs.
    run = block_size
    while run < n:
        for left in range(0, n, 2 * run):
            mid = min(left + run, n)
            right = min(left + 2 * run, n)
            if mid < right:
                merge_ranges(out, left, mid, right, temp, key)
        run *= 2

    return out


def is_sorted_non_decreasing(arr: Sequence[int]) -> bool:
    return all(arr[i] <= arr[i + 1] for i in range(len(arr) - 1))


def check_stability(sorted_records: Sequence[Tuple[int, int]]) -> bool:
    """For equal values, original indices should remain non-decreasing."""
    groups: dict[int, List[int]] = {}
    for value, original_idx in sorted_records:
        groups.setdefault(value, []).append(original_idx)
    return all(
        all(indices[i] <= indices[i + 1] for i in range(len(indices) - 1))
        for indices in groups.values()
    )


def main() -> None:
    cfg = BlockSortConfig()
    cfg.validate()

    rng = np.random.default_rng(cfg.seed)
    data = rng.integers(cfg.value_low, cfg.value_high + 1, size=cfg.n_items, endpoint=False).tolist()

    t0 = perf_counter()
    sorted_data = block_sort(data, block_size=cfg.block_size)
    elapsed_ms = (perf_counter() - t0) * 1000.0

    expected = sorted(data)
    if not is_sorted_non_decreasing(sorted_data):
        raise AssertionError("Block sort result is not non-decreasing")
    if sorted_data != expected:
        raise AssertionError("Block sort result differs from Python sorted")

    # Stability check on (value, original_index) records.
    records = [(v, i) for i, v in enumerate(data)]
    sorted_records = block_sort(records, block_size=cfg.block_size, key=lambda x: x[0])
    if not check_stability(sorted_records):
        raise AssertionError("Stability check failed for duplicate keys")

    print("Block Sort MVP")
    print(f"n_items={cfg.n_items}, block_size={cfg.block_size}, seed={cfg.seed}")
    print(f"input head : {data[:20]}")
    print(f"sorted head: {sorted_data[:20]}")
    print(f"time_ms    : {elapsed_ms:.3f}")
    print("All checks passed.")


if __name__ == "__main__":
    main()

"""Quicksort MVP.

A minimal, auditable quicksort implementation with deterministic test cases.
No interactive input is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple, TypeVar

import numpy as np

T = TypeVar("T")


@dataclass
class QuickSortStats:
    comparisons: int = 0
    swaps: int = 0
    partition_calls: int = 0
    recursive_calls: int = 0
    max_depth: int = 0


def quick_sort(data: Sequence[T], key: Callable[[T], object] = lambda x: x) -> Tuple[List[T], QuickSortStats]:
    """Return an in-place quicksorted copy of `data` and operation stats."""
    arr = list(data)
    stats = QuickSortStats()

    def swap(i: int, j: int) -> None:
        if i != j:
            arr[i], arr[j] = arr[j], arr[i]
            stats.swaps += 1

    def choose_pivot_index(lo: int, hi: int) -> int:
        """Median-of-three pivot selection on indices lo, mid, hi."""
        mid = (lo + hi) // 2
        a = key(arr[lo])
        b = key(arr[mid])
        c = key(arr[hi])

        if a < b:
            if b < c:
                return mid
            return hi if a < c else lo

        if a < c:
            return lo
        return hi if b < c else mid

    def partition(lo: int, hi: int) -> int:
        """Lomuto partition on closed interval [lo, hi]."""
        stats.partition_calls += 1

        pivot_index = choose_pivot_index(lo, hi)
        swap(pivot_index, hi)
        pivot_value = key(arr[hi])

        i = lo
        for j in range(lo, hi):
            stats.comparisons += 1
            if key(arr[j]) <= pivot_value:
                swap(i, j)
                i += 1

        swap(i, hi)
        return i

    def sort(lo: int, hi: int, depth: int) -> None:
        stats.recursive_calls += 1
        if depth > stats.max_depth:
            stats.max_depth = depth

        if lo >= hi:
            return

        pivot = partition(lo, hi)
        sort(lo, pivot - 1, depth + 1)
        sort(pivot + 1, hi, depth + 1)

    if arr:
        sort(0, len(arr) - 1, 0)

    return arr, stats


def check_stability(sorted_records: Sequence[Tuple[int, int]]) -> bool:
    """For equal keys, original indices should remain non-decreasing."""
    groups: dict[int, List[int]] = {}
    for value, original_idx in sorted_records:
        groups.setdefault(value, []).append(original_idx)

    return all(
        all(indices[i] <= indices[i + 1] for i in range(len(indices) - 1))
        for indices in groups.values()
    )


def run_case(case_name: str, data: Sequence[int]) -> bool:
    result, stats = quick_sort(data)
    expected = sorted(data)
    ok = result == expected

    print(f"\n[{case_name}]")
    print(f"input          : {list(data)}")
    print(f"quick_sort     : {result}")
    print(f"python_sorted  : {expected}")
    print(f"match          : {ok}")
    print(f"comparisons    : {stats.comparisons}")
    print(f"swaps          : {stats.swaps}")
    print(f"partition_calls: {stats.partition_calls}")
    print(f"recursive_calls: {stats.recursive_calls}")
    print(f"max_depth      : {stats.max_depth}")

    return ok


def main() -> None:
    rng = np.random.default_rng(2026)

    fixed_case = [5, 2, 4, 6, 1, 3]
    duplicate_case = [3, -1, 3, 2, -1, 0, 3]
    already_sorted_case = [-5, -1, 0, 2, 4, 9]
    reverse_case = [9, 7, 5, 3, 1, -1]
    all_equal_case = [7, 7, 7, 7, 7]
    empty_case: List[int] = []
    single_case = [42]
    random_case = rng.integers(low=-20, high=21, size=15).tolist()

    checks = [
        run_case("fixed", fixed_case),
        run_case("duplicates_and_negatives", duplicate_case),
        run_case("already_sorted", already_sorted_case),
        run_case("reverse", reverse_case),
        run_case("all_equal", all_equal_case),
        run_case("empty", empty_case),
        run_case("single", single_case),
        run_case("random", random_case),
    ]

    # Quicksort is generally unstable; this record case should expose that.
    records = [(v, i) for i, v in enumerate([2, 2, 1, 2, 1, 2])]
    sorted_records, _ = quick_sort(records, key=lambda x: x[0])
    stable_ok = check_stability(sorted_records)

    print("\n[stability_check]")
    print(f"records_input : {records}")
    print(f"records_sorted: {sorted_records}")
    print(f"stable        : {stable_ok}")

    instability_observed = not stable_ok
    all_ok = all(checks) and instability_observed

    print("\n=== Summary ===")
    print(f"all_cases_passed={all_ok}")
    print(f"sorting_cases={len(checks)}")
    print(f"instability_observed={instability_observed}")


if __name__ == "__main__":
    main()

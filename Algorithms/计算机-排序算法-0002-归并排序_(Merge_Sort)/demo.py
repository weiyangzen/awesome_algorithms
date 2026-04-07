"""Merge Sort MVP.

A minimal, auditable merge-sort implementation with deterministic tests.
No interactive input is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple, TypeVar

import numpy as np

T = TypeVar("T")


@dataclass
class MergeSortStats:
    comparisons: int = 0
    writes: int = 0
    merge_calls: int = 0
    max_depth: int = 0


def merge_sort(data: Sequence[T], key: Callable[[T], object] = lambda x: x) -> Tuple[List[T], MergeSortStats]:
    """Return a stably sorted copy of `data` and operation stats."""
    arr = list(data)
    aux = list(arr)
    stats = MergeSortStats()

    def merge(lo: int, mid: int, hi: int) -> None:
        stats.merge_calls += 1

        i = lo
        j = mid
        k = lo

        # Merge two sorted slices: arr[lo:mid] and arr[mid:hi].
        while i < mid and j < hi:
            stats.comparisons += 1
            if key(arr[i]) <= key(arr[j]):
                aux[k] = arr[i]
                i += 1
            else:
                aux[k] = arr[j]
                j += 1
            k += 1
            stats.writes += 1

        while i < mid:
            aux[k] = arr[i]
            i += 1
            k += 1
            stats.writes += 1

        while j < hi:
            aux[k] = arr[j]
            j += 1
            k += 1
            stats.writes += 1

        for p in range(lo, hi):
            arr[p] = aux[p]
            stats.writes += 1

    def sort(lo: int, hi: int, depth: int) -> None:
        if depth > stats.max_depth:
            stats.max_depth = depth

        if hi - lo <= 1:
            return

        mid = (lo + hi) // 2
        sort(lo, mid, depth + 1)
        sort(mid, hi, depth + 1)
        merge(lo, mid, hi)

    sort(0, len(arr), 0)
    return arr, stats


def check_stability(sorted_records: Sequence[Tuple[int, int]]) -> bool:
    """For equal values, original indices should stay non-decreasing."""
    groups: dict[int, List[int]] = {}
    for value, original_idx in sorted_records:
        groups.setdefault(value, []).append(original_idx)
    return all(
        all(indices[i] <= indices[i + 1] for i in range(len(indices) - 1))
        for indices in groups.values()
    )


def run_case(case_name: str, data: Sequence[int]) -> bool:
    result, stats = merge_sort(data)
    expected = sorted(data)
    ok = result == expected

    print(f"\n[{case_name}]")
    print(f"input        : {list(data)}")
    print(f"merge_sort   : {result}")
    print(f"python_sorted: {expected}")
    print(f"match        : {ok}")
    print(f"comparisons  : {stats.comparisons}")
    print(f"writes       : {stats.writes}")
    print(f"merge_calls  : {stats.merge_calls}")
    print(f"max_depth    : {stats.max_depth}")

    return ok


def main() -> None:
    rng = np.random.default_rng(2026)

    fixed_case = [5, 2, 4, 6, 1, 3]
    duplicate_case = [3, -1, 3, 2, -1, 0, 3]
    already_sorted_case = [-5, -1, 0, 2, 4, 9]
    reverse_case = [9, 7, 5, 3, 1, -1]
    empty_case: List[int] = []
    single_case = [42]
    random_case = rng.integers(low=-20, high=21, size=15).tolist()

    checks = [
        run_case("fixed", fixed_case),
        run_case("duplicates_and_negatives", duplicate_case),
        run_case("already_sorted", already_sorted_case),
        run_case("reverse", reverse_case),
        run_case("empty", empty_case),
        run_case("single", single_case),
        run_case("random", random_case),
    ]

    # Stability check on records: (value, original_index)
    records = [(v, i) for i, v in enumerate(duplicate_case)]
    sorted_records, _ = merge_sort(records, key=lambda x: x[0])
    stable_ok = check_stability(sorted_records)

    print("\n[stability_check]")
    print(f"records_input : {records}")
    print(f"records_sorted: {sorted_records}")
    print(f"stable        : {stable_ok}")

    all_ok = all(checks) and stable_ok
    print("\n=== Summary ===")
    print(f"all_cases_passed={all_ok}")
    print(f"num_cases={len(checks)}")


if __name__ == "__main__":
    main()

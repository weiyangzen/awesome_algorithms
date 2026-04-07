"""Insertion Sort MVP.

A minimal, auditable insertion-sort implementation with deterministic tests.
No interactive input is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple, TypeVar

import numpy as np

T = TypeVar("T")


@dataclass
class InsertionSortStats:
    comparisons: int = 0
    shifts: int = 0
    insertions: int = 0


def insertion_sort_in_place(arr: List[T], key: Callable[[T], object] = lambda x: x) -> InsertionSortStats:
    """Sort `arr` in place using stable insertion sort and return operation stats."""
    stats = InsertionSortStats()

    for i in range(1, len(arr)):
        current = arr[i]
        current_key = key(current)
        j = i - 1

        # Shift bigger elements one step to the right.
        while j >= 0:
            stats.comparisons += 1
            if key(arr[j]) > current_key:
                arr[j + 1] = arr[j]
                stats.shifts += 1
                j -= 1
            else:
                break

        arr[j + 1] = current
        stats.insertions += 1

    return stats


def insertion_sort(data: Sequence[T], key: Callable[[T], object] = lambda x: x) -> Tuple[List[T], InsertionSortStats]:
    """Return a sorted copy of `data` and stats."""
    out = list(data)
    stats = insertion_sort_in_place(out, key=key)
    return out, stats


def check_stability(sorted_records: Sequence[Tuple[int, int]]) -> bool:
    """For equal values, original indices must stay non-decreasing."""
    groups: dict[int, List[int]] = {}
    for value, original_idx in sorted_records:
        groups.setdefault(value, []).append(original_idx)
    return all(
        all(indices[i] <= indices[i + 1] for i in range(len(indices) - 1))
        for indices in groups.values()
    )


def run_case(case_name: str, data: Sequence[int]) -> bool:
    result, stats = insertion_sort(data)
    expected = sorted(data)
    ok = result == expected

    print(f"\n[{case_name}]")
    print(f"input         : {list(data)}")
    print(f"insertion_sort: {result}")
    print(f"python_sorted : {expected}")
    print(f"match         : {ok}")
    print(f"comparisons   : {stats.comparisons}")
    print(f"shifts        : {stats.shifts}")
    print(f"insertions    : {stats.insertions}")

    return ok


def main() -> None:
    rng = np.random.default_rng(2026)

    fixed_case = [5, 2, 4, 6, 1, 3]
    duplicate_case = [3, -1, 3, 2, -1, 0, 3]
    empty_case: List[int] = []
    single_case = [42]
    random_case = rng.integers(low=-20, high=21, size=15).tolist()

    checks = [
        run_case("fixed", fixed_case),
        run_case("duplicates_and_negatives", duplicate_case),
        run_case("empty", empty_case),
        run_case("single", single_case),
        run_case("random", random_case),
    ]

    # Stability check on records: (value, original_index)
    records = [(v, i) for i, v in enumerate(duplicate_case)]
    sorted_records, _ = insertion_sort(records, key=lambda x: x[0])
    stable_ok = check_stability(sorted_records)

    print("\n[stability_check]")
    print(f"records_input  : {records}")
    print(f"records_sorted : {sorted_records}")
    print(f"stable         : {stable_ok}")

    all_ok = all(checks) and stable_ok
    print("\n=== Summary ===")
    print(f"all_cases_passed={all_ok}")
    print(f"num_cases={len(checks)}")


if __name__ == "__main__":
    main()

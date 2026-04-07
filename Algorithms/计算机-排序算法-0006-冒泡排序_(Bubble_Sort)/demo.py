"""Bubble Sort MVP.

A minimal, auditable bubble-sort implementation with deterministic tests.
No interactive input is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple, TypeVar

import numpy as np

T = TypeVar("T")


@dataclass
class BubbleSortStats:
    comparisons: int = 0
    swaps: int = 0
    passes: int = 0
    early_stop: bool = False


def bubble_sort(data: Sequence[T], key: Callable[[T], object] = lambda x: x) -> Tuple[List[T], BubbleSortStats]:
    """Return a stably sorted copy of `data` and operation stats."""
    arr = list(data)
    stats = BubbleSortStats()
    n = len(arr)

    if n <= 1:
        return arr, stats

    for i in range(n - 1):
        stats.passes += 1
        swapped = False
        upper = n - 1 - i

        for j in range(upper):
            stats.comparisons += 1
            if key(arr[j]) > key(arr[j + 1]):
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                stats.swaps += 1
                swapped = True

        if not swapped:
            stats.early_stop = True
            break

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
    result, stats = bubble_sort(data)
    expected = sorted(data)
    ok = result == expected

    print(f"\n[{case_name}]")
    print(f"input        : {list(data)}")
    print(f"bubble_sort  : {result}")
    print(f"python_sorted: {expected}")
    print(f"match        : {ok}")
    print(f"comparisons  : {stats.comparisons}")
    print(f"swaps        : {stats.swaps}")
    print(f"passes       : {stats.passes}")
    print(f"early_stop   : {stats.early_stop}")

    return ok


def main() -> None:
    rng = np.random.default_rng(2026)

    fixed_case = [5, 1, 4, 2, 8]
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
    sorted_records, _ = bubble_sort(records, key=lambda x: x[0])
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

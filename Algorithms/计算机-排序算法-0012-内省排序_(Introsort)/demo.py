"""Runnable MVP for Introsort (CS-0012).

Introsort = quicksort + depth-limit fallback to heapsort + insertion sort for
small partitions. This script implements the algorithm from scratch and uses
NumPy/Pandas only for validation and readable reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IntrosortStats:
    """Execution statistics for one introsort run."""

    n: int
    max_depth_limit: int
    insertion_threshold: int
    comparisons: int
    swaps: int
    partitions: int
    heapsort_fallbacks: int
    insertion_calls: int
    trace: list[dict[str, Any]]


def validate_numeric_sequence(values: Sequence[float]) -> list[float]:
    """Validate input and return a finite 1D float list."""
    if isinstance(values, (str, bytes)):
        raise TypeError("Input must be a numeric sequence, not string/bytes.")

    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Input must be a 1D sequence.")
    if not np.isfinite(arr).all():
        raise ValueError("Input contains non-finite values (NaN or Inf).")

    return arr.tolist()


def _swap(arr: list[float], i: int, j: int, counters: dict[str, int]) -> None:
    if i == j:
        return
    arr[i], arr[j] = arr[j], arr[i]
    counters["swaps"] += 1


def _median_of_three(
    arr: list[float],
    a: int,
    b: int,
    c: int,
    counters: dict[str, int],
) -> int:
    """Return index of median(arr[a], arr[b], arr[c]) and count comparisons."""
    av, bv, cv = arr[a], arr[b], arr[c]

    counters["comparisons"] += 1
    if av < bv:
        counters["comparisons"] += 1
        if bv < cv:
            return b
        counters["comparisons"] += 1
        if av < cv:
            return c
        return a

    counters["comparisons"] += 1
    if av < cv:
        return a
    counters["comparisons"] += 1
    if bv < cv:
        return c
    return b


def _partition(
    arr: list[float],
    lo: int,
    hi: int,
    counters: dict[str, int],
    trace: list[dict[str, Any]],
    depth_limit: int,
) -> int:
    """Partition arr[lo:hi) using Lomuto scheme and median-of-three pivot."""
    mid = lo + (hi - lo) // 2
    pivot_idx = _median_of_three(arr, lo, mid, hi - 1, counters)
    _swap(arr, pivot_idx, hi - 1, counters)
    pivot = arr[hi - 1]

    store = lo
    for i in range(lo, hi - 1):
        counters["comparisons"] += 1
        if arr[i] < pivot:
            _swap(arr, store, i, counters)
            store += 1

    _swap(arr, store, hi - 1, counters)
    counters["partitions"] += 1

    trace.append(
        {
            "event": "partition",
            "lo": lo,
            "hi": hi,
            "pivot_final_index": store,
            "pivot_value": pivot,
            "depth_limit_before_split": depth_limit,
            "left_size": store - lo,
            "right_size": hi - (store + 1),
        }
    )

    return store


def _insertion_sort_range(
    arr: list[float],
    lo: int,
    hi: int,
    counters: dict[str, int],
) -> None:
    counters["insertion_calls"] += 1

    for i in range(lo + 1, hi):
        key = arr[i]
        j = i - 1

        while j >= lo:
            counters["comparisons"] += 1
            if arr[j] <= key:
                break
            arr[j + 1] = arr[j]
            j -= 1

        arr[j + 1] = key


def _sift_down_heap(
    arr: list[float],
    base: int,
    start: int,
    end: int,
    counters: dict[str, int],
) -> None:
    """Sift-down in a max heap over arr[base:base+end)."""
    root = start
    while True:
        left = 2 * root + 1
        if left >= end:
            return

        largest = root

        counters["comparisons"] += 1
        if arr[base + left] > arr[base + largest]:
            largest = left

        right = left + 1
        if right < end:
            counters["comparisons"] += 1
            if arr[base + right] > arr[base + largest]:
                largest = right

        if largest == root:
            return

        _swap(arr, base + root, base + largest, counters)
        root = largest


def _heapsort_range(
    arr: list[float],
    lo: int,
    hi: int,
    counters: dict[str, int],
) -> None:
    """In-place heapsort on arr[lo:hi)."""
    counters["heapsort_fallbacks"] += 1
    length = hi - lo
    if length < 2:
        return

    for start in range((length // 2) - 1, -1, -1):
        _sift_down_heap(arr, lo, start, length, counters)

    for end in range(length - 1, 0, -1):
        _swap(arr, lo, lo + end, counters)
        _sift_down_heap(arr, lo, 0, end, counters)


def _introsort_loop(
    arr: list[float],
    lo: int,
    hi: int,
    depth_limit: int,
    insertion_threshold: int,
    counters: dict[str, int],
    trace: list[dict[str, Any]],
) -> None:
    """Sort arr[lo:hi) with depth-aware quicksort + heapsort fallback."""
    while hi - lo > insertion_threshold:
        if depth_limit == 0:
            trace.append({"event": "heapsort_fallback", "lo": lo, "hi": hi})
            _heapsort_range(arr, lo, hi, counters)
            return

        depth_limit -= 1
        p = _partition(arr, lo, hi, counters, trace, depth_limit + 1)

        # Recurse on the smaller half first; iterate on the larger one.
        if p - lo < hi - (p + 1):
            _introsort_loop(
                arr,
                lo,
                p,
                depth_limit,
                insertion_threshold,
                counters,
                trace,
            )
            lo = p + 1
        else:
            _introsort_loop(
                arr,
                p + 1,
                hi,
                depth_limit,
                insertion_threshold,
                counters,
                trace,
            )
            hi = p

    _insertion_sort_range(arr, lo, hi, counters)


def introsort(
    values: Sequence[float],
    insertion_threshold: int = 16,
    max_depth_override: int | None = None,
) -> tuple[list[float], IntrosortStats]:
    """Sort values in non-decreasing order with a hand-written introsort."""
    arr = validate_numeric_sequence(values)
    n = len(arr)

    if n <= 1:
        stats = IntrosortStats(
            n=n,
            max_depth_limit=0,
            insertion_threshold=insertion_threshold,
            comparisons=0,
            swaps=0,
            partitions=0,
            heapsort_fallbacks=0,
            insertion_calls=0,
            trace=[],
        )
        return arr, stats

    if insertion_threshold < 2:
        raise ValueError("insertion_threshold must be >= 2")

    max_depth_limit = (
        max_depth_override
        if max_depth_override is not None
        else 2 * int(math.log2(n))
    )

    counters = {
        "comparisons": 0,
        "swaps": 0,
        "partitions": 0,
        "heapsort_fallbacks": 0,
        "insertion_calls": 0,
    }
    trace: list[dict[str, Any]] = []

    _introsort_loop(
        arr=arr,
        lo=0,
        hi=n,
        depth_limit=max_depth_limit,
        insertion_threshold=insertion_threshold,
        counters=counters,
        trace=trace,
    )

    stats = IntrosortStats(
        n=n,
        max_depth_limit=max_depth_limit,
        insertion_threshold=insertion_threshold,
        comparisons=counters["comparisons"],
        swaps=counters["swaps"],
        partitions=counters["partitions"],
        heapsort_fallbacks=counters["heapsort_fallbacks"],
        insertion_calls=counters["insertion_calls"],
        trace=trace,
    )
    return arr, stats


def is_non_decreasing(values: Sequence[float]) -> bool:
    """Return whether the sequence is sorted in non-decreasing order."""
    return all(values[i] <= values[i + 1] for i in range(len(values) - 1))


def run_case(
    case_name: str,
    raw_values: Sequence[float],
    max_depth_override: int | None = None,
) -> None:
    """Run one deterministic case and print audit-friendly output."""
    values = validate_numeric_sequence(raw_values)
    sorted_values, stats = introsort(values, max_depth_override=max_depth_override)

    expected_py = sorted(values)
    expected_np = np.sort(np.asarray(values, dtype=float)).tolist()

    if sorted_values != expected_py or sorted_values != expected_np:
        raise RuntimeError(f"{case_name}: sorted output mismatch.")

    if not is_non_decreasing(sorted_values):
        raise RuntimeError(f"{case_name}: output is not non-decreasing.")

    print(f"\n=== {case_name} ===")
    print(f"Input: {values}")
    print(f"Sorted by introsort: {sorted_values}")
    print(f"Expected sorted:     {expected_py}")
    print(
        "Stats: "
        f"n={stats.n}, depth_limit={stats.max_depth_limit}, "
        f"comparisons={stats.comparisons}, swaps={stats.swaps}, "
        f"partitions={stats.partitions}, heapsort_fallbacks={stats.heapsort_fallbacks}, "
        f"insertion_calls={stats.insertion_calls}, trace_events={len(stats.trace)}"
    )

    if stats.trace:
        trace_df = pd.DataFrame(stats.trace)
        print("Trace table (last 12 events):")
        print(trace_df.tail(12).to_string(index=False))


def main() -> None:
    """Execute deterministic validation cases (non-interactive)."""
    fixed_case = [12, -5, 7, 7, 3.5, -11, 0, 12, 2]

    rng = np.random.default_rng(seed=20260407)
    random_case = rng.integers(low=-50, high=51, size=20).tolist()

    reverse_case = list(range(30, -1, -1))
    duplicates_case = [5, 1, 5, 5, 2, 2, 2, -3, -3, 10, 0]

    # Force depth limit to 0 so heapsort fallback path is explicitly exercised.
    fallback_demo_case = rng.integers(low=-100, high=101, size=25).tolist()

    run_case("Case 1: fixed mixed numbers", fixed_case)
    run_case("Case 2: seeded random integers", random_case)
    run_case("Case 3: reverse ordered", reverse_case)
    run_case("Case 4: many duplicates", duplicates_case)
    run_case(
        "Case 5: forced heapsort fallback",
        fallback_demo_case,
        max_depth_override=0,
    )

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()

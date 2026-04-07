"""Runnable MVP for Heap Sort (CS-0003)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HeapSortStats:
    """Execution statistics for one heap sort run."""

    comparisons: int
    swaps: int
    heapify_calls: int
    trace: list[dict[str, Any]]


def validate_numeric_sequence(values: Sequence[float]) -> list[float]:
    """Validate input and return a 1D finite float list."""
    if isinstance(values, (str, bytes)):
        raise TypeError("Input must be a numeric sequence, not string/bytes.")

    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Input must be a 1D sequence.")
    if not np.isfinite(arr).all():
        raise ValueError("Input contains non-finite values (NaN or Inf).")

    return arr.tolist()


def _sift_down(
    arr: list[float],
    start: int,
    end: int,
    trace: list[dict[str, Any]],
    counters: dict[str, int],
    phase: str,
    round_id: int,
) -> None:
    """Restore max-heap property from `start` to `end` (inclusive)."""
    root = start

    while True:
        left = 2 * root + 1
        if left > end:
            break

        largest = root

        counters["comparisons"] += 1
        if arr[left] > arr[largest]:
            largest = left

        right = left + 1
        if right <= end:
            counters["comparisons"] += 1
            if arr[right] > arr[largest]:
                largest = right

        if largest == root:
            break

        arr[root], arr[largest] = arr[largest], arr[root]
        counters["swaps"] += 1

        trace.append(
            {
                "phase": phase,
                "round": round_id,
                "root": root,
                "swapped_with": largest,
                "heap_end": end,
                "array_state": arr.copy(),
            }
        )

        root = largest


def heap_sort(values: Sequence[float]) -> tuple[list[float], HeapSortStats]:
    """Sort values in non-decreasing order using hand-written heap sort."""
    arr = validate_numeric_sequence(values)
    n = len(arr)

    counters = {"comparisons": 0, "swaps": 0, "heapify_calls": 0}
    trace: list[dict[str, Any]] = []

    if n < 2:
        return arr, HeapSortStats(comparisons=0, swaps=0, heapify_calls=0, trace=[])

    # Build max heap.
    for start in range((n - 2) // 2, -1, -1):
        counters["heapify_calls"] += 1
        _sift_down(
            arr=arr,
            start=start,
            end=n - 1,
            trace=trace,
            counters=counters,
            phase="build",
            round_id=start,
        )

    # Repeatedly extract max element to the sorted suffix.
    for end in range(n - 1, 0, -1):
        arr[0], arr[end] = arr[end], arr[0]
        counters["swaps"] += 1

        trace.append(
            {
                "phase": "extract",
                "round": n - end,
                "root": 0,
                "swapped_with": end,
                "heap_end": end - 1,
                "array_state": arr.copy(),
            }
        )

        counters["heapify_calls"] += 1
        _sift_down(
            arr=arr,
            start=0,
            end=end - 1,
            trace=trace,
            counters=counters,
            phase="restore",
            round_id=n - end,
        )

    return (
        arr,
        HeapSortStats(
            comparisons=counters["comparisons"],
            swaps=counters["swaps"],
            heapify_calls=counters["heapify_calls"],
            trace=trace,
        ),
    )


def is_non_decreasing(values: Sequence[float]) -> bool:
    """Return whether the sequence is sorted in non-decreasing order."""
    return all(values[i] <= values[i + 1] for i in range(len(values) - 1))


def run_case(case_name: str, raw_values: Sequence[float]) -> None:
    """Run one deterministic validation case and print audit-friendly output."""
    values = validate_numeric_sequence(raw_values)
    sorted_values, stats = heap_sort(values)

    expected_py = sorted(values)
    expected_np = np.sort(np.asarray(values, dtype=float)).tolist()

    if sorted_values != expected_py or sorted_values != expected_np:
        raise RuntimeError(f"{case_name}: sorted output mismatch.")

    if not is_non_decreasing(sorted_values):
        raise RuntimeError(f"{case_name}: output is not non-decreasing.")

    print(f"\n=== {case_name} ===")
    print(f"Input: {values}")
    print(f"Sorted by heap sort: {sorted_values}")
    print(f"Expected sorted:     {expected_py}")
    print(
        "Stats: "
        f"comparisons={stats.comparisons}, swaps={stats.swaps}, "
        f"heapify_calls={stats.heapify_calls}, trace_events={len(stats.trace)}"
    )

    if stats.trace:
        trace_df = pd.DataFrame(stats.trace)
        print("Trace table (last 15 events):")
        print(trace_df.tail(15).to_string(index=False))


def main() -> None:
    """Execute two non-interactive validation cases."""
    fixed_case = [9, -3, 9, 0.5, 7, -11, 4, 4]

    rng = np.random.default_rng(seed=2026)
    random_case = rng.integers(low=-30, high=31, size=12).tolist()

    run_case("Case 1: fixed mixed numbers", fixed_case)
    run_case("Case 2: seeded random integers", random_case)

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()

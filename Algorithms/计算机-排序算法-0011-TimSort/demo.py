"""Runnable MVP for TimSort (CS-0011).

This script implements a compact TimSort-style algorithm from source-level
building blocks instead of delegating to Python's built-in sorting black box.

Implemented pieces:
- natural run detection (ascending / strict descending)
- descending-run reversal
- minrun calculation
- binary insertion sort extension for short runs
- run-stack invariant maintenance and stable run merging

NumPy and pandas are used only for validation/reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimSortStats:
    """Execution statistics for one TimSort run."""

    n: int
    minrun: int
    comparisons: int
    moves: int
    swaps: int
    run_detections: int
    reversals: int
    insertion_extensions: int
    merges: int
    max_stack_size: int
    run_trace: list[dict[str, Any]]
    merge_trace: list[dict[str, Any]]


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


def _calc_minrun(n: int) -> int:
    """Return TimSort minrun (usually in [32, 64] when n >= 64)."""
    r = 0
    while n >= 64:
        r |= n & 1
        n >>= 1
    return n + r


def _swap(arr: list[float], i: int, j: int, counters: dict[str, int]) -> None:
    if i == j:
        return
    arr[i], arr[j] = arr[j], arr[i]
    counters["swaps"] += 1
    counters["moves"] += 2


def _reverse_slice(
    arr: list[float],
    lo: int,
    hi: int,
    counters: dict[str, int],
) -> None:
    """Reverse arr[lo:hi) in place."""
    i = lo
    j = hi - 1
    while i < j:
        _swap(arr, i, j, counters)
        i += 1
        j -= 1


def _count_run_and_make_ascending(
    arr: list[float],
    lo: int,
    hi: int,
    counters: dict[str, int],
) -> tuple[int, str, bool]:
    """Detect a natural run at arr[lo:hi), normalize it to ascending order."""
    run_hi = lo + 1
    if run_hi >= hi:
        return 1, "single", False

    counters["comparisons"] += 1
    if arr[run_hi] < arr[lo]:
        direction = "descending"
        run_hi += 1
        while run_hi < hi:
            counters["comparisons"] += 1
            if arr[run_hi] < arr[run_hi - 1]:
                run_hi += 1
            else:
                break

        _reverse_slice(arr, lo, run_hi, counters)
        counters["reversals"] += 1
        return run_hi - lo, direction, True

    direction = "ascending"
    run_hi += 1
    while run_hi < hi:
        counters["comparisons"] += 1
        if arr[run_hi] >= arr[run_hi - 1]:
            run_hi += 1
        else:
            break

    return run_hi - lo, direction, False


def _binary_insertion_sort(
    arr: list[float],
    lo: int,
    hi: int,
    start: int,
    counters: dict[str, int],
) -> None:
    """Stable binary insertion sort for arr[lo:hi), assuming arr[lo:start) sorted."""
    if start <= lo:
        start = lo + 1

    for i in range(start, hi):
        pivot = arr[i]
        left = lo
        right = i

        while left < right:
            mid = left + (right - left) // 2
            counters["comparisons"] += 1
            if pivot < arr[mid]:
                right = mid
            else:
                left = mid + 1

        j = i
        while j > left:
            arr[j] = arr[j - 1]
            counters["moves"] += 1
            j -= 1

        arr[left] = pivot
        counters["moves"] += 1


def _merge_at(
    arr: list[float],
    stack: list[tuple[int, int]],
    i: int,
    counters: dict[str, int],
    merge_trace: list[dict[str, Any]],
) -> None:
    """Merge stack[i] and stack[i+1] stably; update stack in place."""
    left_start, left_len = stack[i]
    right_start, right_len = stack[i + 1]

    if left_start + left_len != right_start:
        raise RuntimeError("Run stack lost contiguity.")

    left_buf = arr[left_start:right_start]
    counters["moves"] += left_len  # copy into temporary buffer

    li = 0
    ri = right_start
    dest = left_start
    right_end = right_start + right_len

    while li < left_len and ri < right_end:
        counters["comparisons"] += 1
        if left_buf[li] <= arr[ri]:
            arr[dest] = left_buf[li]
            li += 1
        else:
            arr[dest] = arr[ri]
            ri += 1
        dest += 1
        counters["moves"] += 1

    if li < left_len:
        remain = left_len - li
        arr[dest : dest + remain] = left_buf[li:left_len]
        counters["moves"] += remain

    merged_len = left_len + right_len
    stack[i] = (left_start, merged_len)
    del stack[i + 1]

    counters["merges"] += 1
    merge_trace.append(
        {
            "left_start": left_start,
            "left_len": left_len,
            "right_start": right_start,
            "right_len": right_len,
            "merged_len": merged_len,
            "stack_size_after": len(stack),
        }
    )


def _merge_collapse(
    arr: list[float],
    stack: list[tuple[int, int]],
    counters: dict[str, int],
    merge_trace: list[dict[str, Any]],
) -> None:
    """Maintain TimSort run stack invariants after each push."""
    while len(stack) > 1:
        n = len(stack)

        if n >= 3 and stack[n - 3][1] <= stack[n - 2][1] + stack[n - 1][1]:
            if stack[n - 3][1] < stack[n - 1][1]:
                _merge_at(arr, stack, n - 3, counters, merge_trace)
            else:
                _merge_at(arr, stack, n - 2, counters, merge_trace)
        elif stack[n - 2][1] <= stack[n - 1][1]:
            _merge_at(arr, stack, n - 2, counters, merge_trace)
        else:
            break


def _merge_force_collapse(
    arr: list[float],
    stack: list[tuple[int, int]],
    counters: dict[str, int],
    merge_trace: list[dict[str, Any]],
) -> None:
    """Merge remaining runs until one full run remains."""
    while len(stack) > 1:
        n = len(stack)
        if n >= 3 and stack[n - 3][1] < stack[n - 1][1]:
            _merge_at(arr, stack, n - 3, counters, merge_trace)
        else:
            _merge_at(arr, stack, n - 2, counters, merge_trace)


def timsort(
    values: Sequence[float],
    minrun_override: int | None = None,
) -> tuple[list[float], TimSortStats]:
    """Sort values in non-decreasing order using a TimSort-style implementation."""
    arr = validate_numeric_sequence(values)
    n = len(arr)

    if n <= 1:
        stats = TimSortStats(
            n=n,
            minrun=n,
            comparisons=0,
            moves=0,
            swaps=0,
            run_detections=0,
            reversals=0,
            insertion_extensions=0,
            merges=0,
            max_stack_size=1 if n == 1 else 0,
            run_trace=[],
            merge_trace=[],
        )
        return arr, stats

    minrun = minrun_override if minrun_override is not None else _calc_minrun(n)
    if minrun < 2:
        raise ValueError("minrun_override must be >= 2")

    counters = {
        "comparisons": 0,
        "moves": 0,
        "swaps": 0,
        "run_detections": 0,
        "reversals": 0,
        "insertion_extensions": 0,
        "merges": 0,
    }

    run_trace: list[dict[str, Any]] = []
    merge_trace: list[dict[str, Any]] = []
    stack: list[tuple[int, int]] = []
    max_stack_size = 0

    lo = 0
    while lo < n:
        detected_len, direction, reversed_flag = _count_run_and_make_ascending(
            arr,
            lo,
            n,
            counters,
        )

        forced_len = min(minrun, n - lo)
        final_len = detected_len
        extended = False

        if detected_len < forced_len:
            _binary_insertion_sort(
                arr,
                lo=lo,
                hi=lo + forced_len,
                start=lo + detected_len,
                counters=counters,
            )
            final_len = forced_len
            counters["insertion_extensions"] += 1
            extended = True

        stack.append((lo, final_len))
        counters["run_detections"] += 1
        max_stack_size = max(max_stack_size, len(stack))

        run_trace.append(
            {
                "run_start": lo,
                "detected_len": detected_len,
                "final_len": final_len,
                "direction": direction,
                "reversed": reversed_flag,
                "extended_by_insertion": extended,
                "stack_size_after_push": len(stack),
            }
        )

        _merge_collapse(arr, stack, counters, merge_trace)
        lo += final_len

    _merge_force_collapse(arr, stack, counters, merge_trace)

    if len(stack) != 1 or stack[0] != (0, n):
        raise RuntimeError("Final run stack is invalid after merge collapse.")

    stats = TimSortStats(
        n=n,
        minrun=minrun,
        comparisons=counters["comparisons"],
        moves=counters["moves"],
        swaps=counters["swaps"],
        run_detections=counters["run_detections"],
        reversals=counters["reversals"],
        insertion_extensions=counters["insertion_extensions"],
        merges=counters["merges"],
        max_stack_size=max_stack_size,
        run_trace=run_trace,
        merge_trace=merge_trace,
    )
    return arr, stats


def is_non_decreasing(values: Sequence[float]) -> bool:
    """Return whether values are in non-decreasing order."""
    return all(values[i] <= values[i + 1] for i in range(len(values) - 1))


def run_case(
    case_name: str,
    raw_values: Sequence[float],
    minrun_override: int | None = None,
) -> None:
    """Run one deterministic case and print audit-friendly output."""
    values = validate_numeric_sequence(raw_values)
    sorted_values, stats = timsort(values, minrun_override=minrun_override)

    expected_py = sorted(values)
    expected_np = np.sort(np.asarray(values, dtype=float)).tolist()

    if sorted_values != expected_py or sorted_values != expected_np:
        raise RuntimeError(f"{case_name}: sorted output mismatch.")
    if not is_non_decreasing(sorted_values):
        raise RuntimeError(f"{case_name}: output is not non-decreasing.")

    print(f"\n=== {case_name} ===")
    print(f"Input: {values}")
    print(f"Sorted by timsort: {sorted_values}")
    print(f"Expected sorted:   {expected_py}")
    print(
        "Stats: "
        f"n={stats.n}, minrun={stats.minrun}, "
        f"comparisons={stats.comparisons}, moves={stats.moves}, swaps={stats.swaps}, "
        f"run_detections={stats.run_detections}, reversals={stats.reversals}, "
        f"insertion_extensions={stats.insertion_extensions}, merges={stats.merges}, "
        f"max_stack_size={stats.max_stack_size}"
    )

    if stats.run_trace:
        run_df = pd.DataFrame(stats.run_trace)
        print("Run trace (last 10 rows):")
        print(run_df.tail(10).to_string(index=False))

    if stats.merge_trace:
        merge_df = pd.DataFrame(stats.merge_trace)
        print("Merge trace (last 10 rows):")
        print(merge_df.tail(10).to_string(index=False))


def main() -> None:
    """Execute deterministic validation cases (non-interactive)."""
    fixed_case = [9, 1, 5, 3, 7, 3, -2, 6, 8, 0]

    nearly_sorted_case = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        15,
        16,
        17,
        18,
        10,
        11,
        12,
        13,
        14,
        19,
        20,
    ]

    reverse_case = list(range(32, -1, -1))
    duplicate_heavy_case = [5, 5, 1, 1, 1, 3, 3, 2, 2, 2, 9, 9, 0, 0, -1, -1]

    rng = np.random.default_rng(seed=20260407)
    random_case = rng.integers(low=-100, high=101, size=40).tolist()

    run_case("Case 1: fixed mixed numbers", fixed_case)
    run_case("Case 2: nearly sorted with local disorder", nearly_sorted_case)
    run_case("Case 3: reverse ordered", reverse_case)
    run_case("Case 4: duplicate-heavy", duplicate_heavy_case)
    run_case("Case 5: seeded random integers", random_case, minrun_override=16)

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()

"""Runnable MVP for Selection Sort (CS-0005)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SelectionSortStats:
    """Execution statistics for one selection sort run."""

    comparisons: int
    swaps: int
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


def selection_sort(values: Sequence[float]) -> tuple[list[float], SelectionSortStats]:
    """Sort values in non-decreasing order via hand-written selection sort."""
    arr = validate_numeric_sequence(values)
    n = len(arr)
    comparisons = 0
    swaps = 0
    trace: list[dict[str, Any]] = []

    for i in range(max(n - 1, 0)):
        min_idx = i
        for j in range(i + 1, n):
            comparisons += 1
            if arr[j] < arr[min_idx]:
                min_idx = j

        swapped = False
        if min_idx != i:
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
            swaps += 1
            swapped = True

        trace.append(
            {
                "i": i,
                "min_index": min_idx,
                "min_value": arr[i],
                "swapped": swapped,
                "array_state": arr.copy(),
            }
        )

    return arr, SelectionSortStats(comparisons=comparisons, swaps=swaps, trace=trace)


def theoretical_comparisons(n: int) -> int:
    """Exact number of comparisons in selection sort for length n."""
    return n * (n - 1) // 2 if n >= 2 else 0


def run_case(case_name: str, raw_values: Sequence[float]) -> None:
    """Run one deterministic validation case and print audit-friendly output."""
    values = validate_numeric_sequence(raw_values)
    sorted_values, stats = selection_sort(values)

    expected_py = sorted(values)
    expected_np = np.sort(np.asarray(values, dtype=float)).tolist()

    if sorted_values != expected_py or sorted_values != expected_np:
        raise RuntimeError(f"{case_name}: sorted output mismatch.")

    expected_comparisons = theoretical_comparisons(len(values))
    if stats.comparisons != expected_comparisons:
        raise RuntimeError(
            f"{case_name}: comparisons={stats.comparisons}, expected={expected_comparisons}."
        )

    print(f"\n=== {case_name} ===")
    print(f"Input: {values}")
    print(f"Sorted by selection sort: {sorted_values}")
    print(f"Expected sorted:         {expected_py}")
    print(
        "Stats: "
        f"comparisons={stats.comparisons}, swaps={stats.swaps}, "
        f"rounds={len(stats.trace)}"
    )

    if stats.trace:
        trace_df = pd.DataFrame(stats.trace)
        print("Trace table:")
        print(trace_df.to_string(index=False))


def main() -> None:
    """Execute two non-interactive validation cases."""
    fixed_case = [7, -2, 7, 0.5, -10, 3, 3]

    rng = np.random.default_rng(seed=2026)
    random_case = rng.integers(low=-20, high=21, size=10).tolist()

    run_case("Case 1: fixed mixed numbers", fixed_case)
    run_case("Case 2: seeded random integers", random_case)

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()

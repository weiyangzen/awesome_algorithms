"""Runnable MVP for Radix Sort (CS-0009).

This script provides an auditable LSD radix-sort implementation for integer
sequences, including negative numbers via monotonic shifting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RadixSortStats:
    """Execution statistics for one radix-sort run."""

    base: int
    shift: int
    digit_passes: int
    bucket_writes: int
    trace: list[dict[str, Any]]


def validate_integer_sequence(values: Sequence[int]) -> list[int]:
    """Validate input and return a 1D finite integer list."""
    if isinstance(values, (str, bytes)):
        raise TypeError("Input must be a numeric sequence, not string/bytes.")

    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Input must be a 1D sequence.")
    if not np.isfinite(arr).all():
        raise ValueError("Input contains non-finite values (NaN or Inf).")

    rounded = np.round(arr)
    if not np.allclose(arr, rounded):
        raise ValueError("Radix sort MVP accepts integers only.")

    return rounded.astype(int).tolist()


def _stable_counting_pass(values: list[int], exp: int, base: int) -> tuple[list[int], list[int]]:
    """Stable counting distribution for one digit (defined by exp/base)."""
    counts = [0] * base
    for x in values:
        digit = (x // exp) % base
        counts[digit] += 1

    histogram = counts.copy()

    for i in range(1, base):
        counts[i] += counts[i - 1]

    output = [0] * len(values)
    for i in range(len(values) - 1, -1, -1):
        digit = (values[i] // exp) % base
        counts[digit] -= 1
        output[counts[digit]] = values[i]

    return output, histogram


def radix_sort_lsd(values: Sequence[int], base: int = 10) -> tuple[list[int], RadixSortStats]:
    """Sort integers in non-decreasing order via LSD radix sort."""
    if base < 2:
        raise ValueError("base must be >= 2.")

    arr = validate_integer_sequence(values)
    n = len(arr)

    if n <= 1:
        return (
            arr.copy(),
            RadixSortStats(base=base, shift=0, digit_passes=0, bucket_writes=0, trace=[]),
        )

    min_val = min(arr)
    shift = -min_val if min_val < 0 else 0
    shifted = [x + shift for x in arr]

    max_val = max(shifted)
    exp = 1
    digit_passes = 0
    bucket_writes = 0
    trace: list[dict[str, Any]] = []

    while max_val // exp > 0:
        shifted, histogram = _stable_counting_pass(shifted, exp=exp, base=base)
        digit_passes += 1
        bucket_writes += n
        trace.append(
            {
                "pass": digit_passes,
                "exp": exp,
                "bucket_hist": histogram,
                "array_state": shifted.copy(),
            }
        )
        exp *= base

    sorted_values = [x - shift for x in shifted]
    return (
        sorted_values,
        RadixSortStats(
            base=base,
            shift=shift,
            digit_passes=digit_passes,
            bucket_writes=bucket_writes,
            trace=trace,
        ),
    )


def run_case(case_name: str, raw_values: Sequence[int]) -> None:
    """Run one deterministic validation case and print audit-friendly output."""
    values = validate_integer_sequence(raw_values)
    sorted_values, stats = radix_sort_lsd(values, base=10)

    expected_py = sorted(values)
    expected_np = np.sort(np.asarray(values, dtype=np.int64)).astype(int).tolist()

    if sorted_values != expected_py or sorted_values != expected_np:
        raise RuntimeError(f"{case_name}: sorted output mismatch.")

    print(f"\n=== {case_name} ===")
    print(f"Input: {values}")
    print(f"Sorted by radix sort: {sorted_values}")
    print(f"Expected sorted:      {expected_py}")
    print(
        "Stats: "
        f"base={stats.base}, shift={stats.shift}, "
        f"digit_passes={stats.digit_passes}, bucket_writes={stats.bucket_writes}"
    )

    if stats.trace:
        trace_df = pd.DataFrame(stats.trace)
        print("Trace table:")
        print(trace_df.to_string(index=False))


def main() -> None:
    """Execute non-interactive validation cases for radix sort MVP."""
    fixed_case = [170, 45, 75, -90, 802, 24, 2, 66, -90, 0]

    rng = np.random.default_rng(seed=2026)
    random_case = rng.integers(low=-999, high=1000, size=12).tolist()

    run_case("Case 1: fixed mixed integers", fixed_case)
    run_case("Case 2: seeded random integers", random_case)
    run_case("Case 3: empty list", [])
    run_case("Case 4: single element", [42])

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()

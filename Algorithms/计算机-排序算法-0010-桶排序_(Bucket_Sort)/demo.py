"""Runnable MVP for Bucket Sort (CS-0010)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BucketSortStats:
    """Execution statistics for one bucket sort run."""

    bucket_count: int
    bucket_histogram: list[int]
    non_empty_buckets: int
    max_bucket_size: int
    insertion_comparisons: int
    distribution_trace: list[dict[str, Any]]


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


def insertion_sort_inplace(bucket: list[float]) -> int:
    """In-place insertion sort used inside each bucket."""
    comparisons = 0
    for i in range(1, len(bucket)):
        key = bucket[i]
        j = i - 1
        while j >= 0:
            comparisons += 1
            if bucket[j] <= key:
                break
            bucket[j + 1] = bucket[j]
            j -= 1
        bucket[j + 1] = key
    return comparisons


def value_to_bucket_index(
    value: float, value_min: float, value_max: float, bucket_count: int
) -> int:
    """Map one value to its bucket index."""
    if value_max == value_min:
        return 0

    normalized = (value - value_min) / (value_max - value_min)
    idx = int(normalized * bucket_count)
    if idx >= bucket_count:
        idx = bucket_count - 1
    if idx < 0:
        idx = 0
    return idx


def bucket_sort(
    values: Sequence[float], bucket_count: int | None = None
) -> tuple[list[float], BucketSortStats]:
    """Sort values in non-decreasing order via bucket sort."""
    arr = validate_numeric_sequence(values)
    n = len(arr)

    if n <= 1:
        stats = BucketSortStats(
            bucket_count=1,
            bucket_histogram=[n],
            non_empty_buckets=1 if n == 1 else 0,
            max_bucket_size=n,
            insertion_comparisons=0,
            distribution_trace=[
                {
                    "bucket": 0,
                    "count": n,
                    "range_low": None,
                    "range_high": None,
                    "bucket_min": arr[0] if n == 1 else None,
                    "bucket_max": arr[0] if n == 1 else None,
                }
            ],
        )
        return arr, stats

    if bucket_count is None:
        bucket_count = max(1, int(np.sqrt(n)))
    if bucket_count <= 0:
        raise ValueError("bucket_count must be a positive integer.")

    value_min = min(arr)
    value_max = max(arr)
    value_range = value_max - value_min

    buckets: list[list[float]] = [[] for _ in range(bucket_count)]

    for value in arr:
        idx = value_to_bucket_index(value, value_min, value_max, bucket_count)
        buckets[idx].append(value)

    insertion_comparisons = 0
    sorted_values: list[float] = []
    for bucket in buckets:
        insertion_comparisons += insertion_sort_inplace(bucket)
        sorted_values.extend(bucket)

    bucket_histogram = [len(bucket) for bucket in buckets]
    non_empty_buckets = sum(count > 0 for count in bucket_histogram)
    max_bucket_size = max(bucket_histogram) if bucket_histogram else 0

    distribution_trace: list[dict[str, Any]] = []
    for i, bucket in enumerate(buckets):
        if value_range == 0:
            range_low = value_min
            range_high = value_max
        else:
            width = value_range / bucket_count
            range_low = value_min + i * width
            if i == bucket_count - 1:
                range_high = value_max
            else:
                range_high = value_min + (i + 1) * width

        distribution_trace.append(
            {
                "bucket": i,
                "count": len(bucket),
                "range_low": range_low,
                "range_high": range_high,
                "bucket_min": min(bucket) if bucket else None,
                "bucket_max": max(bucket) if bucket else None,
            }
        )

    stats = BucketSortStats(
        bucket_count=bucket_count,
        bucket_histogram=bucket_histogram,
        non_empty_buckets=non_empty_buckets,
        max_bucket_size=max_bucket_size,
        insertion_comparisons=insertion_comparisons,
        distribution_trace=distribution_trace,
    )
    return sorted_values, stats


def run_case(case_name: str, raw_values: Sequence[float], bucket_count: int | None) -> None:
    """Run one deterministic validation case and print audit-friendly output."""
    values = validate_numeric_sequence(raw_values)
    sorted_values, stats = bucket_sort(values, bucket_count=bucket_count)

    expected_py = sorted(values)
    expected_np = np.sort(np.asarray(values, dtype=float)).tolist()

    if sorted_values != expected_py or sorted_values != expected_np:
        raise RuntimeError(f"{case_name}: sorted output mismatch.")

    print(f"\n=== {case_name} ===")
    print(f"Input:                 {values}")
    print(f"Sorted by bucket sort: {sorted_values}")
    print(f"Expected sorted:       {expected_py}")
    print(
        "Stats: "
        f"bucket_count={stats.bucket_count}, "
        f"non_empty_buckets={stats.non_empty_buckets}, "
        f"max_bucket_size={stats.max_bucket_size}, "
        f"insertion_comparisons={stats.insertion_comparisons}"
    )

    trace_df = pd.DataFrame(stats.distribution_trace)
    print("Bucket distribution:")
    print(trace_df.to_string(index=False))


def main() -> None:
    """Execute non-interactive validation cases for bucket sort."""
    fixed_case = [0.78, 0.17, 0.39, 0.26, 0.72, 0.94, 0.21, 0.12, 0.23, 0.68]

    rng = np.random.default_rng(seed=2026)
    random_case = rng.uniform(low=-50.0, high=50.0, size=18).round(4).tolist()
    repeated_case = [5.0, 5.0, 5.0, 5.0, 5.0]

    run_case("Case 1: classic [0,1) style floats", fixed_case, bucket_count=5)
    run_case("Case 2: seeded random mixed range", random_case, bucket_count=None)
    run_case("Case 3: all equal values", repeated_case, bucket_count=4)

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()

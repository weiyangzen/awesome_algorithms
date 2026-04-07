"""Runnable MVP for Counting Sort (CS-0008)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CountingSortStats:
    """Execution statistics for one counting sort run."""

    n: int
    min_value: int | None
    max_value: int | None
    range_size: int
    count_ops: int
    placement_ops: int
    counts: list[int]
    prefix_counts: list[int]


def validate_integer_sequence(values: Sequence[int]) -> list[int]:
    """Validate input as a 1D finite integer sequence and return Python ints."""
    if isinstance(values, (str, bytes)):
        raise TypeError("Input must be a numeric sequence, not string/bytes.")

    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Input must be a 1D sequence.")
    if not np.isfinite(arr).all():
        raise ValueError("Input contains non-finite values (NaN or Inf).")

    rounded = np.rint(arr)
    if not np.allclose(arr, rounded):
        raise ValueError("Counting sort in this demo only accepts integer values.")

    return rounded.astype(np.int64).tolist()


def counting_sort(
    values: Sequence[int], *, max_range_size: int = 200_000
) -> tuple[list[int], CountingSortStats]:
    """Stable counting sort supporting negative integers via value-offset indexing."""
    arr = validate_integer_sequence(values)
    n = len(arr)

    if n == 0:
        return (
            [],
            CountingSortStats(
                n=0,
                min_value=None,
                max_value=None,
                range_size=0,
                count_ops=0,
                placement_ops=0,
                counts=[],
                prefix_counts=[],
            ),
        )

    min_value = min(arr)
    max_value = max(arr)
    range_size = max_value - min_value + 1

    if range_size > max_range_size:
        raise ValueError(
            f"Range size {range_size} exceeds max_range_size={max_range_size}; "
            "counting sort is not memory-efficient for this input range."
        )

    counts = [0] * range_size
    count_ops = 0
    for x in arr:
        counts[x - min_value] += 1
        count_ops += 1

    prefix = counts.copy()
    running = 0
    for i, c in enumerate(prefix):
        running += c
        prefix[i] = running

    output = [0] * n
    placement_ops = 0
    for i in range(n - 1, -1, -1):
        x = arr[i]
        idx = x - min_value
        prefix[idx] -= 1
        pos = prefix[idx]
        output[pos] = x
        placement_ops += 1

    stats = CountingSortStats(
        n=n,
        min_value=min_value,
        max_value=max_value,
        range_size=range_size,
        count_ops=count_ops,
        placement_ops=placement_ops,
        counts=counts,
        prefix_counts=np.cumsum(np.asarray(counts, dtype=np.int64)).tolist(),
    )
    return output, stats


def build_frequency_table(stats: CountingSortStats) -> pd.DataFrame:
    """Build a compact frequency table for non-zero counts."""
    if stats.range_size == 0 or stats.min_value is None:
        return pd.DataFrame(columns=["value", "count", "prefix_end"])

    values = np.arange(stats.min_value, stats.max_value + 1)
    df = pd.DataFrame(
        {
            "value": values,
            "count": stats.counts,
            "prefix_end": stats.prefix_counts,
        }
    )
    return df[df["count"] > 0].reset_index(drop=True)


def run_case(case_name: str, raw_values: Sequence[int]) -> None:
    """Run one deterministic case and print audit-friendly output."""
    values = validate_integer_sequence(raw_values)
    sorted_values, stats = counting_sort(values)

    expected_py = sorted(values)
    expected_np = np.sort(np.asarray(values, dtype=np.int64)).tolist()

    if sorted_values != expected_py or sorted_values != expected_np:
        raise RuntimeError(f"{case_name}: sorted output mismatch.")
    if stats.count_ops != len(values) or stats.placement_ops != len(values):
        raise RuntimeError(
            f"{case_name}: invalid op counters (count_ops={stats.count_ops}, "
            f"placement_ops={stats.placement_ops}, n={len(values)})."
        )

    print(f"\n=== {case_name} ===")
    print(f"Input: {values}")
    print(f"Sorted by counting sort: {sorted_values}")
    print(f"Expected sorted:         {expected_py}")
    print(
        "Stats: "
        f"n={stats.n}, min={stats.min_value}, max={stats.max_value}, "
        f"range_size={stats.range_size}, count_ops={stats.count_ops}, "
        f"placement_ops={stats.placement_ops}"
    )

    freq_table = build_frequency_table(stats)
    if not freq_table.empty:
        print("Frequency table (non-zero counts):")
        print(freq_table.to_string(index=False))


def main() -> None:
    """Execute deterministic non-interactive counting sort validation cases."""
    fixed_case = [4, 2, 2, 8, 3, 3, 1]
    negative_case = [0, -5, 2, -1, 2, -5, 3]

    rng = np.random.default_rng(seed=2026)
    random_case = rng.integers(low=-10, high=11, size=15).tolist()

    run_case("Case 1: fixed non-negative", fixed_case)
    run_case("Case 2: fixed with negatives", negative_case)
    run_case("Case 3: seeded random integers", random_case)
    run_case("Case 4: empty input", [])
    run_case("Case 5: single element", [42])

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()

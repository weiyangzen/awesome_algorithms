"""Greedy merge-intervals MVP for CS-0074.

Run:
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

Interval = tuple[int, int]


@dataclass(frozen=True)
class FixedCase:
    """Deterministic test case for merge-intervals."""

    name: str
    intervals: list[Interval]
    expected: list[Interval]


def _normalize_and_validate(intervals: Iterable[Sequence[int]]) -> list[Interval]:
    """Validate input format and convert to a normalized list of int tuples."""
    normalized: list[Interval] = []
    for idx, item in enumerate(intervals):
        if len(item) != 2:
            raise ValueError(f"Interval at index {idx} must have exactly 2 endpoints: {item!r}")
        start, end = int(item[0]), int(item[1])
        if start > end:
            raise ValueError(
                f"Interval at index {idx} is invalid because start > end: {(start, end)}"
            )
        normalized.append((start, end))
    return normalized


def merge_intervals(intervals: Iterable[Sequence[int]]) -> list[Interval]:
    """Merge overlapping closed intervals using sort + one pass.

    Overlap rule (closed intervals):
    - [a, b] and [c, d] overlap when c <= b.
    """
    normalized = _normalize_and_validate(intervals)
    if not normalized:
        return []

    ordered = sorted(normalized, key=lambda x: (x[0], x[1]))
    merged: list[Interval] = []

    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end

    merged.append((current_start, current_end))
    return merged


def merge_intervals_numpy(interval_array: np.ndarray) -> list[Interval]:
    """Bridge function: accept a numpy n x 2 array, return merged intervals."""
    arr = np.asarray(interval_array)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected shape (n, 2), got {arr.shape}")
    return merge_intervals(arr.tolist())


def _is_sorted_and_disjoint(intervals: Sequence[Interval]) -> bool:
    for i in range(1, len(intervals)):
        prev_s, prev_e = intervals[i - 1]
        curr_s, _ = intervals[i]
        if prev_s > curr_s:
            return False
        if prev_e >= curr_s:
            return False
    return True


def _every_original_interval_is_covered(
    original: Sequence[Interval], merged: Sequence[Interval]
) -> bool:
    for s, e in original:
        covered = any(ms <= s and e <= me for ms, me in merged)
        if not covered:
            return False
    return True


def assert_valid_merge(original: Sequence[Interval], merged: Sequence[Interval]) -> None:
    """MVP-level validation of merge result shape and coverage."""
    assert _is_sorted_and_disjoint(merged), f"Merged result is not sorted/disjoint: {merged}"
    assert _every_original_interval_is_covered(
        original, merged
    ), f"Merged result does not cover all original intervals: original={original}, merged={merged}"


def run_fixed_cases() -> None:
    cases = [
        FixedCase(
            name="basic overlap",
            intervals=[(1, 3), (2, 6), (8, 10), (15, 18)],
            expected=[(1, 6), (8, 10), (15, 18)],
        ),
        FixedCase(
            name="touching boundaries",
            intervals=[(1, 4), (4, 5)],
            expected=[(1, 5)],
        ),
        FixedCase(
            name="chain overlap",
            intervals=[(1, 2), (2, 4), (3, 8), (10, 12), (11, 13)],
            expected=[(1, 8), (10, 13)],
        ),
        FixedCase(
            name="unsorted and nested",
            intervals=[(5, 7), (1, 10), (2, 3), (11, 11)],
            expected=[(1, 10), (11, 11)],
        ),
    ]

    print("=== Fixed Cases ===")
    for i, case in enumerate(cases, start=1):
        merged = merge_intervals(case.intervals)
        assert merged == case.expected, (
            f"Case {case.name} failed: expected={case.expected}, got={merged}"
        )
        assert_valid_merge(case.intervals, merged)
        print(f"[{i}] {case.name}: {case.intervals} -> {merged}")


def run_numpy_case() -> None:
    rng = np.random.default_rng(2026)
    starts = rng.integers(-3, 12, size=8)
    lengths = rng.integers(0, 6, size=8)
    interval_array = np.column_stack((starts, starts + lengths))

    merged = merge_intervals_numpy(interval_array)
    original = [tuple(map(int, row)) for row in interval_array.tolist()]
    assert_valid_merge(original, merged)

    print("\n=== Numpy Case ===")
    print(f"input array:\n{interval_array}")
    print(f"merged: {merged}")


def main() -> None:
    run_fixed_cases()
    run_numpy_case()
    print("\nAll checks passed for CS-0074 (合并区间).")


if __name__ == "__main__":
    main()

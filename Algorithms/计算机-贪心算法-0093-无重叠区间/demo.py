"""Greedy MVP for CS-0073: 无重叠区间.

Run:
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence

import numpy as np

Interval = tuple[int, int]


@dataclass(frozen=True)
class FixedCase:
    """Deterministic test case for erase-overlap-intervals."""

    name: str
    intervals: list[Interval]
    expected_removed: int


def _normalize_and_validate(intervals: Iterable[Sequence[int]]) -> list[Interval]:
    """Validate interval format and convert into integer tuples."""
    normalized: list[Interval] = []
    for idx, item in enumerate(intervals):
        if len(item) != 2:
            raise ValueError(f"Interval at index {idx} must have 2 endpoints: {item!r}")

        start, end = int(item[0]), int(item[1])
        if start > end:
            raise ValueError(
                f"Interval at index {idx} is invalid because start > end: {(start, end)}"
            )
        normalized.append((start, end))

    return normalized


def is_non_overlapping(intervals: Sequence[Interval]) -> bool:
    """Check if intervals are pairwise non-overlapping under rule: next.start >= prev.end."""
    if not intervals:
        return True

    ordered = sorted(intervals, key=lambda x: (x[0], x[1]))
    for i in range(1, len(ordered)):
        prev_start, prev_end = ordered[i - 1]
        curr_start, _ = ordered[i]
        _ = prev_start  # keep variable unpacking explicit for readability
        if curr_start < prev_end:
            return False
    return True


def select_max_non_overlapping(intervals: Iterable[Sequence[int]]) -> list[Interval]:
    """Select a maximum-size non-overlapping subset via end-time greedy."""
    normalized = _normalize_and_validate(intervals)
    if not normalized:
        return []

    ordered = sorted(normalized, key=lambda x: (x[1], x[0]))

    selected: list[Interval] = []
    last_end: int | None = None

    for start, end in ordered:
        if last_end is None or start >= last_end:
            selected.append((start, end))
            last_end = end

    return selected


def erase_overlap_intervals(intervals: Iterable[Sequence[int]]) -> int:
    """Return minimum number of intervals to remove so that remaining do not overlap."""
    normalized = _normalize_and_validate(intervals)
    kept = select_max_non_overlapping(normalized)
    return len(normalized) - len(kept)


def max_non_overlap_bruteforce(intervals: Iterable[Sequence[int]]) -> int:
    """Bruteforce maximum keep count (for correctness checks on small cases)."""
    normalized = _normalize_and_validate(intervals)
    n = len(normalized)

    best = 0
    for r in range(n + 1):
        for idxs in combinations(range(n), r):
            subset = [normalized[i] for i in idxs]
            if is_non_overlapping(subset):
                best = max(best, len(subset))
    return best


def assert_greedy_is_optimal(intervals: Sequence[Interval]) -> None:
    """Verify greedy deletion count equals bruteforce optimum deletion count."""
    greedy_removed = erase_overlap_intervals(intervals)
    optimal_keep = max_non_overlap_bruteforce(intervals)
    optimal_removed = len(intervals) - optimal_keep

    assert greedy_removed == optimal_removed, (
        "Greedy mismatch: "
        f"intervals={intervals}, greedy_removed={greedy_removed}, optimal_removed={optimal_removed}"
    )


def run_fixed_cases() -> None:
    cases = [
        FixedCase(
            name="basic overlap",
            intervals=[(1, 2), (2, 3), (3, 4), (1, 3)],
            expected_removed=1,
        ),
        FixedCase(
            name="all touching",
            intervals=[(1, 2), (2, 3), (3, 5)],
            expected_removed=0,
        ),
        FixedCase(
            name="nested intervals",
            intervals=[(1, 10), (2, 3), (3, 4), (4, 5)],
            expected_removed=1,
        ),
        FixedCase(
            name="all overlapping chain",
            intervals=[(1, 4), (2, 5), (3, 6), (4, 7)],
            expected_removed=2,
        ),
        FixedCase(
            name="already non-overlapping",
            intervals=[(-3, -1), (0, 1), (1, 2), (5, 8)],
            expected_removed=0,
        ),
        FixedCase(
            name="unsorted mixed",
            intervals=[(5, 7), (1, 2), (2, 6), (7, 8), (3, 4)],
            expected_removed=1,
        ),
    ]

    print("=== Fixed Cases ===")
    for i, case in enumerate(cases, start=1):
        removed = erase_overlap_intervals(case.intervals)
        kept = select_max_non_overlapping(case.intervals)

        assert removed == case.expected_removed, (
            f"Case {case.name} failed: expected={case.expected_removed}, got={removed}"
        )
        assert is_non_overlapping(kept), f"Kept set overlaps unexpectedly: {kept}"
        assert_greedy_is_optimal(case.intervals)

        print(
            f"[{i}] {case.name}: intervals={case.intervals}, removed={removed}, kept={kept}"
        )


def run_numpy_case() -> None:
    rng = np.random.default_rng(73)
    starts = rng.integers(-2, 10, size=8)
    lengths = rng.integers(1, 6, size=8)
    interval_array = np.column_stack((starts, starts + lengths))

    intervals = [tuple(map(int, row)) for row in interval_array.tolist()]
    removed = erase_overlap_intervals(intervals)
    kept = select_max_non_overlapping(intervals)

    assert is_non_overlapping(kept)
    assert_greedy_is_optimal(intervals)

    print("\n=== Numpy Case ===")
    print(f"input array:\n{interval_array}")
    print(f"removed={removed}")
    print(f"kept={kept}")


def main() -> None:
    run_fixed_cases()
    run_numpy_case()
    print("\nAll checks passed for CS-0073 (无重叠区间).")


if __name__ == "__main__":
    main()

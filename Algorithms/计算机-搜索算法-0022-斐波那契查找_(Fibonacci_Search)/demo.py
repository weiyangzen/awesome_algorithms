"""Fibonacci Search MVP demo.

Run:
    uv run python demo.py
"""

from __future__ import annotations

from bisect import bisect_left
from typing import Sequence


def _ensure_non_decreasing(arr: Sequence[int]) -> None:
    """Raise ValueError if arr is not sorted in non-decreasing order."""
    for i in range(1, len(arr)):
        if arr[i] < arr[i - 1]:
            raise ValueError(
                f"Input must be sorted in non-decreasing order: arr[{i - 1}]={arr[i - 1]} > arr[{i}]={arr[i]}"
            )


def fibonacci_search(arr: Sequence[int], target: int) -> int:
    """Return an index of target in sorted arr, or -1 if absent."""
    _ensure_non_decreasing(arr)

    n = len(arr)
    if n == 0:
        return -1

    # Build the smallest Fibonacci number >= n.
    fib_mm2 = 0  # (m-2)th Fibonacci
    fib_mm1 = 1  # (m-1)th Fibonacci
    fib_m = fib_mm1 + fib_mm2  # mth Fibonacci

    while fib_m < n:
        fib_mm2 = fib_mm1
        fib_mm1 = fib_m
        fib_m = fib_mm1 + fib_mm2

    offset = -1

    # While we still have a valid Fibonacci window.
    while fib_m > 1:
        i = min(offset + fib_mm2, n - 1)
        value = arr[i]

        if value < target:
            # Move three Fibonacci variables one step down.
            fib_m = fib_mm1
            fib_mm1 = fib_mm2
            fib_mm2 = fib_m - fib_mm1
            offset = i
        elif value > target:
            # Cut the right part, keep the left Fibonacci partition.
            fib_m = fib_mm2
            fib_mm1 = fib_mm1 - fib_mm2
            fib_mm2 = fib_m - fib_mm1
        else:
            return i

    # Check if the last candidate can match.
    last = offset + 1
    if fib_mm1 and last < n and arr[last] == target:
        return last

    return -1


def _expected_via_bisect(arr: Sequence[int], target: int) -> int:
    """Reference answer via bisect_left; returns first match or -1."""
    pos = bisect_left(arr, target)
    if pos < len(arr) and arr[pos] == target:
        return pos
    return -1


def main() -> None:
    cases = [
        ([], 3),
        ([5], 5),
        ([5], 7),
        ([1, 3, 5, 7, 9], 1),
        ([1, 3, 5, 7, 9], 9),
        ([1, 3, 5, 7, 9], 8),
        ([10, 22, 35, 40, 45, 50, 80, 82, 85, 90, 100], 85),
        ([10, 22, 35, 40, 45, 50, 80, 82, 85, 90, 100], 10),
        ([10, 22, 35, 40, 45, 50, 80, 82, 85, 90, 100], 101),
        ([-11, -4, -1, 0, 2, 9, 15], -4),
        ([-11, -4, -1, 0, 2, 9, 15], 8),
        ([1, 2, 2, 2, 5, 8, 13], 2),
    ]

    print("Fibonacci Search MVP demo")
    print("-" * 34)

    for idx, (arr, target) in enumerate(cases, start=1):
        result = fibonacci_search(arr, target)

        # For duplicates, any matching index is accepted by fibonacci_search.
        if result != -1 and arr[result] != target:
            raise AssertionError(
                f"Case {idx}: returned index {result}, but arr[result]={arr[result]} != target={target}"
            )

        expected_first = _expected_via_bisect(arr, target)
        found_flag = int(result != -1)
        expected_flag = int(expected_first != -1)
        if found_flag != expected_flag:
            raise AssertionError(
                f"Case {idx}: found_flag={found_flag}, expected_flag={expected_flag}, arr={arr}, target={target}"
            )

        print(
            f"Case {idx:02d}: arr={arr}, target={target}, index={result}, "
            f"exists={bool(found_flag)}, bisect_first={expected_first}"
        )

    try:
        fibonacci_search([3, 1, 2], 1)
        raise AssertionError("Unsorted-input check failed: ValueError was expected")
    except ValueError:
        print("Unsorted input check: passed (ValueError raised as expected)")

    print("All checks passed.")


if __name__ == "__main__":
    main()

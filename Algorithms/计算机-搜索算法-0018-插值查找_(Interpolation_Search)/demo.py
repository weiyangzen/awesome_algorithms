"""Interpolation Search MVP demo.

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


def interpolation_search(arr: Sequence[int], target: int) -> int:
    """Return an index of target in sorted arr, or -1 if absent."""
    _ensure_non_decreasing(arr)

    n = len(arr)
    low, high = 0, n - 1

    while low <= high and n > 0 and arr[low] <= target <= arr[high]:
        # Prevent division by zero when all values in current interval are equal.
        if arr[low] == arr[high]:
            return low if arr[low] == target else -1

        pos = low + ((target - arr[low]) * (high - low)) // (arr[high] - arr[low])
        pos_value = arr[pos]

        if pos_value == target:
            return pos
        if pos_value < target:
            low = pos + 1
        else:
            high = pos - 1

    return -1


def _expected_via_bisect(arr: Sequence[int], target: int) -> int:
    """Reference answer via bisect_left; returns first match or -1."""
    pos = bisect_left(arr, target)
    if pos < len(arr) and arr[pos] == target:
        return pos
    return -1


def main() -> None:
    cases = [
        ([], 10),
        ([5], 5),
        ([5], 7),
        ([10, 20, 30, 40, 50, 60, 70], 10),
        ([10, 20, 30, 40, 50, 60, 70], 50),
        ([10, 20, 30, 40, 50, 60, 70], 70),
        ([10, 20, 30, 40, 50, 60, 70], 35),
        ([1, 2, 4, 8, 16, 32, 64], 8),
        ([1, 2, 4, 8, 16, 32, 64], 3),
        ([-20, -10, -5, 0, 7, 13, 21], -5),
        ([-20, -10, -5, 0, 7, 13, 21], 6),
        ([2, 2, 2, 2, 2], 2),
        ([2, 2, 2, 2, 2], 3),
        ([1, 2, 2, 2, 5, 8, 13], 2),
    ]

    print("Interpolation Search MVP demo")
    print("-" * 38)

    for idx, (arr, target) in enumerate(cases, start=1):
        result = interpolation_search(arr, target)

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
        interpolation_search([3, 1, 2], 1)
        raise AssertionError("Unsorted-input check failed: ValueError was expected")
    except ValueError:
        print("Unsorted input check: passed (ValueError raised as expected)")

    print("All checks passed.")


if __name__ == "__main__":
    main()

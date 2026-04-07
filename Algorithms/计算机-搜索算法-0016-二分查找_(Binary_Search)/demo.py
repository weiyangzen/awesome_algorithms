"""Binary Search MVP demo.

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


def binary_search(arr: Sequence[int], target: int) -> int:
    """Return an index of target in sorted arr, or -1 if absent."""
    _ensure_non_decreasing(arr)

    left, right = 0, len(arr) - 1
    while left <= right:
        mid = left + (right - left) // 2
        mid_value = arr[mid]

        if mid_value == target:
            return mid
        if mid_value < target:
            left = mid + 1
        else:
            right = mid - 1

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
        ([1, 3, 5, 7, 9], 7),
        ([1, 3, 5, 7, 9], 10),
        ([1, 2, 2, 2, 5, 8], 2),
        ([-10, -3, 0, 4, 9, 15], -3),
        ([-10, -3, 0, 4, 9, 15], 6),
    ]

    print("Binary Search MVP demo")
    print("-" * 32)

    for idx, (arr, target) in enumerate(cases, start=1):
        result = binary_search(arr, target)

        # For duplicates, any matching index is accepted by this binary_search.
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

    # Explicitly verify unsorted input handling.
    try:
        binary_search([3, 1, 2], 1)
        raise AssertionError("Unsorted-input check failed: ValueError was expected")
    except ValueError:
        print("Unsorted input check: passed (ValueError raised as expected)")

    print("All checks passed.")


if __name__ == "__main__":
    main()

"""Jump Search MVP demo.

Run:
    uv run python demo.py
"""

from __future__ import annotations

from bisect import bisect_left
from math import isqrt
from typing import Sequence


def _ensure_non_decreasing(arr: Sequence[int]) -> None:
    """Raise ValueError if arr is not sorted in non-decreasing order."""
    for i in range(1, len(arr)):
        if arr[i] < arr[i - 1]:
            raise ValueError(
                f"Input must be sorted in non-decreasing order: arr[{i - 1}]={arr[i - 1]} > arr[{i}]={arr[i]}"
            )


def jump_search(arr: Sequence[int], target: int) -> int:
    """Return an index of target in sorted arr, or -1 if absent."""
    _ensure_non_decreasing(arr)

    n = len(arr)
    if n == 0:
        return -1

    step = max(1, isqrt(n))
    prev = 0

    # Phase 1: jump block by block until the block end is >= target.
    while prev < n:
        block_end = min(prev + step, n) - 1
        if arr[block_end] < target:
            prev += step
            continue
        break

    if prev >= n:
        return -1

    # Phase 2: linear scan inside the candidate block.
    upper = min(prev + step, n)
    for i in range(prev, upper):
        value = arr[i]
        if value == target:
            return i
        if value > target:
            return -1

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
        ([1, 3, 5, 7, 9, 11, 13], 1),
        ([1, 3, 5, 7, 9, 11, 13], 13),
        ([1, 3, 5, 7, 9, 11, 13], 8),
        ([2, 4, 6, 8, 10, 12, 14, 16, 18], 12),
        ([2, 4, 6, 8, 10, 12, 14, 16, 18], 7),
        ([-10, -5, -3, 0, 2, 8, 12], -3),
        ([-10, -5, -3, 0, 2, 8, 12], 6),
        ([1, 2, 2, 2, 5, 8, 13], 2),
    ]

    print("Jump Search MVP demo")
    print("-" * 30)

    for idx, (arr, target) in enumerate(cases, start=1):
        result = jump_search(arr, target)

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
        jump_search([3, 1, 2], 1)
        raise AssertionError("Unsorted-input check failed: ValueError was expected")
    except ValueError:
        print("Unsorted input check: passed (ValueError raised as expected)")

    print("All checks passed.")


if __name__ == "__main__":
    main()

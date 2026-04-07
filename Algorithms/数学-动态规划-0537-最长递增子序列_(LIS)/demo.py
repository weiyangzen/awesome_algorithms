"""LIS MVP: O(n log n) patience sorting + O(n^2) DP cross-check."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np


@dataclass
class LISResult:
    length: int
    indices: List[int]
    subsequence: List[float]


def to_1d_finite_array(values: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"Input must be a 1D sequence, got shape={arr.shape}.")
    if arr.size > 0 and not np.all(np.isfinite(arr)):
        raise ValueError("Input contains non-finite values (nan or inf).")
    return arr


def lis_patience_strict(values: Sequence[float] | np.ndarray) -> LISResult:
    """Return one strict LIS using patience sorting + predecessor reconstruction."""
    arr = to_1d_finite_array(values)
    n = int(arr.size)
    if n == 0:
        return LISResult(length=0, indices=[], subsequence=[])

    tails_values: List[float] = []
    tails_indices: List[int] = []
    prev = [-1] * n

    for i in range(n):
        x = float(arr[i])
        pos = bisect_left(tails_values, x)

        if pos == len(tails_values):
            tails_values.append(x)
            tails_indices.append(i)
        else:
            tails_values[pos] = x
            tails_indices[pos] = i

        if pos > 0:
            prev[i] = tails_indices[pos - 1]

    lis_indices_reversed: List[int] = []
    cur = tails_indices[-1]
    while cur != -1:
        lis_indices_reversed.append(cur)
        cur = prev[cur]

    indices = list(reversed(lis_indices_reversed))
    subsequence = [float(arr[idx]) for idx in indices]
    return LISResult(length=len(indices), indices=indices, subsequence=subsequence)


def lis_dp_quadratic(values: Sequence[float] | np.ndarray) -> LISResult:
    """Classic O(n^2) DP version, used here as a correctness baseline."""
    arr = to_1d_finite_array(values)
    n = int(arr.size)
    if n == 0:
        return LISResult(length=0, indices=[], subsequence=[])

    dp = [1] * n
    parent = [-1] * n

    for i in range(n):
        for j in range(i):
            if arr[j] < arr[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                parent[i] = j

    end_idx = max(range(n), key=lambda idx: dp[idx])

    idx_rev: List[int] = []
    cur = end_idx
    while cur != -1:
        idx_rev.append(cur)
        cur = parent[cur]

    indices = list(reversed(idx_rev))
    subsequence = [float(arr[idx]) for idx in indices]
    return LISResult(length=len(indices), indices=indices, subsequence=subsequence)


def is_strictly_increasing(seq: Iterable[float]) -> bool:
    items = list(seq)
    return all(items[i] < items[i + 1] for i in range(len(items) - 1))


def run_case(name: str, values: Sequence[float]) -> None:
    fast = lis_patience_strict(values)
    slow = lis_dp_quadratic(values)

    length_equal = fast.length == slow.length
    fast_is_strict = is_strictly_increasing(fast.subsequence)

    print(f"=== {name} ===")
    print(f"Input: {list(values)}")
    print(
        f"Fast O(n log n): length={fast.length}, "
        f"indices={fast.indices}, subseq={fast.subsequence}"
    )
    print(
        f"DP   O(n^2):    length={slow.length}, "
        f"indices={slow.indices}, subseq={slow.subsequence}"
    )
    print(f"Checks: length_equal={length_equal}, fast_is_strict={fast_is_strict}\n")

    if not length_equal:
        raise AssertionError(f"Length mismatch in case '{name}': {fast.length} != {slow.length}")
    if not fast_is_strict:
        raise AssertionError(f"Fast LIS is not strictly increasing in case '{name}'.")


def main() -> None:
    cases = {
        "Case 1: classic": [10, 9, 2, 5, 3, 7, 101, 18],
        "Case 2: with duplicates": [0, 1, 0, 3, 2, 3],
        "Case 3: descending": [5, 4, 3, 2, 1],
        "Case 4: all equal": [7, 7, 7, 7],
        "Case 5: empty": [],
        "Case 6: mixed signs": [-1, 3, 4, 5, 2, 2, 2, 2],
    }
    for name, values in cases.items():
        run_case(name, values)


if __name__ == "__main__":
    main()

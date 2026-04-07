"""Minimal runnable MVP for QuickSelect (k-th order statistic)."""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Sequence

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None


@dataclass
class QuickSelectStats:
    """Lightweight runtime counters for demonstration and sanity checks."""

    partition_rounds: int = 0
    comparisons: int = 0
    swaps: int = 0
    pivots: list[int] = field(default_factory=list)


def _validate_input(values: Sequence[int], k: int) -> None:
    if len(values) == 0:
        raise ValueError("values must be non-empty")
    if not (0 <= k < len(values)):
        raise IndexError(f"k out of range: k={k}, n={len(values)}")


def _partition_three_way(
    arr: list[int],
    left: int,
    right: int,
    pivot_value: int,
    stats: QuickSelectStats,
) -> tuple[int, int]:
    """Dutch National Flag partition on arr[left:right+1].

    Returns:
        (lt, gt):
        - arr[left:lt] < pivot_value
        - arr[lt:gt+1] == pivot_value
        - arr[gt+1:right+1] > pivot_value
    """
    lt = left
    i = left
    gt = right

    while i <= gt:
        stats.comparisons += 1
        if arr[i] < pivot_value:
            if i != lt:
                arr[i], arr[lt] = arr[lt], arr[i]
                stats.swaps += 1
            i += 1
            lt += 1
            continue

        stats.comparisons += 1
        if arr[i] > pivot_value:
            if i != gt:
                arr[i], arr[gt] = arr[gt], arr[i]
                stats.swaps += 1
            gt -= 1
            continue

        i += 1

    return lt, gt


def quickselect(values: Sequence[int], k: int, *, seed: int = 0) -> tuple[int, QuickSelectStats]:
    """Return the k-th smallest element (0-based rank) using QuickSelect.

    This implementation is iterative and uses random pivots plus three-way partitioning
    to handle duplicate values robustly.
    """
    _validate_input(values, k)

    arr = list(values)
    stats = QuickSelectStats()
    rng = random.Random(seed)

    left = 0
    right = len(arr) - 1

    while True:
        if left == right:
            return arr[left], stats

        pivot_idx = rng.randint(left, right)
        pivot_value = arr[pivot_idx]

        stats.partition_rounds += 1
        stats.pivots.append(pivot_value)

        lt, gt = _partition_three_way(arr, left, right, pivot_value, stats)

        if k < lt:
            right = lt - 1
        elif k > gt:
            left = gt + 1
        else:
            return arr[k], stats


def _numpy_reference(values: Sequence[int], k: int) -> int | None:
    """Optional external cross-check using numpy.partition."""
    if np is None:
        return None
    arr = np.asarray(values)
    return int(np.partition(arr, k)[k])


def _run_case(name: str, values: Sequence[int], k: int) -> None:
    expected = sorted(values)[k]
    got, stats = quickselect(values, k, seed=2026)
    assert got == expected, f"{name}: got={got}, expected={expected}"

    np_ref = _numpy_reference(values, k)
    if np_ref is not None:
        assert np_ref == expected, f"{name}: numpy mismatch {np_ref} != {expected}"

    print(f"[{name}]")
    print(f"  n={len(values)}, k={k}")
    print(f"  quickselect={got}, sorted_ref={expected}")
    if np_ref is None:
        print("  numpy_ref=SKIPPED (numpy not installed)")
    else:
        print(f"  numpy_ref={np_ref}")
    print(
        "  rounds={rounds}, comparisons={comparisons}, swaps={swaps}, pivot_preview={pivots}".format(
            rounds=stats.partition_rounds,
            comparisons=stats.comparisons,
            swaps=stats.swaps,
            pivots=stats.pivots[:8],
        )
    )


def _random_regression() -> None:
    rng = random.Random(7)
    for i in range(30):
        n = rng.randint(10, 300)
        values = [rng.randint(-5000, 5000) for _ in range(n)]
        k = rng.randrange(n)
        got, _ = quickselect(values, k, seed=1000 + i)
        expected = sorted(values)[k]
        assert got == expected, f"random-{i} failed: got={got}, expected={expected}"


def main() -> None:
    _run_case(
        "case-1-small",
        [12, 3, 5, 7, 4, 19, 26],
        3,
    )
    _run_case(
        "case-2-duplicates",
        [9, 1, 5, 3, 5, 5, 2, 8, 5, 0, 3, 3],
        6,
    )
    _run_case(
        "case-3-mixed",
        [42, -4, 17, 17, 99, 0, -12, 8, 8, 73, -1, 21],
        5,
    )

    _random_regression()
    print("All QuickSelect checks passed.")


if __name__ == "__main__":
    main()

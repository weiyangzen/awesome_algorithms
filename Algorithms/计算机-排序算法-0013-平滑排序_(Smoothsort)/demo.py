"""Smoothsort MVP demo.

This script implements Dijkstra's Smoothsort with Leonardo heaps,
then validates correctness on deterministic test cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, MutableSequence, Sequence, Tuple, TypeVar

import numpy as np

T = TypeVar("T")

# Leonardo numbers: LP[k] = Leonardo(k), enough for practical list sizes.
LP: List[int] = [
    1,
    1,
    3,
    5,
    9,
    15,
    25,
    41,
    67,
    109,
    177,
    287,
    465,
    753,
    1219,
    1973,
    3193,
    5167,
    8361,
    13529,
    21891,
    35421,
    57313,
    92735,
    150049,
    242785,
    392835,
    635621,
    1028457,
    1664079,
    2692537,
    4356617,
    7049155,
    11405773,
    18454929,
    29860703,
    48315633,
    78176337,
    126491971,
    204668309,
    331160281,
    535828591,
    866988873,
]


@dataclass
class SmoothsortStats:
    comparisons: int = 0
    sifts: int = 0
    trinkles: int = 0


@dataclass
class CaseResult:
    name: str
    n: int
    comparisons: int
    sifts: int
    trinkles: int


def _ctz(x: int) -> int:
    """Count trailing zero bits. Return 0 when x == 0."""
    if x == 0:
        return 0
    return (x & -x).bit_length() - 1


def _le(a: T, b: T, stats: SmoothsortStats) -> bool:
    stats.comparisons += 1
    return a <= b


def _ge(a: T, b: T, stats: SmoothsortStats) -> bool:
    stats.comparisons += 1
    return a >= b


def _sift(
    arr: MutableSequence[T],
    pshift: int,
    head: int,
    stats: SmoothsortStats,
) -> None:
    """Fix heap property inside one Leonardo tree."""
    stats.sifts += 1

    while pshift > 1:
        rt = head - 1
        lf = head - 1 - LP[pshift - 2]

        if _ge(arr[head], arr[lf], stats) and _ge(arr[head], arr[rt], stats):
            break

        if _ge(arr[lf], arr[rt], stats):
            arr[head], arr[lf] = arr[lf], arr[head]
            head = lf
            pshift -= 1
        else:
            arr[head], arr[rt] = arr[rt], arr[head]
            head = rt
            pshift -= 2


def _trinkle(
    arr: MutableSequence[T],
    p: int,
    pshift: int,
    head: int,
    trusty: bool,
    stats: SmoothsortStats,
) -> None:
    """Fix heap property across Leonardo heap forest."""
    stats.trinkles += 1

    while p != 1:
        stepson = head - LP[pshift]
        if _le(arr[stepson], arr[head], stats):
            break

        if not trusty and pshift > 1:
            rt = head - 1
            lf = head - 1 - LP[pshift - 2]
            if _ge(arr[rt], arr[stepson], stats) or _ge(arr[lf], arr[stepson], stats):
                break

        arr[head], arr[stepson] = arr[stepson], arr[head]
        head = stepson

        trail = _ctz(p & ~1)
        p >>= trail
        pshift += trail
        trusty = False

    if not trusty:
        _sift(arr, pshift, head, stats)


def smoothsort_inplace(arr: MutableSequence[T], stats: SmoothsortStats | None = None) -> SmoothsortStats:
    """Sort arr in-place using Smoothsort."""
    if stats is None:
        stats = SmoothsortStats()

    n = len(arr)
    if n < 2:
        return stats

    if n > LP[-1]:
        raise ValueError(f"Input too large for current Leonardo table: n={n}, max={LP[-1]}.")

    p = 1
    pshift = 1
    head = 0

    # Heap construction phase.
    while head < n - 1:
        if (p & 3) == 3:
            _sift(arr, pshift, head, stats)
            p >>= 2
            pshift += 2
        else:
            if LP[pshift - 1] >= n - 1 - head:
                _trinkle(arr, p, pshift, head, False, stats)
            else:
                _sift(arr, pshift, head, stats)

            if pshift == 1:
                p <<= 1
                pshift -= 1
            else:
                p <<= pshift - 1
                pshift = 1

        p |= 1
        head += 1

    _trinkle(arr, p, pshift, head, False, stats)

    # Heap extraction phase.
    while pshift != 1 or p != 1:
        if pshift <= 1:
            trail = _ctz(p & ~1)
            p >>= trail
            pshift += trail
        else:
            p <<= 2
            p ^= 7
            pshift -= 2

            _trinkle(arr, p >> 1, pshift + 1, head - LP[pshift] - 1, True, stats)
            _trinkle(arr, p, pshift, head - 1, True, stats)

        head -= 1

    return stats


def smoothsort(values: Sequence[T]) -> Tuple[List[T], SmoothsortStats]:
    arr = list(values)
    stats = smoothsort_inplace(arr)
    return arr, stats


def _make_nearly_sorted(n: int, swaps: int, seed: int) -> List[int]:
    arr = list(range(n))
    rng = np.random.default_rng(seed)
    for _ in range(swaps):
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        arr[i], arr[j] = arr[j], arr[i]
    return arr


def _run_case(name: str, data: Sequence[int]) -> CaseResult:
    sorted_values, stats = smoothsort(data)
    expected = sorted(list(data))

    if sorted_values != expected:
        raise RuntimeError(f"{name}: smoothsort result mismatch.")

    print(
        f"{name:<18} n={len(data):3d} | comparisons={stats.comparisons:5d} | "
        f"sifts={stats.sifts:3d} | trinkles={stats.trinkles:3d}"
    )

    return CaseResult(
        name=name,
        n=len(data),
        comparisons=stats.comparisons,
        sifts=stats.sifts,
        trinkles=stats.trinkles,
    )


def main() -> None:
    n = 64
    rng = np.random.default_rng(20260407)

    sorted_case = list(range(n))
    reverse_case = list(range(n - 1, -1, -1))
    nearly_case = _make_nearly_sorted(n=n, swaps=4, seed=7)
    random_case = rng.integers(-10_000, 10_000, size=n).tolist()
    duplicate_case = rng.choice([0, 1, 2, 3, 4], size=n, p=[0.3, 0.25, 0.2, 0.15, 0.1]).tolist()

    print("Smoothsort deterministic validation")
    print("-" * 78)
    results = [
        _run_case("already_sorted", sorted_case),
        _run_case("reverse_sorted", reverse_case),
        _run_case("nearly_sorted", nearly_case),
        _run_case("random", random_case),
        _run_case("many_duplicates", duplicate_case),
    ]

    # NumPy cross-check: verify identical order as numpy.sort for a fixed array.
    np_case = np.array([9, -3, 7, 7, 2, 0, -8, 4, 4, 1], dtype=int)
    np_sorted = np.sort(np_case).tolist()
    our_sorted, np_stats = smoothsort(np_case.tolist())
    if our_sorted != np_sorted:
        raise RuntimeError("numpy cross-check failed.")

    print("-" * 78)
    print(
        "numpy_cross_check    n= 10 | comparisons="
        f"{np_stats.comparisons:5d} | sifts={np_stats.sifts:3d} | trinkles={np_stats.trinkles:3d}"
    )

    by_name = {r.name: r for r in results}
    print("-" * 78)
    print(
        "adaptiveness_hint: comparisons(already_sorted) <= comparisons(random) is",
        by_name["already_sorted"].comparisons <= by_name["random"].comparisons,
    )
    print("All validation checks passed.")


if __name__ == "__main__":
    main()

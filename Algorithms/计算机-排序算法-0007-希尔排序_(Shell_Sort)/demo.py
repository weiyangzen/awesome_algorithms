"""Shell Sort MVP.

A small, auditable shell-sort implementation with deterministic tests.
No interactive input is required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Sequence, Tuple, TypeVar

import numpy as np

T = TypeVar("T")


@dataclass
class GapPassStat:
    gap: int
    comparisons: int = 0
    shifts: int = 0
    writes: int = 0


@dataclass
class ShellSortStats:
    strategy: str
    gaps: List[int] = field(default_factory=list)
    comparisons: int = 0
    shifts: int = 0
    writes: int = 0
    passes: List[GapPassStat] = field(default_factory=list)


def build_gaps(n: int, strategy: str = "halving") -> List[int]:
    """Build a descending gap sequence ending with 1."""
    if n <= 1:
        return []

    if strategy == "halving":
        gaps: List[int] = []
        gap = n // 2
        while gap > 0:
            gaps.append(gap)
            gap //= 2
        return gaps

    if strategy == "knuth":
        gaps = []
        h = 1
        while h < n:
            gaps.append(h)
            h = 3 * h + 1
        gaps.reverse()
        return gaps

    raise ValueError(f"Unknown gap strategy: {strategy}")


def gapped_insertion_pass(
    arr: List[T],
    gap: int,
    key: Callable[[T], object],
) -> GapPassStat:
    """Run one insertion-sort pass with a fixed gap."""
    stat = GapPassStat(gap=gap)

    for i in range(gap, len(arr)):
        current = arr[i]
        current_key = key(current)
        j = i

        while j >= gap:
            stat.comparisons += 1
            if key(arr[j - gap]) > current_key:
                arr[j] = arr[j - gap]
                stat.shifts += 1
                j -= gap
            else:
                break

        arr[j] = current
        stat.writes += 1

    return stat


def shell_sort_in_place(
    arr: List[T],
    *,
    strategy: str = "halving",
    key: Callable[[T], object] = lambda x: x,
) -> ShellSortStats:
    """Sort `arr` in place and return shell-sort statistics."""
    stats = ShellSortStats(strategy=strategy)
    gaps = build_gaps(len(arr), strategy=strategy)
    stats.gaps = gaps.copy()

    for gap in gaps:
        pass_stat = gapped_insertion_pass(arr, gap, key=key)
        stats.passes.append(pass_stat)
        stats.comparisons += pass_stat.comparisons
        stats.shifts += pass_stat.shifts
        stats.writes += pass_stat.writes

    return stats


def shell_sort(
    data: Sequence[T],
    *,
    strategy: str = "halving",
    key: Callable[[T], object] = lambda x: x,
) -> Tuple[List[T], ShellSortStats]:
    """Return a sorted copy and statistics."""
    out = list(data)
    stats = shell_sort_in_place(out, strategy=strategy, key=key)
    return out, stats


def is_stable_on_sorted_records(sorted_records: Sequence[Tuple[int, int]]) -> bool:
    """For equal values, original index order should stay non-decreasing."""
    groups: dict[int, List[int]] = {}
    for value, original_index in sorted_records:
        groups.setdefault(value, []).append(original_index)
    return all(
        all(indices[i] <= indices[i + 1] for i in range(len(indices) - 1))
        for indices in groups.values()
    )


def run_case(case_name: str, data: Sequence[int], strategy: str) -> bool:
    result, stats = shell_sort(data, strategy=strategy)
    expected = sorted(data)
    ok = result == expected

    pass_text = ", ".join(
        f"gap={p.gap}:cmp={p.comparisons},shift={p.shifts},write={p.writes}" for p in stats.passes
    )

    print(f"\n[{case_name}]")
    print(f"input         : {list(data)}")
    print(f"shell_sort    : {result}")
    print(f"python_sorted : {expected}")
    print(f"match         : {ok}")
    print(f"strategy      : {stats.strategy}")
    print(f"gaps          : {stats.gaps}")
    print(f"comparisons   : {stats.comparisons}")
    print(f"shifts        : {stats.shifts}")
    print(f"writes        : {stats.writes}")
    print(f"pass_stats    : {pass_text if pass_text else '(no pass)'}")

    return ok


def demonstrate_not_stable() -> bool:
    """Show shell sort is not stable on one crafted record case."""
    # Same keys(2) have original order indices 0,1,3.
    # A shell pass with gap=2 can move index 0 behind index 1.
    records = [(2, 0), (2, 1), (1, 2), (2, 3)]
    sorted_records, stats = shell_sort(records, strategy="halving", key=lambda x: x[0])
    stable = is_stable_on_sorted_records(sorted_records)

    print("\n[stability_demonstration]")
    print(f"records_input : {records}")
    print(f"records_sorted: {sorted_records}")
    print(f"strategy      : {stats.strategy}")
    print(f"gaps          : {stats.gaps}")
    print(f"stable        : {stable}")

    return stable


def main() -> None:
    rng = np.random.default_rng(2026)

    fixed_case = [23, 12, 1, 8, 34, 54, 2, 3]
    duplicate_case = [5, -1, 5, 3, 3, 0, -1, 5]
    empty_case: List[int] = []
    single_case = [7]
    random_case = rng.integers(low=-30, high=31, size=18).tolist()

    checks = [
        run_case("fixed_halving", fixed_case, strategy="halving"),
        run_case("duplicates_knuth", duplicate_case, strategy="knuth"),
        run_case("empty_halving", empty_case, strategy="halving"),
        run_case("single_knuth", single_case, strategy="knuth"),
        run_case("random_halving", random_case, strategy="halving"),
    ]

    stable = demonstrate_not_stable()
    unstable_demo_ok = not stable

    all_ok = all(checks) and unstable_demo_ok
    print("\n=== Summary ===")
    print(f"all_cases_passed={all_ok}")
    print(f"sorted_cases={len(checks)}")
    print(f"stability_demo_is_unstable={unstable_demo_ok}")


if __name__ == "__main__":
    main()

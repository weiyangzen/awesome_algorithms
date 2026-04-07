"""Greedy MVP for CS-0061: 活动选择问题.

Run:
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class Activity:
    """One activity interval represented as [start, finish)."""

    name: str
    start: int
    finish: int


@dataclass(frozen=True)
class FixedCase:
    """Deterministic regression case."""

    name: str
    activities: list[Activity]
    expected_count: int


def _normalize_activities(activities: Iterable[Activity]) -> list[Activity]:
    """Validate activities and normalize fields to int/str."""
    normalized: list[Activity] = []
    for idx, activity in enumerate(activities):
        start = int(activity.start)
        finish = int(activity.finish)
        if start > finish:
            raise ValueError(
                f"activity[{idx}] has start > finish: name={activity.name!r}, "
                f"start={start}, finish={finish}"
            )
        normalized.append(Activity(name=str(activity.name), start=start, finish=finish))
    return normalized


def _compatible(prev: Activity, curr: Activity) -> bool:
    """Compatibility for half-open intervals [start, finish)."""
    return prev.finish <= curr.start


def activity_selection_greedy(activities: Iterable[Activity]) -> list[Activity]:
    """Select a maximum-size compatible subset via earliest-finish-time greedy."""
    acts = _normalize_activities(activities)
    ordered = sorted(acts, key=lambda x: (x.finish, x.start, x.name))

    selected: list[Activity] = []
    for activity in ordered:
        if not selected or _compatible(selected[-1], activity):
            selected.append(activity)

    return selected


def _is_valid_schedule(schedule: Sequence[Activity]) -> bool:
    """Return whether a schedule contains no overlap under [start, finish) semantics."""
    ordered = sorted(schedule, key=lambda x: (x.start, x.finish, x.name))
    for i in range(1, len(ordered)):
        if ordered[i - 1].finish > ordered[i].start:
            return False
    return True


def exact_maximum_schedule_bruteforce(activities: Iterable[Activity]) -> list[Activity]:
    """Exact solver for small n via subset enumeration.

    This is used only as a correctness oracle for the greedy implementation.
    """
    acts = _normalize_activities(activities)
    n = len(acts)

    best: list[Activity] = []
    best_key: tuple[str, ...] = ()

    for r in range(n + 1):
        for idxs in combinations(range(n), r):
            candidate = [acts[i] for i in idxs]
            if not _is_valid_schedule(candidate):
                continue

            candidate_sorted = sorted(candidate, key=lambda x: (x.finish, x.start, x.name))
            candidate_key = tuple(a.name for a in candidate_sorted)

            if len(candidate_sorted) > len(best):
                best = candidate_sorted
                best_key = candidate_key
            elif len(candidate_sorted) == len(best) and candidate_key < best_key:
                best = candidate_sorted
                best_key = candidate_key

    return best


def activities_from_numpy(interval_array: np.ndarray, name_prefix: str = "np") -> list[Activity]:
    """Bridge: build activities from an (n, 2) numpy array [start, finish]."""
    arr = np.asarray(interval_array, dtype=int)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected shape (n, 2), got {arr.shape}")

    activities: list[Activity] = []
    for i, (start, finish) in enumerate(arr.tolist()):
        activities.append(Activity(name=f"{name_prefix}_{i}", start=int(start), finish=int(finish)))
    return activities


def assert_optimality(activities: Sequence[Activity]) -> None:
    """Assert greedy cardinality equals exact cardinality."""
    greedy = activity_selection_greedy(activities)
    exact = exact_maximum_schedule_bruteforce(activities)

    assert _is_valid_schedule(greedy), f"Greedy returned invalid schedule: {greedy}"
    assert _is_valid_schedule(exact), f"Exact returned invalid schedule: {exact}"
    assert len(greedy) == len(exact), (
        "Greedy is not optimal for this case: "
        f"greedy_count={len(greedy)}, exact_count={len(exact)}"
    )


def _format_schedule(schedule: Sequence[Activity]) -> str:
    if not schedule:
        return "[]"
    parts = [f"{a.name}[{a.start},{a.finish})" for a in schedule]
    return "[" + ", ".join(parts) + "]"


def run_fixed_cases() -> None:
    cases = [
        FixedCase(
            name="CLRS classic",
            activities=[
                Activity("a1", 1, 4),
                Activity("a2", 3, 5),
                Activity("a3", 0, 6),
                Activity("a4", 5, 7),
                Activity("a5", 3, 8),
                Activity("a6", 5, 9),
                Activity("a7", 6, 10),
                Activity("a8", 8, 11),
                Activity("a9", 8, 12),
                Activity("a10", 2, 13),
                Activity("a11", 12, 14),
            ],
            expected_count=4,
        ),
        FixedCase(
            name="all compatible chain",
            activities=[
                Activity("b1", 0, 1),
                Activity("b2", 1, 2),
                Activity("b3", 2, 3),
                Activity("b4", 3, 4),
                Activity("b5", 4, 6),
            ],
            expected_count=5,
        ),
        FixedCase(
            name="heavy overlap",
            activities=[
                Activity("c1", 0, 10),
                Activity("c2", 1, 9),
                Activity("c3", 2, 8),
                Activity("c4", 3, 7),
                Activity("c5", 4, 6),
            ],
            expected_count=1,
        ),
        FixedCase(
            name="boundary-touch mix",
            activities=[
                Activity("d1", 1, 3),
                Activity("d2", 3, 3),
                Activity("d3", 3, 5),
                Activity("d4", 5, 8),
                Activity("d5", 0, 2),
                Activity("d6", 8, 9),
            ],
            expected_count=5,
        ),
    ]

    print("=== Fixed Cases ===")
    for idx, case in enumerate(cases, start=1):
        greedy = activity_selection_greedy(case.activities)
        exact = exact_maximum_schedule_bruteforce(case.activities)

        assert len(greedy) == case.expected_count, (
            f"Case {case.name} failed: expected={case.expected_count}, got={len(greedy)}"
        )
        assert len(exact) == case.expected_count, (
            f"Case {case.name} oracle mismatch: expected={case.expected_count}, got={len(exact)}"
        )

        print(
            f"[{idx}] {case.name}: count={len(greedy)}, "
            f"greedy={_format_schedule(greedy)}"
        )


def run_random_verification(trials: int = 120) -> None:
    """Random small-case regression: greedy count must equal exact count."""
    rng = np.random.default_rng(610061)

    for t in range(trials):
        n = int(rng.integers(1, 11))
        starts = rng.integers(0, 25, size=n)
        durations = rng.integers(0, 8, size=n)
        finishes = starts + durations

        activities = [
            Activity(name=f"r{t}_{i}", start=int(starts[i]), finish=int(finishes[i]))
            for i in range(n)
        ]
        assert_optimality(activities)

    print(f"\nRandom verification passed: {trials} cases")


def run_numpy_case() -> None:
    """Demonstrate numpy input bridge with deterministic sample."""
    interval_array = np.array(
        [
            [0, 2],
            [1, 4],
            [3, 5],
            [5, 7],
            [6, 9],
            [8, 9],
        ],
        dtype=int,
    )

    activities = activities_from_numpy(interval_array, name_prefix="N")
    greedy = activity_selection_greedy(activities)
    exact = exact_maximum_schedule_bruteforce(activities)

    assert len(greedy) == len(exact)

    print("\n=== Numpy Case ===")
    print(f"input array:\n{interval_array}")
    print(f"greedy count: {len(greedy)}")
    print(f"greedy schedule: {_format_schedule(greedy)}")


def main() -> None:
    run_fixed_cases()
    run_random_verification()
    run_numpy_case()
    print("\nAll checks passed for CS-0061 (活动选择问题).")


if __name__ == "__main__":
    main()

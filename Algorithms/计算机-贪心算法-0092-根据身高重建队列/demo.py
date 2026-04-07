"""CS-0072 根据身高重建队列：贪心算法最小可运行 MVP。

运行:
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable, Sequence

import numpy as np

Person = tuple[int, int]


@dataclass(frozen=True)
class FixedCase:
    """Deterministic test case for queue reconstruction."""

    name: str
    people: list[Person]
    expected: list[Person]


def _normalize_people(people: Iterable[Sequence[int]]) -> list[Person]:
    """Validate and normalize input to a list of (height, k) integer tuples."""
    normalized: list[Person] = []
    for idx, item in enumerate(people):
        if len(item) != 2:
            raise ValueError(f"Person at index {idx} must be a pair (h, k): {item!r}")

        h = int(item[0])
        k = int(item[1])
        if h < 0 or k < 0:
            raise ValueError(f"Person at index {idx} must satisfy h>=0 and k>=0: {(h, k)}")
        normalized.append((h, k))
    return normalized


def is_queue_valid_for_people(queue: Iterable[Sequence[int]], people: Iterable[Sequence[int]]) -> bool:
    """Check whether queue is a valid reconstruction for the given people multiset."""
    q = _normalize_people(queue)
    p = _normalize_people(people)

    if len(q) != len(p):
        return False
    if sorted(q) != sorted(p):
        return False

    for i, (h, k) in enumerate(q):
        ahead = sum(1 for prev_h, _ in q[:i] if prev_h >= h)
        if ahead != k:
            return False
    return True


def reconstruct_queue_greedy(people: Iterable[Sequence[int]]) -> list[Person]:
    """Greedy solution: sort by (-h, k), then insert each person at index k."""
    normalized = _normalize_people(people)
    ordered = sorted(normalized, key=lambda x: (-x[0], x[1]))

    queue: list[Person] = []
    for h, k in ordered:
        if k > len(queue):
            raise ValueError(
                f"Infeasible constraints: cannot insert {(h, k)} at index {k} "
                f"when queue length is {len(queue)}"
            )
        queue.insert(k, (h, k))

    if not is_queue_valid_for_people(queue, normalized):
        raise ValueError("Infeasible constraints: no valid queue satisfies all (h, k) pairs")

    return queue


def reconstruct_queue_bruteforce(people: Iterable[Sequence[int]], max_n: int = 8) -> list[Person]:
    """Small-scale backtracking baseline for cross-checking the greedy result."""
    normalized = _normalize_people(people)
    n = len(normalized)

    if n > max_n:
        raise ValueError(f"Bruteforce is limited to n <= {max_n}, got n={n}")

    used = [False] * n
    current: list[Person] = []
    answer: list[Person] | None = None

    def backtrack() -> bool:
        nonlocal answer

        if len(current) == n:
            answer = current.copy()
            return True

        for i, (h, k) in enumerate(normalized):
            if used[i]:
                continue

            # When constructing from front to back, the number of >=h before
            # the current position is already fixed and will not change later.
            ahead = sum(1 for prev_h, _ in current if prev_h >= h)
            if ahead != k:
                continue

            used[i] = True
            current.append((h, k))
            if backtrack():
                return True
            current.pop()
            used[i] = False

        return False

    backtrack()
    if answer is None:
        raise ValueError("No feasible queue exists for the provided constraints")

    return answer


def _people_from_height_queue(height_queue: Sequence[int]) -> list[Person]:
    """Generate (h, k) constraints from a concrete height queue."""
    people: list[Person] = []
    for i, height in enumerate(height_queue):
        h = int(height)
        k = sum(1 for prev_h in height_queue[:i] if int(prev_h) >= h)
        people.append((h, k))
    return people


def _run_fixed_cases() -> None:
    print("=== Fixed Cases ===")
    cases = [
        FixedCase(
            name="canonical example",
            people=[(7, 0), (4, 4), (7, 1), (5, 0), (6, 1), (5, 2)],
            expected=[(5, 0), (7, 0), (5, 2), (6, 1), (4, 4), (7, 1)],
        ),
        FixedCase(
            name="descending heights with varied k",
            people=[(6, 0), (5, 0), (4, 0), (3, 2), (2, 2), (1, 4)],
            expected=[(4, 0), (5, 0), (2, 2), (3, 2), (1, 4), (6, 0)],
        ),
        FixedCase(
            name="single person",
            people=[(9, 0)],
            expected=[(9, 0)],
        ),
    ]

    for i, case in enumerate(cases, start=1):
        got = reconstruct_queue_greedy(case.people)
        assert got == case.expected, (
            f"Case '{case.name}' failed: expected={case.expected}, got={got}"
        )
        assert is_queue_valid_for_people(got, case.people)
        print(f"[{i}] {case.name}: {case.people} -> {got}")

    invalid = [(5, 2), (6, 0)]
    try:
        reconstruct_queue_greedy(invalid)
    except ValueError:
        print("[4] invalid constraints: correctly rejected")
    else:
        raise AssertionError("Expected invalid input to be rejected")


def _run_random_regression(seed: int = 72, rounds: int = 8) -> None:
    """Generate feasible instances from random queues, then reconstruct from shuffled constraints."""
    print("\n=== Random Regression ===")
    rng = np.random.default_rng(seed)

    total = 0
    for n in [2, 3, 4, 5, 6, 7, 8, 12, 20]:
        for _ in range(rounds):
            # Use unique heights so that the reconstructed queue is deterministic
            # and easy to compare to the original queue.
            heights = rng.choice(np.arange(100, 4000), size=n, replace=False)
            target_heights = rng.permutation(heights)
            target_queue = _people_from_height_queue(target_heights.tolist())

            shuffled_people = target_queue.copy()
            rng.shuffle(shuffled_people)

            greedy_queue = reconstruct_queue_greedy(shuffled_people)
            assert greedy_queue == target_queue, (
                f"Greedy mismatch for n={n}: expected={target_queue}, got={greedy_queue}"
            )
            assert is_queue_valid_for_people(greedy_queue, shuffled_people)

            if n <= 8:
                brute_queue = reconstruct_queue_bruteforce(shuffled_people)
                assert brute_queue == target_queue, (
                    f"Bruteforce mismatch for n={n}: expected={target_queue}, got={brute_queue}"
                )

            total += 1

    print(f"random feasible cases={total}, seed={seed}: passed")


def _run_perf_snapshot(seed: int = 2026, n: int = 2000) -> None:
    """Simple performance snapshot for the greedy construction."""
    print("\n=== Performance Snapshot ===")
    rng = np.random.default_rng(seed)

    heights = rng.choice(np.arange(100, 100 + 5 * n), size=n, replace=False)
    target_heights = rng.permutation(heights)
    people = _people_from_height_queue(target_heights.tolist())
    rng.shuffle(people)

    t0 = perf_counter()
    queue = reconstruct_queue_greedy(people)
    t1 = perf_counter()

    assert is_queue_valid_for_people(queue, people)
    print(f"n={n}, greedy_time={t1 - t0:.6f}s")


def main() -> None:
    _run_fixed_cases()
    _run_random_regression()
    _run_perf_snapshot()
    print("\nAll checks passed for CS-0072 (根据身高重建队列).")


if __name__ == "__main__":
    main()

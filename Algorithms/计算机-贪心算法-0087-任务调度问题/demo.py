"""Job sequencing with deadlines and profits.

This MVP fixes the repository's "任务调度问题" entry to the classical
single-machine, unit-duration, profit-maximization variant.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import random
from typing import Iterable


@dataclass(frozen=True)
class Job:
    name: str
    deadline: int
    profit: int


def validate_jobs(jobs: Iterable[Job]) -> list[Job]:
    result = list(jobs)
    for job in result:
        if job.deadline < 1:
            raise ValueError(f"deadline must be >= 1: {job}")
        if job.profit < 0:
            raise ValueError(f"profit must be >= 0: {job}")
    return result


def find(parent: list[int], x: int) -> int:
    if parent[x] != x:
        parent[x] = find(parent, parent[x])
    return parent[x]


def schedule_jobs_greedy_dsu(jobs: Iterable[Job]) -> tuple[list[tuple[int, Job]], int]:
    items = validate_jobs(jobs)
    if not items:
        return [], 0

    max_slot = min(max(job.deadline for job in items), len(items))
    parent = list(range(max_slot + 1))
    slots: list[Job | None] = [None] * (max_slot + 1)

    ordered = sorted(items, key=lambda job: (-job.profit, job.deadline, job.name))
    for job in ordered:
        slot = find(parent, min(job.deadline, max_slot))
        if slot == 0:
            continue
        slots[slot] = job
        parent[slot] = find(parent, slot - 1)

    chosen = [(slot, job) for slot, job in enumerate(slots) if slot > 0 and job is not None]
    total_profit = sum(job.profit for _, job in chosen)
    return chosen, total_profit


def subset_is_schedulable(subset: tuple[Job, ...]) -> bool:
    ordered = sorted(subset, key=lambda job: (job.deadline, -job.profit, job.name))
    for idx, job in enumerate(ordered, start=1):
        if idx > job.deadline:
            return False
    return True


def exact_schedule_for_subset(subset: tuple[Job, ...]) -> list[tuple[int, Job]]:
    ordered = sorted(subset, key=lambda job: (job.deadline, -job.profit, job.name))
    return [(idx, job) for idx, job in enumerate(ordered, start=1)]


def schedule_jobs_exact_bruteforce(jobs: Iterable[Job]) -> tuple[list[tuple[int, Job]], int]:
    items = validate_jobs(jobs)
    best_profit = -1
    best_schedule: list[tuple[int, Job]] = []

    for r in range(len(items) + 1):
        for subset in combinations(items, r):
            if not subset_is_schedulable(subset):
                continue
            profit = sum(job.profit for job in subset)
            schedule = exact_schedule_for_subset(subset)
            chosen_names = tuple(job.name for _, job in schedule)
            best_names = tuple(job.name for _, job in best_schedule)
            if profit > best_profit or (profit == best_profit and chosen_names < best_names):
                best_profit = profit
                best_schedule = schedule

    if best_profit < 0:
        return [], 0
    return best_schedule, best_profit


def assert_schedule_valid(schedule: list[tuple[int, Job]]) -> None:
    used_slots = set()
    used_names = set()
    for slot, job in schedule:
        if slot < 1:
            raise AssertionError(f"invalid slot: {slot}")
        if slot in used_slots:
            raise AssertionError(f"duplicate slot: {slot}")
        if job.name in used_names:
            raise AssertionError(f"duplicate job: {job.name}")
        if slot > job.deadline:
            raise AssertionError(f"deadline violated for {job.name}: slot={slot}, deadline={job.deadline}")
        used_slots.add(slot)
        used_names.add(job.name)


def print_schedule(title: str, schedule: list[tuple[int, Job]], profit: int) -> None:
    print(title)
    if not schedule:
        print("  <empty>")
    else:
        for slot, job in schedule:
            print(
                f"  slot={slot:02d} | job={job.name:<6s} | deadline={job.deadline:02d} | profit={job.profit:03d}"
            )
    print(f"  total_profit={profit}")


def make_random_jobs(seed: int, n_jobs: int = 8) -> list[Job]:
    rng = random.Random(seed)
    jobs = []
    for idx in range(n_jobs):
        jobs.append(
            Job(
                name=f"J{idx}",
                deadline=rng.randint(1, 5),
                profit=rng.randint(5, 50),
            )
        )
    return jobs


def run_case(case_name: str, jobs: list[Job]) -> None:
    greedy_schedule, greedy_profit = schedule_jobs_greedy_dsu(jobs)
    exact_schedule, exact_profit = schedule_jobs_exact_bruteforce(jobs)

    assert_schedule_valid(greedy_schedule)
    assert_schedule_valid(exact_schedule)
    if greedy_profit != exact_profit:
        raise AssertionError(
            f"{case_name}: greedy profit {greedy_profit} != exact profit {exact_profit}"
        )

    print("=" * 72)
    print(case_name)
    print("jobs:")
    for job in jobs:
        print(f"  {job.name}: deadline={job.deadline}, profit={job.profit}")
    print_schedule("greedy schedule:", greedy_schedule, greedy_profit)
    print_schedule("exact schedule:", exact_schedule, exact_profit)


def main() -> None:
    fixed_jobs = [
        Job("A", 2, 100),
        Job("B", 1, 19),
        Job("C", 2, 27),
        Job("D", 1, 25),
        Job("E", 3, 15),
    ]
    run_case("fixed_case_textbook", fixed_jobs)

    easy_jobs = [
        Job("P", 1, 5),
        Job("Q", 2, 7),
        Job("R", 3, 9),
        Job("S", 4, 11),
    ]
    run_case("fixed_case_all_schedulable", easy_jobs)

    crowded_jobs = [
        Job("U", 1, 12),
        Job("V", 1, 30),
        Job("W", 1, 18),
        Job("X", 2, 22),
        Job("Y", 2, 17),
        Job("Z", 2, 45),
    ]
    run_case("fixed_case_competitive_deadlines", crowded_jobs)

    print("=" * 72)
    print("randomized cross-checks")
    for seed in range(25):
        jobs = make_random_jobs(seed, n_jobs=8)
        greedy_schedule, greedy_profit = schedule_jobs_greedy_dsu(jobs)
        exact_schedule, exact_profit = schedule_jobs_exact_bruteforce(jobs)
        assert_schedule_valid(greedy_schedule)
        assert_schedule_valid(exact_schedule)
        if greedy_profit != exact_profit:
            raise AssertionError(
                f"seed={seed}: greedy profit {greedy_profit} != exact profit {exact_profit}"
            )
    print("  randomized seeds checked: 25")
    print("All checks passed.")


if __name__ == "__main__":
    main()

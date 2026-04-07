"""MLFQ scheduling MVP (single-core, no I/O blocking)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Sequence, Tuple


@dataclass
class Process:
    pid: str
    arrival: int
    burst: int
    remaining: int
    level: int = 0
    first_start: int | None = None
    completion: int | None = None


TimelineEvent = Tuple[int, int, str, int]


def _validate_inputs(
    process_specs: Sequence[Tuple[str, int, int]],
    quantums: Sequence[int],
    boost_interval: int | None,
) -> None:
    if not quantums:
        raise ValueError("quantums must not be empty")
    if any(q <= 0 for q in quantums):
        raise ValueError("all quantums must be > 0")
    if boost_interval is not None and boost_interval <= 0:
        raise ValueError("boost_interval must be > 0 when provided")

    seen = set()
    for pid, arrival, burst in process_specs:
        if pid in seen:
            raise ValueError(f"duplicate pid: {pid}")
        seen.add(pid)
        if arrival < 0:
            raise ValueError(f"{pid}: arrival must be >= 0")
        if burst <= 0:
            raise ValueError(f"{pid}: burst must be > 0")


def mlfq_schedule(
    process_specs: Sequence[Tuple[str, int, int]],
    quantums: Sequence[int],
    boost_interval: int | None = None,
) -> Tuple[List[TimelineEvent], Dict[str, Dict[str, int]], Dict[str, float]]:
    _validate_inputs(process_specs, quantums, boost_interval)

    processes = [
        Process(pid=pid, arrival=arrival, burst=burst, remaining=burst)
        for pid, arrival, burst in process_specs
    ]
    processes.sort(key=lambda p: (p.arrival, p.pid))

    num_levels = len(quantums)
    queues: List[Deque[Process]] = [deque() for _ in range(num_levels)]
    timeline: List[TimelineEvent] = []

    time = 0
    next_idx = 0
    completed = 0
    n = len(processes)
    next_boost_time = boost_interval if boost_interval is not None else None

    def enqueue_arrived() -> None:
        nonlocal next_idx
        while next_idx < n and processes[next_idx].arrival <= time:
            p = processes[next_idx]
            p.level = 0
            queues[0].append(p)
            next_idx += 1

    def apply_priority_boost_if_due() -> None:
        nonlocal next_boost_time
        if next_boost_time is None:
            return
        while time >= next_boost_time:
            for level in range(1, num_levels):
                while queues[level]:
                    p = queues[level].popleft()
                    p.level = 0
                    queues[0].append(p)
            next_boost_time += boost_interval

    def highest_non_empty_level() -> int | None:
        for level, q in enumerate(queues):
            if q:
                return level
        return None

    while completed < n:
        enqueue_arrived()
        apply_priority_boost_if_due()

        level = highest_non_empty_level()
        if level is None:
            if next_idx >= n:
                break

            next_time = processes[next_idx].arrival
            if next_boost_time is not None:
                next_time = min(next_time, next_boost_time)
            if next_time > time:
                timeline.append((time, next_time, "IDLE", -1))
            time = next_time
            continue

        current = queues[level].popleft()
        if current.first_start is None:
            current.first_start = time

        remaining_slice = quantums[level]
        requeued_early = False

        while current.remaining > 0 and remaining_slice > 0:
            run_for = min(current.remaining, remaining_slice)

            if next_boost_time is not None and time + run_for > next_boost_time:
                run_for = next_boost_time - time

            if level > 0 and next_idx < n and time + run_for > processes[next_idx].arrival:
                run_for = processes[next_idx].arrival - time

            if run_for == 0:
                enqueue_arrived()
                apply_priority_boost_if_due()
                if level > 0 and queues[0]:
                    queues[level].append(current)
                    requeued_early = True
                    break
                continue

            start = time
            end = time + run_for
            timeline.append((start, end, current.pid, level))

            time = end
            current.remaining -= run_for
            remaining_slice -= run_for

            enqueue_arrived()

            hit_boost = next_boost_time is not None and time == next_boost_time
            if hit_boost:
                apply_priority_boost_if_due()
                current.level = 0
                queues[0].append(current)
                requeued_early = True
                break

            if level > 0 and current.remaining > 0 and queues[0]:
                queues[level].append(current)
                requeued_early = True
                break

        if requeued_early:
            continue

        if current.remaining == 0:
            current.completion = time
            completed += 1
            continue

        if remaining_slice == 0:
            new_level = min(level + 1, num_levels - 1)
            current.level = new_level
            queues[new_level].append(current)
        else:
            queues[level].appendleft(current)

    metrics: Dict[str, Dict[str, int]] = {}
    total_turnaround = 0
    total_waiting = 0
    total_response = 0

    for p in sorted(processes, key=lambda x: x.pid):
        if p.completion is None or p.first_start is None:
            raise RuntimeError(f"incomplete scheduling state for {p.pid}")
        turnaround = p.completion - p.arrival
        waiting = turnaround - p.burst
        response = p.first_start - p.arrival
        metrics[p.pid] = {
            "arrival": p.arrival,
            "burst": p.burst,
            "completion": p.completion,
            "turnaround": turnaround,
            "waiting": waiting,
            "response": response,
        }
        total_turnaround += turnaround
        total_waiting += waiting
        total_response += response

    avg = {
        "avg_turnaround": total_turnaround / n,
        "avg_waiting": total_waiting / n,
        "avg_response": total_response / n,
    }
    return timeline, metrics, avg


def format_timeline(timeline: Sequence[TimelineEvent]) -> str:
    if not timeline:
        return "(empty)"
    blocks: List[str] = []
    for start, end, pid, level in timeline:
        tag = pid if pid == "IDLE" else f"{pid}@Q{level}"
        blocks.append(f"[{start:>2},{end:>2}] {tag}")
    return " | ".join(blocks)


def main() -> None:
    quantums = [2, 4, 8]
    boost_interval = 12
    process_specs = [
        ("P1", 0, 9),
        ("P2", 1, 4),
        ("P3", 2, 6),
        ("P4", 8, 5),
        ("P5", 10, 2),
    ]

    timeline, metrics, avg = mlfq_schedule(
        process_specs=process_specs,
        quantums=quantums,
        boost_interval=boost_interval,
    )

    print("MLFQ Scheduling Demo")
    print(f"quantums = {quantums}")
    print(f"priority_boost_interval = {boost_interval}")
    print()
    print("Timeline:")
    print(format_timeline(timeline))
    print()
    print("Per-process metrics:")
    print(
        f"{'PID':<4}{'arr':>6}{'burst':>8}{'comp':>8}{'turn':>8}"
        f"{'wait':>8}{'resp':>8}"
    )
    for pid in sorted(metrics):
        m = metrics[pid]
        print(
            f"{pid:<4}{m['arrival']:>6}{m['burst']:>8}{m['completion']:>8}"
            f"{m['turnaround']:>8}{m['waiting']:>8}{m['response']:>8}"
        )
    print()
    print(
        "Averages: "
        f"turnaround={avg['avg_turnaround']:.2f}, "
        f"waiting={avg['avg_waiting']:.2f}, "
        f"response={avg['avg_response']:.2f}"
    )


if __name__ == "__main__":
    main()

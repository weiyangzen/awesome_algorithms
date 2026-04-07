"""Round Robin scheduling MVP (single-core, no I/O blocking)."""

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
    first_start: int | None = None
    completion: int | None = None


TimelineEvent = Tuple[int, int, str]


def _validate_inputs(process_specs: Sequence[Tuple[str, int, int]], quantum: int) -> None:
    if quantum <= 0:
        raise ValueError("quantum must be > 0")
    for pid, arrival, burst in process_specs:
        if arrival < 0:
            raise ValueError(f"{pid}: arrival must be >= 0")
        if burst <= 0:
            raise ValueError(f"{pid}: burst must be > 0")


def round_robin(
    process_specs: Sequence[Tuple[str, int, int]],
    quantum: int,
) -> Tuple[List[TimelineEvent], Dict[str, Dict[str, int]], Dict[str, float]]:
    _validate_inputs(process_specs, quantum)

    processes = [
        Process(pid=pid, arrival=arrival, burst=burst, remaining=burst)
        for pid, arrival, burst in process_specs
    ]
    processes.sort(key=lambda p: (p.arrival, p.pid))

    ready: Deque[Process] = deque()
    timeline: List[TimelineEvent] = []
    time = 0
    next_idx = 0
    completed = 0
    n = len(processes)

    def enqueue_arrived() -> None:
        nonlocal next_idx
        while next_idx < n and processes[next_idx].arrival <= time:
            ready.append(processes[next_idx])
            next_idx += 1

    enqueue_arrived()

    while completed < n:
        if not ready:
            if next_idx >= n:
                break
            idle_start = time
            time = processes[next_idx].arrival
            timeline.append((idle_start, time, "IDLE"))
            enqueue_arrived()
            continue

        current = ready.popleft()
        if current.first_start is None:
            current.first_start = time

        run_for = min(quantum, current.remaining)
        start = time
        end = time + run_for
        timeline.append((start, end, current.pid))

        time = end
        current.remaining -= run_for

        enqueue_arrived()

        if current.remaining > 0:
            ready.append(current)
        else:
            current.completion = time
            completed += 1

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

    blocks = []
    for start, end, pid in timeline:
        blocks.append(f"[{start:>2},{end:>2}] {pid}")
    return " | ".join(blocks)


def main() -> None:
    quantum = 2
    process_specs = [
        ("P1", 0, 5),
        ("P2", 1, 3),
        ("P3", 2, 8),
        ("P4", 4, 6),
    ]

    timeline, metrics, avg = round_robin(process_specs, quantum)

    print("Round Robin Scheduling Demo")
    print(f"quantum = {quantum}")
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

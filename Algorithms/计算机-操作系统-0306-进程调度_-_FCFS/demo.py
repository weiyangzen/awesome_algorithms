"""FCFS scheduling MVP (single-core, non-preemptive, no I/O blocking)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


@dataclass
class Process:
    pid: str
    arrival: int
    burst: int
    first_start: int | None = None
    completion: int | None = None


TimelineEvent = Tuple[int, int, str]


def _validate_inputs(process_specs: Sequence[Tuple[str, int, int]]) -> None:
    if not process_specs:
        raise ValueError("process_specs must not be empty")

    seen_pid = set()
    for pid, arrival, burst in process_specs:
        if pid in seen_pid:
            raise ValueError(f"duplicated pid: {pid}")
        seen_pid.add(pid)

        if arrival < 0:
            raise ValueError(f"{pid}: arrival must be >= 0")
        if burst <= 0:
            raise ValueError(f"{pid}: burst must be > 0")


def fcfs(
    process_specs: Sequence[Tuple[str, int, int]],
) -> Tuple[List[TimelineEvent], Dict[str, Dict[str, int]], Dict[str, float]]:
    """Run FCFS and return (timeline, per-process metrics, averages)."""
    _validate_inputs(process_specs)

    processes = [
        Process(pid=pid, arrival=arrival, burst=burst)
        for pid, arrival, burst in process_specs
    ]
    processes.sort(key=lambda p: (p.arrival, p.pid))

    timeline: List[TimelineEvent] = []
    time = 0

    for p in processes:
        if time < p.arrival:
            timeline.append((time, p.arrival, "IDLE"))
            time = p.arrival

        p.first_start = time
        end = time + p.burst
        timeline.append((time, end, p.pid))

        time = end
        p.completion = end

    metrics: Dict[str, Dict[str, int]] = {}
    total_turnaround = 0
    total_waiting = 0
    total_response = 0
    n = len(processes)

    for p in sorted(processes, key=lambda x: x.pid):
        if p.completion is None or p.first_start is None:
            raise RuntimeError(f"incomplete scheduling state for {p.pid}")

        turnaround = p.completion - p.arrival
        waiting = turnaround - p.burst
        response = p.first_start - p.arrival

        metrics[p.pid] = {
            "arrival": p.arrival,
            "burst": p.burst,
            "start": p.first_start,
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
    return " | ".join(f"[{start:>2},{end:>2}] {pid}" for start, end, pid in timeline)


def main() -> None:
    process_specs = [
        ("P1", 1, 6),
        ("P2", 2, 8),
        ("P3", 3, 2),
        ("P4", 5, 4),
    ]

    timeline, metrics, avg = fcfs(process_specs)

    print("FCFS Scheduling Demo")
    print()
    print("Timeline:")
    print(format_timeline(timeline))
    print()
    print("Per-process metrics:")
    print(
        f"{'PID':<4}{'arr':>6}{'burst':>8}{'start':>8}{'comp':>8}"
        f"{'turn':>8}{'wait':>8}{'resp':>8}"
    )
    for pid in sorted(metrics):
        m = metrics[pid]
        print(
            f"{pid:<4}{m['arrival']:>6}{m['burst']:>8}{m['start']:>8}"
            f"{m['completion']:>8}{m['turnaround']:>8}{m['waiting']:>8}{m['response']:>8}"
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

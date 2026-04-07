"""Priority Scheduling (non-preemptive) MVP.

The scheduler assumes:
- Smaller numeric priority value means higher scheduling priority.
- CPU runs one process at a time and does not preempt a running process.
- A process can be chosen only after it has arrived.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence


@dataclass(frozen=True)
class Process:
    pid: str
    arrival: int
    burst: int
    priority: int


@dataclass(frozen=True)
class ScheduleRecord:
    pid: str
    arrival: int
    burst: int
    priority: int
    start: int
    finish: int
    waiting: int
    turnaround: int
    response: int


def _pick_next(ready: Sequence[Process]) -> Process:
    """Pick process by (priority asc, arrival asc, pid asc)."""
    return min(ready, key=lambda p: (p.priority, p.arrival, p.pid))


def priority_non_preemptive(processes: Sequence[Process]) -> List[ScheduleRecord]:
    """Run non-preemptive priority scheduling and return per-process records."""
    pending = sorted(processes, key=lambda p: (p.arrival, p.pid))
    time = 0
    i = 0
    n = len(pending)
    ready: List[Process] = []
    done: List[ScheduleRecord] = []

    while len(done) < n:
        while i < n and pending[i].arrival <= time:
            ready.append(pending[i])
            i += 1

        if not ready:
            # CPU idle until the next process arrives.
            time = pending[i].arrival
            continue

        job = _pick_next(ready)
        ready.remove(job)

        start = time
        finish = start + job.burst
        waiting = start - job.arrival
        turnaround = finish - job.arrival
        response = waiting  # non-preemptive

        done.append(
            ScheduleRecord(
                pid=job.pid,
                arrival=job.arrival,
                burst=job.burst,
                priority=job.priority,
                start=start,
                finish=finish,
                waiting=waiting,
                turnaround=turnaround,
                response=response,
            )
        )

        time = finish

    return done


def _print_table(records: Sequence[ScheduleRecord]) -> None:
    headers = [
        "PID",
        "Arrival",
        "Burst",
        "Prio",
        "Start",
        "Finish",
        "Waiting",
        "Turnaround",
        "Response",
    ]
    rows = [
        [
            r.pid,
            str(r.arrival),
            str(r.burst),
            str(r.priority),
            str(r.start),
            str(r.finish),
            str(r.waiting),
            str(r.turnaround),
            str(r.response),
        ]
        for r in records
    ]

    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def fmt(row: Sequence[str]) -> str:
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    print(fmt(headers))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(fmt(row))


def _print_summary(records: Sequence[ScheduleRecord]) -> None:
    n = len(records)
    avg_wait = sum(r.waiting for r in records) / n
    avg_turn = sum(r.turnaround for r in records) / n
    avg_resp = sum(r.response for r in records) / n
    makespan = max(r.finish for r in records) - min(r.start for r in records)

    print("\nSummary")
    print(f"- Avg waiting time    : {avg_wait:.2f}")
    print(f"- Avg turnaround time : {avg_turn:.2f}")
    print(f"- Avg response time   : {avg_resp:.2f}")
    print(f"- Makespan            : {makespan}")


def _print_gantt(records: Sequence[ScheduleRecord]) -> None:
    print("\nGantt")
    timeline = " ".join(f"[{r.start}-{r.finish}:{r.pid}]" for r in records)
    print(timeline)


def main() -> None:
    # Example workload: smaller priority number => higher priority.
    processes = [
        Process(pid="P1", arrival=0, burst=7, priority=3),
        Process(pid="P2", arrival=2, burst=4, priority=1),
        Process(pid="P3", arrival=4, burst=1, priority=4),
        Process(pid="P4", arrival=5, burst=4, priority=2),
    ]

    records = priority_non_preemptive(processes)
    _print_table(records)
    _print_summary(records)
    _print_gantt(records)


if __name__ == "__main__":
    main()

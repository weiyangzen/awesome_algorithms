"""CFS scheduling MVP (single-core, no I/O blocking)."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

NICE_0_LOAD = 1024
# Linux kernel-like weights for nice=-20..19 (prio_to_weight table).
PRIO_TO_WEIGHT = [
    88761,
    71755,
    56483,
    46273,
    36291,
    29154,
    23254,
    18705,
    14949,
    11916,
    9548,
    7620,
    6100,
    4904,
    3906,
    3121,
    2501,
    1991,
    1586,
    1277,
    1024,
    820,
    655,
    526,
    423,
    335,
    272,
    215,
    172,
    137,
    110,
    87,
    70,
    56,
    45,
    36,
    29,
    23,
    18,
    15,
]


@dataclass
class Process:
    pid: str
    arrival: float
    burst: float
    nice: int
    weight: int
    remaining: float
    vruntime: float = 0.0
    executed: float = 0.0
    first_start: float | None = None
    completion: float | None = None


# start, end, pid, nice, vruntime_before
TimelineEvent = Tuple[float, float, str, int, float]


def nice_to_weight(nice: int) -> int:
    if nice < -20 or nice > 19:
        raise ValueError(f"nice out of range [-20, 19]: {nice}")
    return PRIO_TO_WEIGHT[nice + 20]


def _validate_inputs(
    process_specs: Sequence[Tuple[str, float, float, int]],
    target_latency: float,
    min_granularity: float,
) -> None:
    if target_latency <= 0:
        raise ValueError("target_latency must be > 0")
    if min_granularity <= 0:
        raise ValueError("min_granularity must be > 0")

    seen = set()
    for pid, arrival, burst, nice in process_specs:
        if pid in seen:
            raise ValueError(f"duplicate pid: {pid}")
        seen.add(pid)

        if arrival < 0:
            raise ValueError(f"{pid}: arrival must be >= 0")
        if burst <= 0:
            raise ValueError(f"{pid}: burst must be > 0")
        if nice < -20 or nice > 19:
            raise ValueError(f"{pid}: nice must be in [-20, 19]")


def _sched_period(num_running: int, target_latency: float, min_granularity: float) -> float:
    nr_latency = int(target_latency / min_granularity)
    if nr_latency <= 0:
        nr_latency = 1
    if num_running <= nr_latency:
        return target_latency
    return num_running * min_granularity


def cfs_schedule(
    process_specs: Sequence[Tuple[str, float, float, int]],
    target_latency: float = 12.0,
    min_granularity: float = 1.0,
) -> Tuple[List[TimelineEvent], Dict[str, Dict[str, float]], Dict[str, float]]:
    _validate_inputs(process_specs, target_latency, min_granularity)

    processes = [
        Process(
            pid=pid,
            arrival=float(arrival),
            burst=float(burst),
            nice=nice,
            weight=nice_to_weight(nice),
            remaining=float(burst),
        )
        for pid, arrival, burst, nice in process_specs
    ]
    processes.sort(key=lambda p: (p.arrival, p.pid))

    timeline: List[TimelineEvent] = []
    heap: List[Tuple[float, int, Process]] = []

    time = 0.0
    next_idx = 0
    completed = 0
    n = len(processes)
    tie_seq = 0

    def enqueue_arrivals(up_to_time: float) -> None:
        nonlocal next_idx, tie_seq
        while next_idx < n and processes[next_idx].arrival <= up_to_time:
            p = processes[next_idx]
            heapq.heappush(heap, (p.vruntime, tie_seq, p))
            tie_seq += 1
            next_idx += 1

    while completed < n:
        enqueue_arrivals(time)

        if not heap:
            if next_idx >= n:
                break
            next_time = processes[next_idx].arrival
            if next_time > time:
                timeline.append((time, next_time, "IDLE", 0, 0.0))
            time = next_time
            enqueue_arrivals(time)
            continue

        _, _, current = heapq.heappop(heap)
        if current.first_start is None:
            current.first_start = time

        total_weight = current.weight + sum(p.weight for _, _, p in heap)
        num_running = len(heap) + 1
        period = _sched_period(num_running, target_latency, min_granularity)
        ideal_slice = max(min_granularity, period * current.weight / total_weight)

        run_for = min(current.remaining, ideal_slice)
        if next_idx < n:
            next_arrival = processes[next_idx].arrival
            if next_arrival < time + run_for:
                run_for = next_arrival - time

        if run_for <= 0:
            enqueue_arrivals(time)
            heapq.heappush(heap, (current.vruntime, tie_seq, current))
            tie_seq += 1
            continue

        start = time
        end = time + run_for
        timeline.append((start, end, current.pid, current.nice, current.vruntime))

        time = end
        current.remaining -= run_for
        current.executed += run_for
        current.vruntime += run_for * NICE_0_LOAD / current.weight

        enqueue_arrivals(time)

        if current.remaining <= 1e-12:
            current.remaining = 0.0
            current.completion = time
            completed += 1
        else:
            heapq.heappush(heap, (current.vruntime, tie_seq, current))
            tie_seq += 1

    metrics: Dict[str, Dict[str, float]] = {}
    total_turnaround = 0.0
    total_waiting = 0.0
    total_response = 0.0

    total_exec = sum(p.executed for p in processes)
    total_weight_all = sum(p.weight for p in processes)

    for p in sorted(processes, key=lambda x: x.pid):
        if p.first_start is None or p.completion is None:
            raise RuntimeError(f"incomplete scheduling state for {p.pid}")

        turnaround = p.completion - p.arrival
        waiting = turnaround - p.burst
        response = p.first_start - p.arrival
        cpu_share = p.executed / total_exec if total_exec > 0 else 0.0
        weight_share = p.weight / total_weight_all if total_weight_all > 0 else 0.0

        metrics[p.pid] = {
            "arrival": p.arrival,
            "burst": p.burst,
            "nice": float(p.nice),
            "weight": float(p.weight),
            "completion": p.completion,
            "turnaround": turnaround,
            "waiting": waiting,
            "response": response,
            "exec": p.executed,
            "cpu_share": cpu_share,
            "weight_share": weight_share,
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


def _fmt(x: float) -> str:
    return f"{x:.2f}"


def main() -> None:
    process_specs = [
        ("P1", 0, 18, 0),
        ("P2", 0, 8, 5),
        ("P3", 2, 6, -5),
        ("P4", 4, 12, 10),
        ("P5", 6, 4, -10),
    ]
    target_latency = 12.0
    min_granularity = 1.0

    timeline, metrics, avg = cfs_schedule(
        process_specs=process_specs,
        target_latency=target_latency,
        min_granularity=min_granularity,
    )

    print("CFS Scheduling Demo")
    print(f"target_latency={target_latency}, min_granularity={min_granularity}")
    print("processes=(pid, arrival, burst, nice):")
    for spec in process_specs:
        print(f"  {spec}")

    print("\nTimeline:")
    print(f"{'start':>8}{'end':>8}{'pid':>6}{'nice':>8}{'vruntime_before':>18}")
    for start, end, pid, nice, vbefore in timeline:
        print(
            f"{_fmt(start):>8}{_fmt(end):>8}{pid:>6}{nice:>8}"
            f"{_fmt(vbefore):>18}"
        )

    print("\nPer-process metrics:")
    print(
        f"{'PID':<4}{'arr':>7}{'burst':>7}{'nice':>7}{'weight':>8}{'comp':>8}"
        f"{'turn':>8}{'wait':>8}{'resp':>8}{'cpu%':>8}{'w%':>8}"
    )
    for pid in sorted(metrics):
        m = metrics[pid]
        print(
            f"{pid:<4}{_fmt(m['arrival']):>7}{_fmt(m['burst']):>7}{int(m['nice']):>7}"
            f"{int(m['weight']):>8}{_fmt(m['completion']):>8}{_fmt(m['turnaround']):>8}"
            f"{_fmt(m['waiting']):>8}{_fmt(m['response']):>8}"
            f"{100*m['cpu_share']:>7.2f}%{100*m['weight_share']:>7.2f}%"
        )

    print("\nAverages:")
    print(
        f"turnaround={avg['avg_turnaround']:.2f}, "
        f"waiting={avg['avg_waiting']:.2f}, "
        f"response={avg['avg_response']:.2f}"
    )


if __name__ == "__main__":
    main()

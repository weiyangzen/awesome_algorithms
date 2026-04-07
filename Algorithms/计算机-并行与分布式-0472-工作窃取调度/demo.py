"""工作窃取调度最小可运行 MVP (CS-0311).

运行:
    uv run python demo.py
"""

from __future__ import annotations

import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Task:
    """调度任务定义."""

    task_id: int
    cost_s: float
    home_worker: int


class WorkStealingDeque:
    """线程安全双端队列：owner 在 bottom 操作，thief 在 top 窃取。"""

    def __init__(self) -> None:
        self._dq: deque[Task] = deque()
        self._lock = threading.Lock()

    def push_bottom(self, task: Task) -> None:
        with self._lock:
            self._dq.append(task)

    def pop_bottom(self) -> Task | None:
        with self._lock:
            if not self._dq:
                return None
            return self._dq.pop()

    def steal_top(self) -> Task | None:
        with self._lock:
            if not self._dq:
                return None
            return self._dq.popleft()


@dataclass
class SchedulerStats:
    """调度统计信息（线程安全写入）。"""

    tasks_executed: list[int]
    stolen_executed: list[int]
    steal_attempt: list[int]
    steal_success: list[int]
    busy_time_s: list[float]
    planned_work_s: list[float]
    seen_task_ids: set[int]
    lock: threading.Lock

    @classmethod
    def create(cls, workers: int) -> "SchedulerStats":
        return cls(
            tasks_executed=[0] * workers,
            stolen_executed=[0] * workers,
            steal_attempt=[0] * workers,
            steal_success=[0] * workers,
            busy_time_s=[0.0] * workers,
            planned_work_s=[0.0] * workers,
            seen_task_ids=set(),
            lock=threading.Lock(),
        )

    def record_steal_attempt(self, worker_id: int) -> None:
        with self.lock:
            self.steal_attempt[worker_id] += 1

    def record_steal_success(self, worker_id: int) -> None:
        with self.lock:
            self.steal_success[worker_id] += 1

    def record_execution(self, worker_id: int, task: Task, elapsed_s: float) -> None:
        with self.lock:
            if task.task_id in self.seen_task_ids:
                raise RuntimeError(f"task duplicated: {task.task_id}")
            self.seen_task_ids.add(task.task_id)

            self.tasks_executed[worker_id] += 1
            self.busy_time_s[worker_id] += elapsed_s
            self.planned_work_s[worker_id] += task.cost_s
            if task.home_worker != worker_id:
                self.stolen_executed[worker_id] += 1


@dataclass
class RunResult:
    strategy: str
    makespan_s: float
    throughput_task_per_s: float
    stats: SchedulerStats
    total_tasks: int
    total_work_s: float


def generate_task_costs(num_tasks: int, seed: int) -> np.ndarray:
    """生成重尾任务耗时（秒）。"""
    if num_tasks <= 0:
        raise ValueError("num_tasks must be positive")

    rng = np.random.default_rng(seed)
    # 对数正态分布模拟任务长短不一，裁剪到可控范围。
    costs = rng.lognormal(mean=-5.1, sigma=0.8, size=num_tasks)
    costs = np.clip(costs, 0.0015, 0.0300)
    return costs.astype(float)


def build_initial_assignment(
    task_costs: np.ndarray,
    workers: int,
    skew_ratio: float,
    seed: int,
) -> list[list[Task]]:
    """按倾斜比例构造初始任务分配。"""
    if workers < 2:
        raise ValueError("workers must be >= 2")
    if not (0.0 <= skew_ratio <= 1.0):
        raise ValueError("skew_ratio must be in [0, 1]")

    n = len(task_costs)
    rng = np.random.default_rng(seed)
    order = np.arange(n)
    rng.shuffle(order)

    hot_count = int(round(n * skew_ratio))
    assignments: list[list[Task]] = [[] for _ in range(workers)]

    for pos, idx in enumerate(order):
        if pos < hot_count:
            home = 0
        else:
            home = int(rng.integers(1, workers))
        assignments[home].append(Task(task_id=int(idx), cost_s=float(task_costs[idx]), home_worker=home))

    return assignments


def _build_deques(assignments: list[list[Task]]) -> list[WorkStealingDeque]:
    deques = [WorkStealingDeque() for _ in range(len(assignments))]
    for worker_id, tasks in enumerate(assignments):
        _ = worker_id
        for task in tasks:
            deques[task.home_worker].push_bottom(task)
    return deques


def run_scheduler(
    strategy: str,
    assignments: list[list[Task]],
    allow_steal: bool,
    seed: int,
    idle_backoff_s: float = 0.0005,
) -> RunResult:
    """运行调度器，支持切换是否允许窃取。"""
    workers = len(assignments)
    deques = _build_deques(assignments)
    stats = SchedulerStats.create(workers)

    total_tasks = sum(len(x) for x in assignments)
    total_work_s = float(sum(task.cost_s for queue in assignments for task in queue))

    remaining = [total_tasks]
    remaining_lock = threading.Lock()

    def worker_loop(worker_id: int) -> None:
        rng = np.random.default_rng(seed + 1000 + worker_id)
        victims = [v for v in range(workers) if v != worker_id]

        while True:
            with remaining_lock:
                if remaining[0] == 0:
                    return

            task = deques[worker_id].pop_bottom()

            if task is None and allow_steal and victims:
                start = int(rng.integers(0, len(victims)))
                for offset in range(len(victims)):
                    victim = victims[(start + offset) % len(victims)]
                    stats.record_steal_attempt(worker_id)
                    task = deques[victim].steal_top()
                    if task is not None:
                        stats.record_steal_success(worker_id)
                        break

            if task is None:
                time.sleep(idle_backoff_s)
                continue

            t0 = perf_counter()
            time.sleep(task.cost_s)
            elapsed = perf_counter() - t0
            stats.record_execution(worker_id, task, elapsed)

            with remaining_lock:
                remaining[0] -= 1
                if remaining[0] < 0:
                    raise RuntimeError("remaining task counter became negative")

    t0 = perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(worker_loop, wid) for wid in range(workers)]
        for fut in futures:
            fut.result()
    makespan = perf_counter() - t0

    executed_total = sum(stats.tasks_executed)
    if executed_total != total_tasks:
        raise AssertionError(f"executed tasks mismatch: {executed_total} != {total_tasks}")

    if len(stats.seen_task_ids) != total_tasks:
        raise AssertionError("task uniqueness check failed")

    executed_work = float(sum(stats.planned_work_s))
    if not np.isclose(executed_work, total_work_s, atol=1e-10):
        raise AssertionError("executed work mismatch")

    throughput = total_tasks / makespan if makespan > 0 else float("inf")

    return RunResult(
        strategy=strategy,
        makespan_s=makespan,
        throughput_task_per_s=throughput,
        stats=stats,
        total_tasks=total_tasks,
        total_work_s=total_work_s,
    )


def summarize_result(result: RunResult) -> tuple[pd.DataFrame, float]:
    """构建每个 worker 的统计表与负载 CV 指标。"""
    workers = len(result.stats.tasks_executed)
    rows: list[dict[str, float | int | str]] = []

    for wid in range(workers):
        busy = result.stats.busy_time_s[wid]
        rows.append(
            {
                "worker": f"w{wid}",
                "tasks_executed": result.stats.tasks_executed[wid],
                "stolen_executed": result.stats.stolen_executed[wid],
                "steal_attempt": result.stats.steal_attempt[wid],
                "steal_success": result.stats.steal_success[wid],
                "planned_work_s": result.stats.planned_work_s[wid],
                "busy_time_s": busy,
                "utilization": busy / result.makespan_s if result.makespan_s > 0 else 0.0,
            }
        )

    df = pd.DataFrame(rows)
    counts = df["tasks_executed"].to_numpy(dtype=float)
    cv = float(np.std(counts) / np.mean(counts)) if np.mean(counts) > 0 else 0.0
    return df, cv


def print_result(result: RunResult) -> float:
    df, cv = summarize_result(result)

    total_steal_success = int(sum(result.stats.steal_success))
    total_steal_attempt = int(sum(result.stats.steal_attempt))

    print(f"\n=== {result.strategy} ===")
    print(f"makespan_s: {result.makespan_s:.4f}")
    print(f"throughput_task_per_s: {result.throughput_task_per_s:.2f}")
    print(f"total_steal_success: {total_steal_success}")
    print(f"total_steal_attempt: {total_steal_attempt}")
    print(f"task_count_cv: {cv:.4f}")
    print(
        df.to_string(
            index=False,
            formatters={
                "planned_work_s": "{:.4f}".format,
                "busy_time_s": "{:.4f}".format,
                "utilization": "{:.2%}".format,
            },
        )
    )
    return cv


def main() -> None:
    workers = 8
    num_tasks = 240
    skew_ratio = 0.85
    seed = 20260407

    task_costs = generate_task_costs(num_tasks=num_tasks, seed=seed)
    assignments = build_initial_assignment(
        task_costs=task_costs,
        workers=workers,
        skew_ratio=skew_ratio,
        seed=seed + 1,
    )

    total_work_s = float(np.sum(task_costs))
    print("=== Work-Stealing Scheduling MVP ===")
    print(f"workers: {workers}, tasks: {num_tasks}, skew_ratio: {skew_ratio:.2f}, seed: {seed}")
    print(f"total_planned_work_s: {total_work_s:.4f}")

    baseline = run_scheduler(
        strategy="Baseline(no-steal)",
        assignments=assignments,
        allow_steal=False,
        seed=seed + 11,
    )
    ws = run_scheduler(
        strategy="WorkStealing",
        assignments=assignments,
        allow_steal=True,
        seed=seed + 29,
    )

    cv_baseline = print_result(baseline)
    cv_ws = print_result(ws)

    speedup = baseline.makespan_s / ws.makespan_s if ws.makespan_s > 0 else float("inf")
    print("\n=== Comparison ===")
    print(f"speedup(no-steal / work-stealing): {speedup:.3f}x")
    print(f"cv_improvement: {cv_baseline:.4f} -> {cv_ws:.4f}")

    if sum(ws.stats.steal_success) <= 0:
        raise AssertionError("work-stealing run should have successful steals under skewed setup")

    print("All checks passed for CS-0311 (工作窃取调度).")


if __name__ == "__main__":
    main()

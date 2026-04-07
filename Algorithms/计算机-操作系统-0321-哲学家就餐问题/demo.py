"""哲学家就餐问题：监视器 + 条件变量的最小可运行示例。"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np


THINKING = 0
HUNGRY = 1
EATING = 2


class DiningMonitor:
    """哲学家就餐监视器。

    关键思想：
    - 全局锁串行化状态更新；
    - 每位哲学家一个条件变量；
    - 只有左右邻居都不在进餐时，才允许当前哲学家进餐。
    """

    def __init__(self, num_philosophers: int) -> None:
        if num_philosophers < 2:
            raise ValueError("num_philosophers must be >= 2")

        self.n = num_philosophers
        self.lock = threading.Lock()
        self.conditions = [threading.Condition(self.lock) for _ in range(self.n)]
        self.state = [THINKING for _ in range(self.n)]

    def left(self, i: int) -> int:
        return (i - 1 + self.n) % self.n

    def right(self, i: int) -> int:
        return (i + 1) % self.n

    def _test(self, i: int) -> None:
        if (
            self.state[i] == HUNGRY
            and self.state[self.left(i)] != EATING
            and self.state[self.right(i)] != EATING
        ):
            self.state[i] = EATING
            self.conditions[i].notify()

    def pickup(self, i: int) -> float:
        """请求进餐，返回等待时间（秒）。"""
        start = time.perf_counter()
        with self.lock:
            self.state[i] = HUNGRY
            self._test(i)
            while self.state[i] != EATING:
                self.conditions[i].wait()
        return time.perf_counter() - start

    def putdown(self, i: int) -> None:
        with self.lock:
            self.state[i] = THINKING
            self._test(self.left(i))
            self._test(self.right(i))


@dataclass
class SimulationResult:
    eat_counts: list[int]
    wait_stats: dict[str, float]
    mean_wait_by_philosopher_ms: list[float]
    jain_fairness_index: float
    max_active_eaters: int
    adjacent_conflicts: int
    total_events: int


def summarize_wait(wait_seconds: list[float]) -> dict[str, float]:
    arr = np.array(wait_seconds, dtype=float)
    if arr.size == 0:
        return {"mean_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    return {
        "mean_ms": float(arr.mean() * 1000.0),
        "p95_ms": float(np.percentile(arr, 95) * 1000.0),
        "max_ms": float(arr.max() * 1000.0),
    }


def jain_index(x: list[int]) -> float:
    arr = np.array(x, dtype=float)
    if arr.size == 0:
        return 1.0
    denom = float(arr.size * np.square(arr).sum())
    if denom == 0.0:
        return 1.0
    return float((arr.sum() ** 2) / denom)


def run_simulation(
    num_philosophers: int = 5,
    rounds: int = 30,
    seed: int = 20260407,
    think_range: tuple[float, float] = (0.0008, 0.0030),
    eat_range: tuple[float, float] = (0.0008, 0.0025),
) -> SimulationResult:
    monitor = DiningMonitor(num_philosophers)

    eat_counts = [0 for _ in range(num_philosophers)]
    wait_all: list[float] = []
    wait_by_philosopher: list[list[float]] = [[] for _ in range(num_philosophers)]

    events: list[dict[str, Any]] = []
    metrics_lock = threading.Lock()
    active_eaters: set[int] = set()
    max_active_eaters = 0
    adjacent_conflicts = 0

    def philosopher_worker(i: int) -> None:
        nonlocal max_active_eaters, adjacent_conflicts
        rng = random.Random(seed + i * 137)
        actor = f"P{i}"

        for step in range(rounds):
            time.sleep(rng.uniform(*think_range))

            wait_s = monitor.pickup(i)
            enter_t = time.perf_counter()

            with metrics_lock:
                wait_all.append(wait_s)
                wait_by_philosopher[i].append(wait_s)

                left_neighbor = monitor.left(i)
                right_neighbor = monitor.right(i)
                if left_neighbor in active_eaters or right_neighbor in active_eaters:
                    adjacent_conflicts += 1

                active_eaters.add(i)
                max_active_eaters = max(max_active_eaters, len(active_eaters))
                eat_counts[i] += 1

                events.append(
                    {
                        "t": enter_t,
                        "actor": actor,
                        "kind": "eat_enter",
                        "step": step,
                        "wait_ms": wait_s * 1000.0,
                        "active_eaters": len(active_eaters),
                    }
                )

            time.sleep(rng.uniform(*eat_range))

            with metrics_lock:
                active_eaters.remove(i)
                events.append(
                    {
                        "t": time.perf_counter(),
                        "actor": actor,
                        "kind": "eat_exit",
                        "step": step,
                        "active_eaters": len(active_eaters),
                    }
                )

            monitor.putdown(i)

    threads = [
        threading.Thread(target=philosopher_worker, args=(i,), name=f"philosopher-{i}")
        for i in range(num_philosophers)
    ]

    random.Random(seed).shuffle(threads)

    start_t = time.perf_counter()
    for th in threads:
        th.start()

    for th in threads:
        th.join(timeout=20.0)

    elapsed_ms = (time.perf_counter() - start_t) * 1000.0

    alive = [th.name for th in threads if th.is_alive()]
    assert not alive, f"Threads did not finish: {alive}"

    expected_each = rounds
    expected_total = num_philosophers * rounds

    assert all(c == expected_each for c in eat_counts), (
        f"Each philosopher should eat {expected_each} times, got {eat_counts}"
    )
    assert sum(eat_counts) == expected_total, (
        f"Total eat count mismatch: got {sum(eat_counts)}, expected {expected_total}"
    )
    assert adjacent_conflicts == 0, f"Adjacent philosophers ate together: {adjacent_conflicts}"
    assert max_active_eaters <= num_philosophers // 2, (
        "Max simultaneously eating philosophers exceeds theoretical bound"
    )

    wait_stats = summarize_wait(wait_all)
    mean_wait_by_philosopher_ms = [
        float(np.mean(w) * 1000.0) if w else 0.0 for w in wait_by_philosopher
    ]
    fairness = jain_index(eat_counts)

    print("=== Dining Philosophers Monitor Simulation ===")
    print(f"Philosophers: {num_philosophers}, rounds each: {rounds}")
    print(f"Elapsed: {elapsed_ms:.2f} ms")
    print(f"Eat counts: {eat_counts}")
    print(
        "Wait stats (ms): "
        f"mean={wait_stats['mean_ms']:.3f}, "
        f"p95={wait_stats['p95_ms']:.3f}, "
        f"max={wait_stats['max_ms']:.3f}"
    )
    print("Mean wait by philosopher (ms):", [round(x, 3) for x in mean_wait_by_philosopher_ms])
    print(f"Jain fairness index (eat counts): {fairness:.4f}")
    print(f"Max active eaters: {max_active_eaters}")
    print(f"Adjacent conflict events: {adjacent_conflicts}")
    print(f"Total events recorded: {len(events)}")
    print("All assertions passed.")

    return SimulationResult(
        eat_counts=eat_counts,
        wait_stats=wait_stats,
        mean_wait_by_philosopher_ms=mean_wait_by_philosopher_ms,
        jain_fairness_index=fairness,
        max_active_eaters=max_active_eaters,
        adjacent_conflicts=adjacent_conflicts,
        total_events=len(events),
    )


def main() -> None:
    run_simulation()


if __name__ == "__main__":
    main()

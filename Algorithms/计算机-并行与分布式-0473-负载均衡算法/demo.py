"""负载均衡算法最小可运行 MVP（离散事件仿真版）。

策略包含：
- round_robin: 轮询
- weighted_round_robin: 加权轮询
- least_connections: 最少连接（按连接数/容量比）
- power_of_two_choices: 二选一随机（P2C）
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Backend:
    """后端节点定义。"""

    name: str
    weight: int
    capacity: int


class LoadBalancer:
    """支持多策略的负载均衡器。"""

    def __init__(self, backends: Sequence[Backend], seed: int = 0) -> None:
        if not backends:
            raise ValueError("backends must not be empty")
        self.backends = list(backends)
        self._n = len(backends)
        self._rng = np.random.default_rng(seed)

        self._rr_cursor = 0
        self._wrr_schedule = [
            idx for idx, backend in enumerate(self.backends) for _ in range(backend.weight)
        ]
        if not self._wrr_schedule:
            raise ValueError("at least one backend must have positive weight")
        self._wrr_cursor = 0

    def choose(self, strategy: str, active_connections: np.ndarray) -> int:
        if strategy == "round_robin":
            idx = self._rr_cursor
            self._rr_cursor = (self._rr_cursor + 1) % self._n
            return idx

        if strategy == "weighted_round_robin":
            idx = self._wrr_schedule[self._wrr_cursor]
            self._wrr_cursor = (self._wrr_cursor + 1) % len(self._wrr_schedule)
            return idx

        capacities = np.array([b.capacity for b in self.backends], dtype=float)
        load_ratio = active_connections / capacities

        if strategy == "least_connections":
            return self._argmin_with_rotating_tie_break(load_ratio)

        if strategy == "power_of_two_choices":
            if self._n == 1:
                return 0
            i, j = self._rng.choice(self._n, size=2, replace=False)
            if load_ratio[i] < load_ratio[j]:
                return int(i)
            if load_ratio[j] < load_ratio[i]:
                return int(j)
            return int(min(i, j))

        raise ValueError(f"unsupported strategy: {strategy}")

    def _argmin_with_rotating_tie_break(self, scores: np.ndarray) -> int:
        min_score = float(np.min(scores))
        candidates = [i for i, s in enumerate(scores) if float(s) == min_score]
        candidates.sort()
        for step in range(self._n):
            idx = (self._rr_cursor + step) % self._n
            if idx in candidates:
                self._rr_cursor = (idx + 1) % self._n
                return idx
        return candidates[0]


def generate_workload(
    n_requests: int,
    arrival_rate: float,
    service_log_mean: float,
    service_log_sigma: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """生成到达时间与服务时长。"""
    rng = np.random.default_rng(seed)
    inter_arrivals = rng.exponential(scale=1.0 / arrival_rate, size=n_requests)
    arrivals = np.cumsum(inter_arrivals)
    service_times = rng.lognormal(
        mean=service_log_mean, sigma=service_log_sigma, size=n_requests
    )
    return arrivals, service_times


def simulate_strategy(
    strategy: str,
    backends: Sequence[Backend],
    arrivals: np.ndarray,
    service_times: np.ndarray,
    lb_seed: int,
) -> Dict[str, object]:
    lb = LoadBalancer(backends, seed=lb_seed)
    n = len(backends)

    completion_heaps: List[List[float]] = [[] for _ in range(n)]
    assigned_counts = np.zeros(n, dtype=int)
    busy_time = np.zeros(n, dtype=float)

    waiting_time = np.zeros_like(arrivals)
    finish_time = np.zeros_like(arrivals)

    for req_id, (arrival, service) in enumerate(zip(arrivals, service_times)):
        active = np.zeros(n, dtype=int)
        for i in range(n):
            heap = completion_heaps[i]
            while heap and heap[0] <= arrival:
                heapq.heappop(heap)
            active[i] = len(heap)

        target = lb.choose(strategy, active)
        backend = backends[target]
        heap = completion_heaps[target]

        if len(heap) < backend.capacity:
            start = float(arrival)
        else:
            # 容量已满：请求在该节点排队到最早空闲时刻。
            start = float(heapq.heappop(heap))

        finish = start + float(service)
        heapq.heappush(heap, finish)

        assigned_counts[target] += 1
        busy_time[target] += float(service)
        waiting_time[req_id] = start - float(arrival)
        finish_time[req_id] = finish

    makespan = float(np.max(finish_time))
    capacities = np.array([b.capacity for b in backends], dtype=float)
    utilization = busy_time / np.maximum(makespan * capacities, 1e-12)

    return {
        "strategy": strategy,
        "counts": {
            backends[i].name: int(assigned_counts[i]) for i in range(n)
        },
        "mean_wait": float(np.mean(waiting_time)),
        "p95_wait": float(np.percentile(waiting_time, 95)),
        "max_wait": float(np.max(waiting_time)),
        "imbalance_cv": float(np.std(assigned_counts) / np.mean(assigned_counts)),
        "utilization": {
            backends[i].name: float(utilization[i]) for i in range(n)
        },
    }


def format_result_line(result: Dict[str, object], backend_names: Sequence[str]) -> str:
    counts = result["counts"]  # type: ignore[assignment]
    util = result["utilization"]  # type: ignore[assignment]

    count_text = ", ".join(f"{name}:{counts[name]}" for name in backend_names)
    util_text = ", ".join(f"{name}:{util[name]:.3f}" for name in backend_names)

    return (
        f"{result['strategy']:<22} | "
        f"mean_wait={result['mean_wait']:.4f}, "
        f"p95_wait={result['p95_wait']:.4f}, "
        f"max_wait={result['max_wait']:.4f}, "
        f"imbalance_cv={result['imbalance_cv']:.4f} | "
        f"counts[{count_text}] | util[{util_text}]"
    )


def main() -> None:
    backends = [
        Backend(name="node-A", weight=5, capacity=6),
        Backend(name="node-B", weight=3, capacity=4),
        Backend(name="node-C", weight=2, capacity=3),
    ]

    arrivals, services = generate_workload(
        n_requests=5000,
        arrival_rate=8.0,
        service_log_mean=-1.8,
        service_log_sigma=0.9,
        seed=2026,
    )

    strategies = [
        "round_robin",
        "weighted_round_robin",
        "least_connections",
        "power_of_two_choices",
    ]

    print("Load Balancing Simulation (5000 requests)")
    print("-" * 120)
    backend_names = [b.name for b in backends]
    for idx, strategy in enumerate(strategies):
        result = simulate_strategy(
            strategy=strategy,
            backends=backends,
            arrivals=arrivals,
            service_times=services,
            lb_seed=9000 + idx,
        )
        print(format_result_line(result, backend_names))


if __name__ == "__main__":
    main()

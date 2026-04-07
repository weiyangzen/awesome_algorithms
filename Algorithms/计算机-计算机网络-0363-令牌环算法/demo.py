"""令牌环算法（Token Ring）最小可运行 MVP。

模型要点：
- 网络包含 N 个节点，节点按逻辑环连接。
- 环上只有一个令牌（token），令牌持有者才允许发送。
- 持有者每次访问最多发送 `token_quota_frames` 个帧，然后令牌交给下一个节点。

该脚本使用离散时间仿真展示令牌环如何实现无冲突的介质访问与公平轮转。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np


@dataclass
class TickRecord:
    tick: int
    token_holder: int
    arrivals: Tuple[int, ...]
    queue_before: Tuple[int, ...]
    sent_by_holder: int
    queue_after: Tuple[int, ...]
    total_backlog: int


@dataclass
class SimulationResult:
    records: List[TickRecord]
    sent_per_node: np.ndarray
    visits_per_node: np.ndarray
    backlog_final: int


class TokenRingSimulator:
    def __init__(
        self,
        node_count: int,
        initial_queues: Sequence[int],
        arrival_plan: Mapping[int, Sequence[int]],
        token_quota_frames: int = 1,
        max_ticks: int = 200,
        start_holder: int = 0,
    ) -> None:
        if node_count <= 1:
            raise ValueError("node_count must be >= 2")
        if len(initial_queues) != node_count:
            raise ValueError("initial_queues length must equal node_count")
        if token_quota_frames <= 0:
            raise ValueError("token_quota_frames must be positive")
        if max_ticks <= 0:
            raise ValueError("max_ticks must be positive")
        if not (0 <= start_holder < node_count):
            raise ValueError("start_holder out of range")

        self.node_count = node_count
        self.initial_queues = np.array(initial_queues, dtype=np.int64)
        if np.any(self.initial_queues < 0):
            raise ValueError("initial_queues must be non-negative")

        self.arrival_plan = self._normalize_arrival_plan(arrival_plan)
        self.token_quota_frames = token_quota_frames
        self.max_ticks = max_ticks
        self.start_holder = start_holder

    def _normalize_arrival_plan(
        self, arrival_plan: Mapping[int, Sequence[int]]
    ) -> Dict[int, np.ndarray]:
        normalized: Dict[int, np.ndarray] = {}
        for tick, arrivals in arrival_plan.items():
            if tick < 0:
                raise ValueError("arrival tick must be non-negative")
            arr = np.array(arrivals, dtype=np.int64)
            if arr.shape != (self.node_count,):
                raise ValueError(
                    f"arrival vector at tick={tick} must have length {self.node_count}"
                )
            if np.any(arr < 0):
                raise ValueError("arrival values must be non-negative")
            normalized[int(tick)] = arr
        return normalized

    def run(self) -> SimulationResult:
        queues = self.initial_queues.copy()
        sent_per_node = np.zeros(self.node_count, dtype=np.int64)
        visits_per_node = np.zeros(self.node_count, dtype=np.int64)
        records: List[TickRecord] = []

        token_holder = self.start_holder
        last_arrival_tick = max(self.arrival_plan.keys(), default=-1)

        for tick in range(self.max_ticks):
            arrivals = self.arrival_plan.get(
                tick, np.zeros(self.node_count, dtype=np.int64)
            )
            queues += arrivals

            queue_before = queues.copy()
            visits_per_node[token_holder] += 1

            send_now = int(min(self.token_quota_frames, queues[token_holder]))
            queues[token_holder] -= send_now
            sent_per_node[token_holder] += send_now

            total_backlog = int(np.sum(queues))
            records.append(
                TickRecord(
                    tick=tick,
                    token_holder=token_holder,
                    arrivals=tuple(int(x) for x in arrivals.tolist()),
                    queue_before=tuple(int(x) for x in queue_before.tolist()),
                    sent_by_holder=send_now,
                    queue_after=tuple(int(x) for x in queues.tolist()),
                    total_backlog=total_backlog,
                )
            )

            token_holder = (token_holder + 1) % self.node_count

            if total_backlog == 0 and tick >= last_arrival_tick:
                break

        return SimulationResult(
            records=records,
            sent_per_node=sent_per_node,
            visits_per_node=visits_per_node,
            backlog_final=int(np.sum(queues)),
        )


def total_demand_per_node(
    initial_queues: Sequence[int], arrival_plan: Mapping[int, Sequence[int]]
) -> np.ndarray:
    demand = np.array(initial_queues, dtype=np.int64)
    for arrivals in arrival_plan.values():
        demand += np.array(arrivals, dtype=np.int64)
    return demand


def jain_fairness(values: np.ndarray) -> float:
    x = values.astype(np.float64)
    numerator = np.sum(x) ** 2
    denominator = x.size * np.sum(x**2)
    if denominator == 0:
        return 1.0
    return float(numerator / denominator)


def print_report(result: SimulationResult, demand_per_node: np.ndarray) -> None:
    print("=== Token Ring Simulation Summary ===")
    print(f"ticks: {len(result.records)}")
    print(f"demand_per_node: {demand_per_node.tolist()}")
    print(f"sent_per_node: {result.sent_per_node.tolist()}")
    print(f"visits_per_node: {result.visits_per_node.tolist()}")
    print(f"backlog_final: {result.backlog_final}")
    print(f"jain_fairness(sent_per_node): {jain_fairness(result.sent_per_node):.4f}")

    print("\n--- First 15 ticks ---")
    print("tick holder sent backlog arrivals queue_before -> queue_after")
    for rec in result.records[:15]:
        print(
            f"{rec.tick:>4} {rec.token_holder:>6} {rec.sent_by_holder:>4} "
            f"{rec.total_backlog:>7} {list(rec.arrivals)!s:>18} "
            f"{list(rec.queue_before)} -> {list(rec.queue_after)}"
        )


def run_demo() -> None:
    node_count = 5
    initial_queues = [6, 2, 0, 4, 1]
    arrival_plan = {
        0: [0, 1, 0, 0, 0],
        1: [2, 0, 1, 0, 0],
        3: [0, 0, 0, 3, 1],
        5: [1, 2, 0, 0, 0],
        7: [0, 0, 2, 0, 0],
    }
    token_quota_frames = 2

    simulator = TokenRingSimulator(
        node_count=node_count,
        initial_queues=initial_queues,
        arrival_plan=arrival_plan,
        token_quota_frames=token_quota_frames,
        max_ticks=120,
        start_holder=0,
    )
    result = simulator.run()
    demand = total_demand_per_node(initial_queues, arrival_plan)

    if result.backlog_final != 0:
        raise AssertionError(
            f"simulation ended with non-empty queues: backlog={result.backlog_final}"
        )
    if not np.array_equal(result.sent_per_node, demand):
        raise AssertionError(
            "sent_per_node must match total demand under complete drain condition: "
            f"sent={result.sent_per_node.tolist()}, demand={demand.tolist()}"
        )

    visit_span = int(np.max(result.visits_per_node) - np.min(result.visits_per_node))
    if visit_span > 1:
        raise AssertionError(
            f"unexpected token visit imbalance: max-min={visit_span}, visits={result.visits_per_node.tolist()}"
        )

    # 只要持有者在该 tick 开始时有待发队列，且配额>0，就应至少发送 1 帧。
    for rec in result.records:
        if rec.queue_before[rec.token_holder] > 0 and rec.sent_by_holder == 0:
            raise AssertionError(
                "token holder had backlog but sent nothing: "
                f"tick={rec.tick}, holder={rec.token_holder}, queue={list(rec.queue_before)}"
            )

    print_report(result, demand)
    print("All checks passed.")


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()

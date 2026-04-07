"""CSMA/CA（Carrier Sense Multiple Access with Collision Avoidance）最小可运行 MVP。

模型说明：
- 多个站点共享同一无线信道，使用离散时隙进行仿真。
- 站点在信道空闲且满足 DIFS 后倒计时退避计数器，计数到 0 时尝试发送。
- 同一时隙若只有一个站点尝试则发送成功；若多个站点同时尝试则发生碰撞。
- 碰撞后使用二进制指数退避（窗口翻倍，受 cw_max 限制）。

该实现聚焦 MAC 层竞争机制，不建模物理层误码和隐藏终端。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np


@dataclass
class SlotRecord:
    slot: int
    arrivals: Tuple[int, ...]
    channel_busy: bool
    attempt_stations: Tuple[int, ...]
    successful_station: int
    collision: bool
    queues_after: Tuple[int, ...]


@dataclass
class SimulationResult:
    records: List[SlotRecord]
    success_per_station: np.ndarray
    attempts_per_station: np.ndarray
    collisions_per_station: np.ndarray
    backlog_final: int


class CSMACASimulator:
    def __init__(
        self,
        station_count: int,
        initial_queues: Sequence[int],
        arrival_plan: Mapping[int, Sequence[int]],
        cw_min: int = 3,
        cw_max: int = 31,
        difs_slots: int = 2,
        tx_slots: int = 3,
        ack_slots: int = 1,
        collision_slots: int = 3,
        max_slots: int = 400,
        seed: int = 7,
    ) -> None:
        if station_count <= 1:
            raise ValueError("station_count must be >= 2")
        if len(initial_queues) != station_count:
            raise ValueError("initial_queues length must equal station_count")
        if cw_min < 1:
            raise ValueError("cw_min must be >= 1")
        if cw_max < cw_min:
            raise ValueError("cw_max must be >= cw_min")
        if difs_slots < 1:
            raise ValueError("difs_slots must be >= 1")
        if tx_slots < 1 or ack_slots < 0 or collision_slots < 1:
            raise ValueError("invalid slot duration parameters")
        if max_slots <= 0:
            raise ValueError("max_slots must be positive")

        self.station_count = station_count
        self.initial_queues = np.array(initial_queues, dtype=np.int64)
        if np.any(self.initial_queues < 0):
            raise ValueError("initial_queues must be non-negative")

        self.arrival_plan = self._normalize_arrival_plan(arrival_plan)

        self.cw_min = cw_min
        self.cw_max = cw_max
        self.difs_slots = difs_slots
        self.tx_slots = tx_slots
        self.ack_slots = ack_slots
        self.collision_slots = collision_slots
        self.max_slots = max_slots
        self.seed = seed

    def _normalize_arrival_plan(
        self, arrival_plan: Mapping[int, Sequence[int]]
    ) -> Dict[int, np.ndarray]:
        normalized: Dict[int, np.ndarray] = {}
        for slot, arrivals in arrival_plan.items():
            if slot < 0:
                raise ValueError("arrival slot must be non-negative")
            arr = np.array(arrivals, dtype=np.int64)
            if arr.shape != (self.station_count,):
                raise ValueError(
                    f"arrival vector at slot={slot} must have length {self.station_count}"
                )
            if np.any(arr < 0):
                raise ValueError("arrival values must be non-negative")
            normalized[int(slot)] = arr
        return normalized

    def run(self) -> SimulationResult:
        rng = np.random.default_rng(self.seed)

        queues = self.initial_queues.copy()
        contention_windows = np.full(self.station_count, self.cw_min, dtype=np.int64)
        backoff_counter = np.full(self.station_count, -1, dtype=np.int64)

        success_per_station = np.zeros(self.station_count, dtype=np.int64)
        attempts_per_station = np.zeros(self.station_count, dtype=np.int64)
        collisions_per_station = np.zeros(self.station_count, dtype=np.int64)
        records: List[SlotRecord] = []

        channel_busy_remaining = 0
        idle_streak = 0
        last_arrival_slot = max(self.arrival_plan.keys(), default=-1)

        for slot in range(self.max_slots):
            arrivals = self.arrival_plan.get(
                slot, np.zeros(self.station_count, dtype=np.int64)
            )
            queues += arrivals

            # 有业务但未初始化退避的站点，抽取初始随机退避计数。
            for station in range(self.station_count):
                if queues[station] > 0 and backoff_counter[station] < 0:
                    backoff_counter[station] = int(
                        rng.integers(0, int(contention_windows[station]) + 1)
                    )

            attempt_stations: Tuple[int, ...] = tuple()
            successful_station = -1
            collision = False

            if channel_busy_remaining > 0:
                channel_busy = True
                channel_busy_remaining -= 1
                idle_streak = 0
            else:
                channel_busy = False
                idle_streak += 1

                can_countdown = idle_streak >= self.difs_slots
                if can_countdown:
                    for station in range(self.station_count):
                        if queues[station] > 0 and backoff_counter[station] > 0:
                            backoff_counter[station] -= 1

                    contenders = np.where(
                        (queues > 0) & (backoff_counter == 0)
                    )[0].astype(int)
                    attempt_stations = tuple(contenders.tolist())

                    if len(attempt_stations) == 1:
                        station = attempt_stations[0]
                        attempts_per_station[station] += 1
                        success_per_station[station] += 1
                        queues[station] -= 1

                        contention_windows[station] = self.cw_min
                        backoff_counter[station] = -1

                        channel_busy_remaining = self.tx_slots + self.ack_slots
                        idle_streak = 0
                        successful_station = station

                    elif len(attempt_stations) > 1:
                        collision = True
                        for station in attempt_stations:
                            attempts_per_station[station] += 1
                            collisions_per_station[station] += 1

                            next_cw = min(
                                self.cw_max,
                                int(contention_windows[station] * 2 + 1),
                            )
                            contention_windows[station] = next_cw
                            backoff_counter[station] = int(
                                rng.integers(0, int(next_cw) + 1)
                            )

                        channel_busy_remaining = self.collision_slots
                        idle_streak = 0

            for station in range(self.station_count):
                if queues[station] == 0:
                    backoff_counter[station] = -1
                    contention_windows[station] = self.cw_min

            records.append(
                SlotRecord(
                    slot=slot,
                    arrivals=tuple(int(x) for x in arrivals.tolist()),
                    channel_busy=channel_busy,
                    attempt_stations=attempt_stations,
                    successful_station=successful_station,
                    collision=collision,
                    queues_after=tuple(int(x) for x in queues.tolist()),
                )
            )

            total_backlog = int(np.sum(queues))
            if (
                total_backlog == 0
                and slot >= last_arrival_slot
                and channel_busy_remaining == 0
            ):
                break

        return SimulationResult(
            records=records,
            success_per_station=success_per_station,
            attempts_per_station=attempts_per_station,
            collisions_per_station=collisions_per_station,
            backlog_final=int(np.sum(queues)),
        )


def total_demand_per_station(
    initial_queues: Sequence[int], arrival_plan: Mapping[int, Sequence[int]]
) -> np.ndarray:
    demand = np.array(initial_queues, dtype=np.int64)
    for arrivals in arrival_plan.values():
        demand += np.array(arrivals, dtype=np.int64)
    return demand


def jain_fairness(values: np.ndarray) -> float:
    x = values.astype(np.float64)
    denominator = x.size * np.sum(x**2)
    if denominator == 0:
        return 1.0
    return float((np.sum(x) ** 2) / denominator)


def print_report(
    result: SimulationResult,
    demand_per_station: np.ndarray,
) -> None:
    total_success = int(np.sum(result.success_per_station))
    total_attempts = int(np.sum(result.attempts_per_station))
    total_collisions = int(np.sum(result.collisions_per_station))
    collision_prob = (total_collisions / total_attempts) if total_attempts > 0 else 0.0

    print("=== CSMA/CA Simulation Summary ===")
    print(f"slots: {len(result.records)}")
    print(f"demand_per_station: {demand_per_station.tolist()}")
    print(f"success_per_station: {result.success_per_station.tolist()}")
    print(f"attempts_per_station: {result.attempts_per_station.tolist()}")
    print(f"collisions_per_station: {result.collisions_per_station.tolist()}")
    print(f"total_success: {total_success}")
    print(f"total_attempts: {total_attempts}")
    print(f"collision_probability: {collision_prob:.4f}")
    print(f"throughput(success/slot): {total_success / max(1, len(result.records)):.4f}")
    print(
        "jain_fairness(success_per_station): "
        f"{jain_fairness(result.success_per_station):.4f}"
    )
    print(f"backlog_final: {result.backlog_final}")

    print("\n--- First 20 slots ---")
    print("slot busy attempts success collision arrivals queues_after")
    for rec in result.records[:20]:
        print(
            f"{rec.slot:>4} {str(rec.channel_busy):>4} "
            f"{list(rec.attempt_stations)!s:>10} {rec.successful_station:>7} "
            f"{str(rec.collision):>9} {list(rec.arrivals)!s:>12} "
            f"{list(rec.queues_after)}"
        )


def run_demo() -> None:
    station_count = 5
    initial_queues = [6, 5, 4, 5, 3]
    arrival_plan = {
        8: [1, 0, 1, 0, 0],
        14: [0, 2, 0, 1, 0],
        21: [1, 1, 0, 0, 2],
    }

    simulator = CSMACASimulator(
        station_count=station_count,
        initial_queues=initial_queues,
        arrival_plan=arrival_plan,
        cw_min=3,
        cw_max=31,
        difs_slots=2,
        tx_slots=3,
        ack_slots=1,
        collision_slots=3,
        max_slots=400,
        seed=11,
    )
    result = simulator.run()
    demand = total_demand_per_station(initial_queues, arrival_plan)

    total_success = int(np.sum(result.success_per_station))
    total_demand = int(np.sum(demand))
    total_collisions = int(np.sum(result.collisions_per_station))

    if result.backlog_final != 0:
        raise AssertionError(
            f"simulation ended with backlog: {result.backlog_final}"
        )
    if total_success != total_demand:
        raise AssertionError(
            f"successful transmissions must equal total demand: success={total_success}, demand={total_demand}"
        )
    if np.any(result.success_per_station > demand):
        raise AssertionError(
            "per-station success must not exceed per-station demand"
        )
    if total_collisions <= 0:
        raise AssertionError("expected at least one collision in this scenario")

    print_report(result, demand)
    print("All checks passed.")


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()

"""CSMA/CD（Carrier Sense Multiple Access with Collision Detection）最小可运行 MVP。

模型说明：
- 多个站点共享同一有线总线介质。
- 站点在信道空闲时进行发送尝试（1-persistent 抽象）。
- 若同一争用时隙有多站尝试，则检测到碰撞，立即中止并发送 jam。
- 碰撞后采用二进制指数退避（Binary Exponential Backoff, BEB）。

该实现聚焦 MAC 层竞争控制逻辑，不模拟物理层编码与误码细节。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class SlotRecord:
    slot: int
    arrivals: Tuple[int, ...]
    channel_busy: bool
    event: str
    attempt_stations: Tuple[int, ...]
    successful_station: int
    collision: bool
    dropped_stations: Tuple[int, ...]
    queues_after: Tuple[int, ...]


@dataclass
class SimulationResult:
    records: List[SlotRecord]
    attempts_per_station: np.ndarray
    success_per_station: np.ndarray
    collisions_per_station: np.ndarray
    drops_per_station: np.ndarray
    backlog_final: int


class CSMACDSimulator:
    def __init__(
        self,
        station_count: int,
        initial_queues: Sequence[int],
        arrival_plan: Mapping[int, Sequence[int]],
        frame_slots: int = 6,
        jam_slots: int = 1,
        max_retries: int = 16,
        max_slots: int = 500,
        seed: int = 17,
    ) -> None:
        if station_count <= 1:
            raise ValueError("station_count must be >= 2")
        if len(initial_queues) != station_count:
            raise ValueError("initial_queues length must equal station_count")
        if frame_slots < 1:
            raise ValueError("frame_slots must be >= 1")
        if jam_slots < 1:
            raise ValueError("jam_slots must be >= 1")
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        if max_slots <= 0:
            raise ValueError("max_slots must be positive")

        self.station_count = station_count
        self.initial_queues = np.array(initial_queues, dtype=np.int64)
        if np.any(self.initial_queues < 0):
            raise ValueError("initial_queues must be non-negative")

        self.arrival_plan = self._normalize_arrival_plan(arrival_plan)
        self.frame_slots = frame_slots
        self.jam_slots = jam_slots
        self.max_retries = max_retries
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

    def _sample_backoff(self, rng: np.random.Generator, collision_count: int) -> int:
        """BEB: after k-th collision, random in [0, 2^min(k,10)-1]."""
        exponent = min(int(collision_count), 10)
        window = 1 << exponent
        return int(rng.integers(0, window))

    def run(self) -> SimulationResult:
        rng = np.random.default_rng(self.seed)

        queues = self.initial_queues.copy()
        backoff_counter = np.full(self.station_count, -1, dtype=np.int64)
        collision_level = np.zeros(self.station_count, dtype=np.int64)

        attempts = np.zeros(self.station_count, dtype=np.int64)
        success = np.zeros(self.station_count, dtype=np.int64)
        collisions = np.zeros(self.station_count, dtype=np.int64)
        drops = np.zeros(self.station_count, dtype=np.int64)

        records: List[SlotRecord] = []

        medium_busy_remaining = 0
        last_arrival_slot = max(self.arrival_plan.keys(), default=-1)

        for slot in range(self.max_slots):
            arrivals = self.arrival_plan.get(
                slot, np.zeros(self.station_count, dtype=np.int64)
            )
            queues += arrivals

            # 激活发送候选：首次发送立即尝试；碰撞后的帧按退避计数。
            for station in range(self.station_count):
                if queues[station] <= 0 or backoff_counter[station] >= 0:
                    continue
                if collision_level[station] == 0:
                    backoff_counter[station] = 0
                else:
                    backoff_counter[station] = self._sample_backoff(
                        rng, int(collision_level[station])
                    )

            channel_busy = medium_busy_remaining > 0
            attempt_stations: Tuple[int, ...] = tuple()
            successful_station = -1
            collision = False
            dropped_stations: Tuple[int, ...] = tuple()
            event = "idle"

            if channel_busy:
                medium_busy_remaining -= 1
                event = "busy"
            else:
                # 仅在空闲信道下倒计时。
                for station in range(self.station_count):
                    if queues[station] > 0 and backoff_counter[station] > 0:
                        backoff_counter[station] -= 1

                contenders = np.where((queues > 0) & (backoff_counter == 0))[0]
                attempt_stations = tuple(int(x) for x in contenders.tolist())

                if len(attempt_stations) == 1:
                    station = attempt_stations[0]
                    attempts[station] += 1
                    success[station] += 1
                    queues[station] -= 1

                    collision_level[station] = 0
                    backoff_counter[station] = -1

                    medium_busy_remaining = self.frame_slots - 1
                    successful_station = station
                    event = "success"

                elif len(attempt_stations) > 1:
                    collision = True
                    event = "collision"
                    medium_busy_remaining = self.jam_slots - 1

                    dropped: List[int] = []
                    for station in attempt_stations:
                        attempts[station] += 1
                        collisions[station] += 1
                        collision_level[station] += 1

                        if collision_level[station] > self.max_retries:
                            # 该队首帧超过重传上限，直接丢弃。
                            drops[station] += 1
                            queues[station] -= 1
                            collision_level[station] = 0
                            backoff_counter[station] = -1
                            dropped.append(station)
                        else:
                            backoff_counter[station] = self._sample_backoff(
                                rng, int(collision_level[station])
                            )

                    dropped_stations = tuple(dropped)

            # 若队列为空，清理该站状态。
            for station in range(self.station_count):
                if queues[station] == 0:
                    backoff_counter[station] = -1
                    collision_level[station] = 0

            records.append(
                SlotRecord(
                    slot=slot,
                    arrivals=tuple(int(x) for x in arrivals.tolist()),
                    channel_busy=channel_busy,
                    event=event,
                    attempt_stations=attempt_stations,
                    successful_station=successful_station,
                    collision=collision,
                    dropped_stations=dropped_stations,
                    queues_after=tuple(int(x) for x in queues.tolist()),
                )
            )

            backlog = int(np.sum(queues))
            if backlog < 0:
                raise AssertionError("backlog must stay non-negative")

            if backlog == 0 and slot >= last_arrival_slot and medium_busy_remaining == 0:
                break

        return SimulationResult(
            records=records,
            attempts_per_station=attempts,
            success_per_station=success,
            collisions_per_station=collisions,
            drops_per_station=drops,
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


def print_report(result: SimulationResult, demand_per_station: np.ndarray) -> None:
    total_attempts = int(np.sum(result.attempts_per_station))
    total_success = int(np.sum(result.success_per_station))
    total_collisions = int(np.sum(result.collisions_per_station))
    total_drops = int(np.sum(result.drops_per_station))
    slot_count = len(result.records)

    collision_probability = (
        (total_collisions / total_attempts) if total_attempts > 0 else 0.0
    )

    print("=== CSMA/CD Simulation Summary ===")
    print(f"slots: {slot_count}")
    print(f"demand_per_station: {demand_per_station.tolist()}")
    print(f"attempts_per_station: {result.attempts_per_station.tolist()}")
    print(f"success_per_station: {result.success_per_station.tolist()}")
    print(f"collisions_per_station: {result.collisions_per_station.tolist()}")
    print(f"drops_per_station: {result.drops_per_station.tolist()}")
    print(f"total_attempts: {total_attempts}")
    print(f"total_success: {total_success}")
    print(f"total_collisions: {total_collisions}")
    print(f"total_drops: {total_drops}")
    print(f"collision_probability: {collision_probability:.4f}")
    print(f"throughput(success/slot): {total_success / max(1, slot_count):.4f}")
    print(
        "jain_fairness(success_per_station): "
        f"{jain_fairness(result.success_per_station):.4f}"
    )
    print(f"backlog_final: {result.backlog_final}")

    table_rows = []
    for rec in result.records[:25]:
        table_rows.append(
            {
                "slot": rec.slot,
                "busy": rec.channel_busy,
                "event": rec.event,
                "attempts": list(rec.attempt_stations),
                "success": rec.successful_station,
                "collision": rec.collision,
                "dropped": list(rec.dropped_stations),
                "arrivals": list(rec.arrivals),
                "queues_after": list(rec.queues_after),
            }
        )

    print("\n--- First 25 slots ---")
    print(pd.DataFrame(table_rows).to_string(index=False))


def run_demo() -> None:
    station_count = 6
    initial_queues = [5, 5, 4, 4, 3, 3]
    arrival_plan = {
        8: [1, 0, 1, 0, 0, 1],
        17: [0, 1, 0, 1, 1, 0],
        26: [1, 1, 0, 0, 1, 0],
    }

    simulator = CSMACDSimulator(
        station_count=station_count,
        initial_queues=initial_queues,
        arrival_plan=arrival_plan,
        frame_slots=6,
        jam_slots=1,
        max_retries=16,
        max_slots=500,
        seed=23,
    )
    result = simulator.run()
    demand = total_demand_per_station(initial_queues, arrival_plan)

    total_demand = int(np.sum(demand))
    total_success = int(np.sum(result.success_per_station))
    total_drops = int(np.sum(result.drops_per_station))
    total_collisions = int(np.sum(result.collisions_per_station))

    if result.backlog_final != 0:
        raise AssertionError(f"simulation ended with backlog: {result.backlog_final}")
    if total_success + total_drops != total_demand:
        raise AssertionError(
            "flow conservation failed: "
            f"success({total_success}) + drops({total_drops}) != demand({total_demand})"
        )
    if total_collisions <= 0:
        raise AssertionError("expected at least one collision in this scenario")
    if total_drops != 0:
        raise AssertionError(
            f"expected no frame drops in this scenario, got drops={total_drops}"
        )
    if np.any(result.success_per_station > demand):
        raise AssertionError("per-station success cannot exceed per-station demand")

    print_report(result, demand)
    print("All checks passed.")


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()

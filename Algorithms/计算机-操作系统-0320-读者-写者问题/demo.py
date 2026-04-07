"""读者-写者问题：公平读写锁（无饥饿）最小可运行示例。"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np


class FairRWLock:
    """公平读写锁（turnstile + room_empty + read_count）。

    算法要点：
    - `turnstile` 充当队列闸门，写者到来后会阻止后续新读者直接插队；
    - `room_empty` 保证写临界区互斥，并与首/末读者协作实现共享读；
    - `read_count` 由 `read_mutex` 保护。
    """

    def __init__(self) -> None:
        self.turnstile = threading.Semaphore(1)
        self.room_empty = threading.Semaphore(1)
        self.read_mutex = threading.Lock()
        self.read_count = 0

    def reader_acquire(self) -> None:
        # 经过 turnstile，避免在写者排队时新读者无限插队。
        self.turnstile.acquire()
        self.turnstile.release()

        with self.read_mutex:
            self.read_count += 1
            if self.read_count == 1:
                self.room_empty.acquire()

    def reader_release(self) -> None:
        with self.read_mutex:
            self.read_count -= 1
            if self.read_count == 0:
                self.room_empty.release()

    def writer_acquire(self) -> None:
        self.turnstile.acquire()
        self.room_empty.acquire()

    def writer_release(self) -> None:
        self.room_empty.release()
        self.turnstile.release()


@dataclass
class SharedState:
    value: int = 0
    active_readers: int = 0
    active_writers: int = 0


@dataclass
class SimulationResult:
    final_value: int
    expected_value: int
    reader_wait_stats: dict[str, float]
    writer_wait_stats: dict[str, float]
    max_active_readers: int
    max_active_writers: int
    conflict_count: int
    total_events: int


def summarize_wait_times(wait_times: list[float]) -> dict[str, float]:
    arr = np.array(wait_times, dtype=float)
    if arr.size == 0:
        return {"mean_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    return {
        "mean_ms": float(arr.mean() * 1000.0),
        "p95_ms": float(np.percentile(arr, 95) * 1000.0),
        "max_ms": float(arr.max() * 1000.0),
    }


def run_simulation(
    num_readers: int = 6,
    num_writers: int = 3,
    reader_rounds: int = 24,
    writer_rounds: int = 12,
    seed: int = 20260407,
) -> SimulationResult:
    rw_lock = FairRWLock()
    shared = SharedState()
    state_lock = threading.Lock()

    log_lock = threading.Lock()
    events: list[dict[str, Any]] = []

    reader_waits: list[float] = []
    writer_waits: list[float] = []

    def log_event(event: dict[str, Any]) -> None:
        with log_lock:
            events.append(event)

    def reader_worker(reader_id: int) -> None:
        rng = random.Random(seed + 1000 + reader_id)
        actor = f"R{reader_id}"

        for step in range(reader_rounds):
            time.sleep(rng.uniform(0.001, 0.004))
            request_ts = time.perf_counter()
            rw_lock.reader_acquire()
            acquire_ts = time.perf_counter()
            wait_s = acquire_ts - request_ts
            reader_waits.append(wait_s)

            with state_lock:
                # 读临界区内不允许存在写者。
                assert shared.active_writers == 0, "Reader entered while writer active"
                shared.active_readers += 1
                snapshot = shared.value
                active_r = shared.active_readers
                active_w = shared.active_writers

            log_event(
                {
                    "t": acquire_ts,
                    "actor": actor,
                    "kind": "reader_enter",
                    "step": step,
                    "wait_ms": wait_s * 1000.0,
                    "value": snapshot,
                    "active_readers": active_r,
                    "active_writers": active_w,
                }
            )

            time.sleep(rng.uniform(0.0008, 0.003))

            with state_lock:
                shared.active_readers -= 1
                active_r = shared.active_readers
                active_w = shared.active_writers

            rw_lock.reader_release()
            log_event(
                {
                    "t": time.perf_counter(),
                    "actor": actor,
                    "kind": "reader_exit",
                    "step": step,
                    "value": snapshot,
                    "active_readers": active_r,
                    "active_writers": active_w,
                }
            )

    def writer_worker(writer_id: int) -> None:
        rng = random.Random(seed + 2000 + writer_id)
        actor = f"W{writer_id}"

        for step in range(writer_rounds):
            time.sleep(rng.uniform(0.0012, 0.0045))
            request_ts = time.perf_counter()
            rw_lock.writer_acquire()
            acquire_ts = time.perf_counter()
            wait_s = acquire_ts - request_ts
            writer_waits.append(wait_s)

            with state_lock:
                # 写临界区要求独占：无读者、无其他写者。
                assert shared.active_readers == 0, "Writer entered while readers active"
                assert shared.active_writers == 0, "Two writers entered at once"
                shared.active_writers += 1
                before = shared.value
                active_r = shared.active_readers
                active_w = shared.active_writers

            log_event(
                {
                    "t": acquire_ts,
                    "actor": actor,
                    "kind": "writer_enter",
                    "step": step,
                    "wait_ms": wait_s * 1000.0,
                    "value_before": before,
                    "active_readers": active_r,
                    "active_writers": active_w,
                }
            )

            time.sleep(rng.uniform(0.001, 0.0035))

            with state_lock:
                shared.value += 1
                after = shared.value
                shared.active_writers -= 1
                active_r = shared.active_readers
                active_w = shared.active_writers

            rw_lock.writer_release()
            log_event(
                {
                    "t": time.perf_counter(),
                    "actor": actor,
                    "kind": "writer_exit",
                    "step": step,
                    "value_after": after,
                    "active_readers": active_r,
                    "active_writers": active_w,
                }
            )

    threads: list[threading.Thread] = []
    for i in range(num_readers):
        threads.append(threading.Thread(target=reader_worker, args=(i,), name=f"reader-{i}"))
    for i in range(num_writers):
        threads.append(threading.Thread(target=writer_worker, args=(i,), name=f"writer-{i}"))

    random.Random(seed).shuffle(threads)
    start_t = time.perf_counter()
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=20.0)

    alive_threads = [th.name for th in threads if th.is_alive()]
    assert not alive_threads, f"Threads did not finish: {alive_threads}"

    elapsed_s = time.perf_counter() - start_t
    expected_value = num_writers * writer_rounds

    max_active_readers = 0
    max_active_writers = 0
    conflict_count = 0
    for ev in events:
        ar = int(ev.get("active_readers", 0))
        aw = int(ev.get("active_writers", 0))
        max_active_readers = max(max_active_readers, ar)
        max_active_writers = max(max_active_writers, aw)
        if ar > 0 and aw > 0:
            conflict_count += 1

    reader_stats = summarize_wait_times(reader_waits)
    writer_stats = summarize_wait_times(writer_waits)

    # 不变量断言。
    assert shared.value == expected_value, (
        f"Final value mismatch: got {shared.value}, expected {expected_value}"
    )
    assert max_active_writers <= 1, f"Writer exclusivity violated: {max_active_writers}"
    assert conflict_count == 0, f"Reader/writer overlap detected: {conflict_count}"

    # 基础“无饥饿”信号：写者等待应当有上界（在本规模和参数下）。
    assert writer_stats["max_ms"] < 4000.0, (
        f"Writer waited too long, possible starvation: {writer_stats['max_ms']:.2f} ms"
    )

    print("=== Readers-Writers Fair Lock Simulation ===")
    print(f"Elapsed: {elapsed_s * 1000.0:.1f} ms")
    print(f"Final shared value: {shared.value} (expected {expected_value})")
    print(
        "Reader waits (ms): "
        f"mean={reader_stats['mean_ms']:.3f}, "
        f"p95={reader_stats['p95_ms']:.3f}, "
        f"max={reader_stats['max_ms']:.3f}"
    )
    print(
        "Writer waits (ms): "
        f"mean={writer_stats['mean_ms']:.3f}, "
        f"p95={writer_stats['p95_ms']:.3f}, "
        f"max={writer_stats['max_ms']:.3f}"
    )
    print(f"Max active readers: {max_active_readers}")
    print(f"Max active writers: {max_active_writers}")
    print(f"Conflict events (reader & writer both active): {conflict_count}")
    print(f"Total events recorded: {len(events)}")
    print("All assertions passed.")

    return SimulationResult(
        final_value=shared.value,
        expected_value=expected_value,
        reader_wait_stats=reader_stats,
        writer_wait_stats=writer_stats,
        max_active_readers=max_active_readers,
        max_active_writers=max_active_writers,
        conflict_count=conflict_count,
        total_events=len(events),
    )


def main() -> None:
    run_simulation()


if __name__ == "__main__":
    main()

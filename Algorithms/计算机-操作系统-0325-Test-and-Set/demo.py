"""Test-and-Set MVP: using a hand-written TAS primitive to build a spin lock."""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd


class TestAndSetFlag:
    """Atomic boolean flag exposing a Test-and-Set primitive.

    test_and_set semantics:
    - Atomically return the old value.
    - Atomically set the flag to True.

    This implementation uses a tiny mutex to emulate one indivisible hardware TAS
    instruction in Python for educational correctness checks.
    """

    def __init__(self, initial: bool = False) -> None:
        self._value = bool(initial)
        self._guard = threading.Lock()

    def test_and_set(self) -> bool:
        with self._guard:
            old = self._value
            self._value = True
            return old

    def clear(self) -> None:
        with self._guard:
            self._value = False

    def load(self) -> bool:
        with self._guard:
            return self._value


class TASSpinLock:
    """Spin lock built from Test-and-Set."""

    def __init__(self) -> None:
        self._flag = TestAndSetFlag(False)

    def acquire(self, rng: random.Random) -> tuple[int, int]:
        """Acquire lock and return (tas_calls, failed_spins) for this acquisition."""
        tas_calls = 0
        failed_spins = 0

        while True:
            tas_calls += 1
            old = self._flag.test_and_set()
            if not old:
                return tas_calls, failed_spins

            failed_spins += 1
            backoff_cap = min(0.0008, 1e-6 * (2 ** min(failed_spins, 10)))
            time.sleep(rng.uniform(0.0, backoff_cap))

    def release(self) -> None:
        self._flag.clear()

    def is_locked(self) -> bool:
        return self._flag.load()


@dataclass
class SimulationReport:
    expected_entries: int
    shared_counter: int
    elapsed_ms: float
    total_tas_calls: int
    total_failed_spins: int
    total_acquires: int
    contention_ratio: float
    mean_failed_spins: float
    p95_failed_spins: float
    max_failed_spins: int
    max_threads_in_cs: int


def run_simulation(
    num_threads: int = 8,
    entries_per_thread: int = 240,
    seed: int = 20260407,
    critical_section_max_s: float = 0.00018,
) -> SimulationReport:
    if num_threads <= 0:
        raise ValueError("num_threads must be > 0")
    if entries_per_thread <= 0:
        raise ValueError("entries_per_thread must be > 0")

    lock = TASSpinLock()

    shared_counter = 0
    threads_in_cs = 0
    max_threads_in_cs = 0
    shared_guard = threading.Lock()

    per_thread_tas_calls = [0] * num_threads
    per_thread_failed_spins = [0] * num_threads
    per_thread_acquires = [0] * num_threads
    wait_time_ns_by_thread = [0] * num_threads
    failed_spins_each_acquire: list[int] = []

    metrics_guard = threading.Lock()

    def worker(tid: int) -> None:
        nonlocal shared_counter, threads_in_cs, max_threads_in_cs

        rng = random.Random(seed + tid * 313)
        local_tas_calls = 0
        local_failed_spins = 0
        local_acquires = 0
        local_wait_ns = 0
        local_spins_each_acquire: list[int] = []

        for _ in range(entries_per_thread):
            t0 = time.perf_counter_ns()
            tas_calls, failed_spins = lock.acquire(rng)
            t1 = time.perf_counter_ns()

            local_tas_calls += tas_calls
            local_failed_spins += failed_spins
            local_acquires += 1
            local_wait_ns += t1 - t0
            local_spins_each_acquire.append(failed_spins)

            with shared_guard:
                threads_in_cs += 1
                if threads_in_cs > max_threads_in_cs:
                    max_threads_in_cs = threads_in_cs
                if threads_in_cs != 1:
                    raise RuntimeError("Mutual exclusion violated in critical section")

                shared_counter += 1
                # Keep CS non-empty for a tiny random window to amplify contention.
                time.sleep(rng.uniform(0.0, critical_section_max_s))

                threads_in_cs -= 1

            lock.release()

        with metrics_guard:
            per_thread_tas_calls[tid] = local_tas_calls
            per_thread_failed_spins[tid] = local_failed_spins
            per_thread_acquires[tid] = local_acquires
            wait_time_ns_by_thread[tid] = local_wait_ns
            failed_spins_each_acquire.extend(local_spins_each_acquire)

    threads = [
        threading.Thread(target=worker, args=(tid,), name=f"tas-worker-{tid}")
        for tid in range(num_threads)
    ]

    random.Random(seed).shuffle(threads)

    start = time.perf_counter()
    for th in threads:
        th.start()

    for th in threads:
        th.join(timeout=30.0)

    alive = [th.name for th in threads if th.is_alive()]
    assert not alive, f"Threads did not finish: {alive}"

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    expected_entries = num_threads * entries_per_thread
    total_tas_calls = int(np.sum(per_thread_tas_calls))
    total_failed_spins = int(np.sum(per_thread_failed_spins))
    total_acquires = int(np.sum(per_thread_acquires))

    spins_arr = np.array(failed_spins_each_acquire, dtype=np.int64)
    mean_failed_spins = float(spins_arr.mean()) if spins_arr.size else 0.0
    p95_failed_spins = (
        float(np.percentile(spins_arr, 95, method="nearest")) if spins_arr.size else 0.0
    )
    max_failed_spins = int(spins_arr.max()) if spins_arr.size else 0

    contention_ratio = (
        float(total_failed_spins / total_tas_calls) if total_tas_calls > 0 else 0.0
    )

    # Correctness and accounting assertions.
    assert shared_counter == expected_entries, (
        f"Counter mismatch: got {shared_counter}, expected {expected_entries}"
    )
    assert total_acquires == expected_entries, (
        f"Acquire mismatch: got {total_acquires}, expected {expected_entries}"
    )
    assert total_tas_calls == total_failed_spins + total_acquires, (
        "TAS accounting mismatch: tas_calls != failed_spins + acquires"
    )
    assert max_threads_in_cs == 1, (
        f"Mutual exclusion mismatch: max threads in CS = {max_threads_in_cs}"
    )
    assert len(failed_spins_each_acquire) == expected_entries, (
        "Missing per-acquire spin records"
    )
    assert lock.is_locked() is False, "Lock still set after all threads completed"

    stats_df = pd.DataFrame(
        {
            "thread": list(range(num_threads)),
            "acquires": per_thread_acquires,
            "tas_calls": per_thread_tas_calls,
            "failed_spins": per_thread_failed_spins,
            "avg_wait_us": [
                (wait_time_ns_by_thread[i] / per_thread_acquires[i]) / 1000.0
                for i in range(num_threads)
            ],
        }
    )

    print("=== Test-and-Set Spin Lock Simulation ===")
    print(f"Threads: {num_threads}, entries per thread: {entries_per_thread}")
    print(f"Elapsed: {elapsed_ms:.3f} ms")
    print(f"Final shared counter: {shared_counter}")
    print(f"Expected entries: {expected_entries}")
    print(f"Total TAS calls: {total_tas_calls}")
    print(f"Total failed spins: {total_failed_spins}")
    print(f"Total lock acquires: {total_acquires}")
    print(f"Contention ratio (failed/tas_calls): {contention_ratio:.4f}")
    print(
        "Failed spins per acquire: "
        f"mean={mean_failed_spins:.3f}, p95={p95_failed_spins:.3f}, max={max_failed_spins}"
    )
    print(f"Max threads observed in critical section: {max_threads_in_cs}")
    print("\nPer-thread summary:")
    print(stats_df.to_string(index=False))
    print("All assertions passed.")

    return SimulationReport(
        expected_entries=expected_entries,
        shared_counter=shared_counter,
        elapsed_ms=elapsed_ms,
        total_tas_calls=total_tas_calls,
        total_failed_spins=total_failed_spins,
        total_acquires=total_acquires,
        contention_ratio=contention_ratio,
        mean_failed_spins=mean_failed_spins,
        p95_failed_spins=p95_failed_spins,
        max_failed_spins=max_failed_spins,
        max_threads_in_cs=max_threads_in_cs,
    )


def main() -> None:
    run_simulation()


if __name__ == "__main__":
    main()

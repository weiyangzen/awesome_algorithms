"""Compare-and-Swap (CAS) MVP：基于线程竞争的最小可运行示例。"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass

import numpy as np


class AtomicCASInteger:
    """用锁模拟“单条原子 CAS 指令”语义。

    说明：
    - 真正硬件 CAS 由 CPU 指令保证原子性；
    - Python 标准库没有直接暴露通用原子 CAS 指令；
    - 这里用短临界区锁来模拟“比较并交换”的线性化点，便于教学与验证。
    """

    def __init__(self, initial: int = 0) -> None:
        self._value = int(initial)
        self._lock = threading.Lock()

    def load(self) -> int:
        with self._lock:
            return self._value

    def compare_and_swap(self, expected: int, new_value: int) -> tuple[bool, int]:
        """尝试把值从 expected 改为 new_value。

        返回：
        - success: 是否交换成功
        - observed: CAS 时刻观察到的旧值
        """
        with self._lock:
            observed = self._value
            if observed == expected:
                self._value = new_value
                return True, observed
            return False, observed


def retry_backoff_sleep(rng: random.Random, retries: int) -> None:
    """指数退避，缓解高冲突下的活锁倾向。"""
    cap = min(0.0008, (2 ** min(retries, 10)) * 1e-6)
    time.sleep(rng.uniform(0.0, cap))


@dataclass
class SimulationResult:
    final_value: int
    expected_value: int
    elapsed_ms: float
    total_attempts: int
    total_successes: int
    total_failures: int
    contention_ratio: float
    mean_retries: float
    p95_retries: float
    max_retries: int
    attempts_by_thread: list[int]
    failures_by_thread: list[int]


def run_simulation(
    num_threads: int = 8,
    ops_per_thread: int = 300,
    seed: int = 20260407,
    read_pause_prob: float = 0.55,
    read_pause_max_s: float = 0.0005,
) -> SimulationResult:
    if num_threads <= 0:
        raise ValueError("num_threads must be > 0")
    if ops_per_thread <= 0:
        raise ValueError("ops_per_thread must be > 0")

    counter = AtomicCASInteger(initial=0)

    retries_per_op: list[int] = []
    attempts_by_thread = [0 for _ in range(num_threads)]
    failures_by_thread = [0 for _ in range(num_threads)]
    successes_by_thread = [0 for _ in range(num_threads)]

    metrics_lock = threading.Lock()

    def worker(tid: int) -> None:
        rng = random.Random(seed + tid * 997)

        local_attempts = 0
        local_failures = 0
        local_successes = 0
        local_retries: list[int] = []

        for _ in range(ops_per_thread):
            retries = 0

            while True:
                expected = counter.load()

                # 主动制造读-改-写窗口，使线程更容易竞争同一 expected 值。
                if rng.random() < read_pause_prob:
                    time.sleep(rng.uniform(0.0, read_pause_max_s))

                ok, _ = counter.compare_and_swap(expected, expected + 1)
                local_attempts += 1

                if ok:
                    local_successes += 1
                    local_retries.append(retries)
                    break

                local_failures += 1
                retries += 1
                retry_backoff_sleep(rng, retries)

                if retries > 100_000:
                    raise RuntimeError("CAS retries exceeded safety threshold")

        with metrics_lock:
            attempts_by_thread[tid] = local_attempts
            failures_by_thread[tid] = local_failures
            successes_by_thread[tid] = local_successes
            retries_per_op.extend(local_retries)

    threads = [
        threading.Thread(target=worker, args=(tid,), name=f"cas-worker-{tid}")
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

    expected_value = num_threads * ops_per_thread
    final_value = counter.load()

    total_attempts = int(np.sum(attempts_by_thread))
    total_failures = int(np.sum(failures_by_thread))
    total_successes = int(np.sum(successes_by_thread))

    retries_arr = np.array(retries_per_op, dtype=np.int64)
    if retries_arr.size == 0:
        mean_retries = 0.0
        p95_retries = 0.0
        max_retries = 0
    else:
        mean_retries = float(retries_arr.mean())
        p95_retries = float(np.percentile(retries_arr, 95))
        max_retries = int(retries_arr.max())

    contention_ratio = (
        float(total_failures / total_attempts) if total_attempts > 0 else 0.0
    )

    # 正确性与一致性断言
    assert final_value == expected_value, (
        f"Final value mismatch: got {final_value}, expected {expected_value}"
    )
    assert len(retries_per_op) == expected_value, (
        f"Retry record count mismatch: got {len(retries_per_op)}, expected {expected_value}"
    )
    assert total_successes == expected_value, (
        f"Success count mismatch: got {total_successes}, expected {expected_value}"
    )
    assert total_attempts == total_successes + total_failures, (
        "Attempt accounting mismatch: attempts != successes + failures"
    )
    assert max_retries < 5000, f"Max retries too large: {max_retries}"

    print("=== Compare-and-Swap Contention Simulation ===")
    print(f"Threads: {num_threads}, operations per thread: {ops_per_thread}")
    print(f"Elapsed: {elapsed_ms:.3f} ms")
    print(f"Final counter value: {final_value}")
    print(f"Expected counter value: {expected_value}")
    print(f"Total CAS attempts: {total_attempts}")
    print(f"Total CAS successes: {total_successes}")
    print(f"Total CAS failures: {total_failures}")
    print(f"Contention ratio (failures/attempts): {contention_ratio:.4f}")
    print(
        "Retries per successful increment: "
        f"mean={mean_retries:.3f}, p95={p95_retries:.3f}, max={max_retries}"
    )
    print("Attempts by thread:", attempts_by_thread)
    print("Failures by thread:", failures_by_thread)
    print("All assertions passed.")

    return SimulationResult(
        final_value=final_value,
        expected_value=expected_value,
        elapsed_ms=elapsed_ms,
        total_attempts=total_attempts,
        total_successes=total_successes,
        total_failures=total_failures,
        contention_ratio=contention_ratio,
        mean_retries=mean_retries,
        p95_retries=p95_retries,
        max_retries=max_retries,
        attempts_by_thread=attempts_by_thread,
        failures_by_thread=failures_by_thread,
    )


def main() -> None:
    run_simulation()


if __name__ == "__main__":
    main()

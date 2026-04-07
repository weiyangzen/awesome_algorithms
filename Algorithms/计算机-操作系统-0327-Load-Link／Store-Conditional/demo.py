"""Load-Link/Store-Conditional (LL/SC) MVP：并发竞争下的最小可运行示例。"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass

import numpy as np


class AtomicLLSCInteger:
    """用锁模拟 LL/SC 在单内存位置上的原子语义。

    说明：
    - `load_link(tid)` 读取当前值并为线程 tid 建立 reservation；
    - `store_conditional(tid, token, expected, new)` 仅在 reservation 仍有效时成功；
    - 任意成功写入都会使版本号递增并失效所有 reservation（模拟硬件失效语义）。
    """

    def __init__(self, initial: int = 0) -> None:
        self._value = int(initial)
        self._version = 0
        self._lock = threading.Lock()
        self._reservations: dict[int, int] = {}

    def load(self) -> int:
        with self._lock:
            return self._value

    def load_link(self, tid: int) -> tuple[int, int]:
        """返回 (value, reservation_token)。"""
        with self._lock:
            token = self._version
            self._reservations[tid] = token
            return self._value, token

    def store_conditional(
        self,
        tid: int,
        token: int,
        expected_value: int,
        new_value: int,
    ) -> tuple[bool, int, str]:
        """尝试条件写入，返回 (success, observed, reason)。

        reason:
        - success
        - invalid_reservation
        - lost_reservation
        - value_mismatch
        """
        with self._lock:
            observed = self._value
            local_token = self._reservations.get(tid)
            if local_token is None or local_token != token:
                return False, observed, "invalid_reservation"

            if self._version != token:
                self._reservations.pop(tid, None)
                return False, observed, "lost_reservation"

            if observed != expected_value:
                self._reservations.pop(tid, None)
                return False, observed, "value_mismatch"

            self._value = new_value
            self._version += 1
            # 成功 SC 后仅消费本线程 reservation；
            # 其他线程旧 reservation 会因 version 变化在后续 SC 中判定为 lost_reservation。
            self._reservations.pop(tid, None)
            return True, observed, "success"


def retry_backoff_sleep(rng: random.Random, retries: int) -> None:
    """指数退避，降低高冲突下重复碰撞概率。"""
    cap = min(0.001, (2 ** min(retries, 10)) * 1e-6)
    time.sleep(rng.uniform(0.0, cap))


@dataclass
class ContentionResult:
    final_value: int
    expected_value: int
    elapsed_ms: float
    total_attempts: int
    total_successes: int
    total_failures: int
    failure_lost_reservation: int
    failure_value_mismatch: int
    failure_invalid_reservation: int
    contention_ratio: float
    mean_retries: float
    p95_retries: float
    max_retries: int
    attempts_by_thread: list[int]
    failures_by_thread: list[int]


def run_contention_simulation(
    num_threads: int = 8,
    ops_per_thread: int = 260,
    seed: int = 20260407,
    ll_sc_pause_prob: float = 0.55,
    ll_sc_pause_max_s: float = 0.0005,
) -> ContentionResult:
    if num_threads <= 0:
        raise ValueError("num_threads must be > 0")
    if ops_per_thread <= 0:
        raise ValueError("ops_per_thread must be > 0")

    counter = AtomicLLSCInteger(initial=0)

    retries_per_success: list[int] = []
    attempts_by_thread = [0 for _ in range(num_threads)]
    failures_by_thread = [0 for _ in range(num_threads)]

    successes_by_thread = [0 for _ in range(num_threads)]
    lost_by_thread = [0 for _ in range(num_threads)]
    value_mismatch_by_thread = [0 for _ in range(num_threads)]
    invalid_resv_by_thread = [0 for _ in range(num_threads)]

    metrics_lock = threading.Lock()

    def worker(tid: int) -> None:
        rng = random.Random(seed + tid * 7919)

        local_attempts = 0
        local_failures = 0
        local_successes = 0
        local_lost = 0
        local_mismatch = 0
        local_invalid = 0
        local_retries: list[int] = []

        for _ in range(ops_per_thread):
            retries = 0
            while True:
                expected, token = counter.load_link(tid)

                if rng.random() < ll_sc_pause_prob:
                    time.sleep(rng.uniform(0.0, ll_sc_pause_max_s))

                success, _, reason = counter.store_conditional(
                    tid=tid,
                    token=token,
                    expected_value=expected,
                    new_value=expected + 1,
                )
                local_attempts += 1

                if success:
                    local_successes += 1
                    local_retries.append(retries)
                    break

                local_failures += 1
                if reason == "lost_reservation":
                    local_lost += 1
                elif reason == "value_mismatch":
                    local_mismatch += 1
                elif reason == "invalid_reservation":
                    local_invalid += 1
                else:
                    raise RuntimeError(f"Unexpected SC failure reason: {reason}")

                retries += 1
                retry_backoff_sleep(rng, retries)

                if retries > 100_000:
                    raise RuntimeError("LL/SC retries exceeded safety threshold")

        with metrics_lock:
            attempts_by_thread[tid] = local_attempts
            failures_by_thread[tid] = local_failures
            successes_by_thread[tid] = local_successes
            lost_by_thread[tid] = local_lost
            value_mismatch_by_thread[tid] = local_mismatch
            invalid_resv_by_thread[tid] = local_invalid
            retries_per_success.extend(local_retries)

    threads = [
        threading.Thread(target=worker, args=(tid,), name=f"llsc-worker-{tid}")
        for tid in range(num_threads)
    ]
    random.Random(seed).shuffle(threads)

    start = time.perf_counter()
    for th in threads:
        th.start()

    for th in threads:
        th.join(timeout=30.0)

    alive_threads = [th.name for th in threads if th.is_alive()]
    assert not alive_threads, f"Threads did not finish: {alive_threads}"

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    expected_value = num_threads * ops_per_thread
    final_value = counter.load()

    total_attempts = int(np.sum(attempts_by_thread))
    total_failures = int(np.sum(failures_by_thread))
    total_successes = int(np.sum(successes_by_thread))
    total_lost = int(np.sum(lost_by_thread))
    total_mismatch = int(np.sum(value_mismatch_by_thread))
    total_invalid = int(np.sum(invalid_resv_by_thread))

    retry_arr = np.array(retries_per_success, dtype=np.int64)
    if retry_arr.size == 0:
        mean_retries = 0.0
        p95_retries = 0.0
        max_retries = 0
    else:
        mean_retries = float(retry_arr.mean())
        p95_retries = float(np.percentile(retry_arr, 95))
        max_retries = int(retry_arr.max())

    contention_ratio = (
        float(total_failures / total_attempts) if total_attempts > 0 else 0.0
    )

    assert final_value == expected_value, (
        f"Final value mismatch: got {final_value}, expected {expected_value}"
    )
    assert total_successes == expected_value, (
        f"Success count mismatch: got {total_successes}, expected {expected_value}"
    )
    assert total_attempts == total_successes + total_failures, (
        "Attempt accounting mismatch: attempts != successes + failures"
    )
    assert len(retries_per_success) == expected_value, (
        f"Retry record count mismatch: got {len(retries_per_success)}, expected {expected_value}"
    )
    assert total_failures == total_lost + total_mismatch + total_invalid, (
        "Failure reason accounting mismatch"
    )
    assert max_retries < 5000, f"Max retries too large: {max_retries}"

    print("=== LL/SC Contention Simulation ===")
    print(f"Threads: {num_threads}, operations per thread: {ops_per_thread}")
    print(f"Elapsed: {elapsed_ms:.3f} ms")
    print(f"Final counter value: {final_value}")
    print(f"Expected counter value: {expected_value}")
    print(f"Total LL/SC attempts: {total_attempts}")
    print(f"Total LL/SC successes: {total_successes}")
    print(f"Total LL/SC failures: {total_failures}")
    print(f"  - lost reservation: {total_lost}")
    print(f"  - value mismatch: {total_mismatch}")
    print(f"  - invalid reservation: {total_invalid}")
    print(f"Contention ratio (failures/attempts): {contention_ratio:.4f}")
    print(
        "Retries per successful increment: "
        f"mean={mean_retries:.3f}, p95={p95_retries:.3f}, max={max_retries}"
    )
    print("Attempts by thread:", attempts_by_thread)
    print("Failures by thread:", failures_by_thread)
    print("All assertions passed for contention simulation.")

    return ContentionResult(
        final_value=final_value,
        expected_value=expected_value,
        elapsed_ms=elapsed_ms,
        total_attempts=total_attempts,
        total_successes=total_successes,
        total_failures=total_failures,
        failure_lost_reservation=total_lost,
        failure_value_mismatch=total_mismatch,
        failure_invalid_reservation=total_invalid,
        contention_ratio=contention_ratio,
        mean_retries=mean_retries,
        p95_retries=p95_retries,
        max_retries=max_retries,
        attempts_by_thread=attempts_by_thread,
        failures_by_thread=failures_by_thread,
    )


def run_aba_resistance_scenario() -> dict[str, object]:
    """构造“值回到旧值”的场景，验证旧 reservation 的 SC 仍会失败。"""
    cell = AtomicLLSCInteger(initial=5)

    thread_a = 100
    thread_b = 200

    value_a, token_a = cell.load_link(thread_a)

    value_b1, token_b1 = cell.load_link(thread_b)
    ok_b1, _, reason_b1 = cell.store_conditional(thread_b, token_b1, value_b1, 6)

    value_b2, token_b2 = cell.load_link(thread_b)
    ok_b2, _, reason_b2 = cell.store_conditional(thread_b, token_b2, value_b2, 5)

    # 对 A 而言：当前值又回到 5，但 token_a 已经过期。
    ok_a, observed_a, reason_a = cell.store_conditional(thread_a, token_a, value_a, 42)

    assert ok_b1 and ok_b2, "Thread B should complete both writes"
    assert reason_b1 == "success" and reason_b2 == "success"
    assert not ok_a, "Old reservation should fail after intervening writes"
    assert reason_a in {"lost_reservation", "invalid_reservation"}
    assert observed_a == 5, "Current value should have returned to original value"

    result = {
        "a_initial_value": value_a,
        "a_token": token_a,
        "b_write1_success": ok_b1,
        "b_write2_success": ok_b2,
        "a_store_conditional_success": ok_a,
        "a_store_conditional_reason": reason_a,
        "observed_value_when_a_sc": observed_a,
        "final_value": cell.load(),
    }

    print("\n=== ABA-like Scenario Check ===")
    print(
        "Thread A LL -> value=", value_a,
        ", token=", token_a,
        "; Thread B does 5->6->5; A old SC success=", ok_a,
        ", reason=", reason_a,
        ", observed=", observed_a,
        sep="",
    )
    print("ABA-like resistance assertions passed.")

    return result


def main() -> None:
    run_contention_simulation()
    run_aba_resistance_scenario()


if __name__ == "__main__":
    main()

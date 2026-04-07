"""生产者-消费者问题: 可运行的最小 Python MVP。

实现要点:
- 使用 `threading.Condition` 显式实现有界缓冲区（不把 `queue.Queue` 当黑箱）。
- 多生产者 + 多消费者并发运行。
- 使用哨兵对象结束消费者。
- 程序内做一致性断言，保证可自动验证。
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import statistics
import threading
import time
from typing import Optional

try:
    import numpy as np
except Exception:  # pragma: no cover - 仅作环境兜底
    np = None  # type: ignore[assignment]


@dataclass(frozen=True)
class Payload:
    """业务负载: 由生产者生成，消费者处理。"""

    producer_id: int
    local_index: int
    value: int


@dataclass(frozen=True)
class EnqueuedItem:
    """进入缓冲区后的数据项，带全局入队序号。"""

    enqueue_seq: int
    payload: Optional[Payload]  # None 表示停止哨兵


class BoundedBuffer:
    """基于条件变量实现的有界 FIFO 缓冲区。"""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._queue: deque[EnqueuedItem] = deque()
        self._lock = threading.Lock()
        self._not_full = threading.Condition(self._lock)
        self._not_empty = threading.Condition(self._lock)

        self._next_enqueue_seq = 0
        self.max_size_seen = 0
        self.put_wait_count = 0
        self.get_wait_count = 0

    def put(self, payload: Optional[Payload]) -> None:
        """生产者入队; 满缓冲区时阻塞等待。"""
        with self._not_full:
            while len(self._queue) >= self.capacity:
                self.put_wait_count += 1
                self._not_full.wait()

            item = EnqueuedItem(enqueue_seq=self._next_enqueue_seq, payload=payload)
            self._next_enqueue_seq += 1
            self._queue.append(item)
            if len(self._queue) > self.max_size_seen:
                self.max_size_seen = len(self._queue)
            self._not_empty.notify()

    def get(self) -> EnqueuedItem:
        """消费者出队; 空缓冲区时阻塞等待。"""
        with self._not_empty:
            while not self._queue:
                self.get_wait_count += 1
                self._not_empty.wait()

            item = self._queue.popleft()
            self._not_full.notify()
            return item


def producer_worker(
    producer_id: int,
    item_count: int,
    buffer: BoundedBuffer,
    produced_ids: list[tuple[int, int]],
    produced_lock: threading.Lock,
) -> None:
    """单个生产者线程: 生产固定数量数据并放入缓冲区。"""
    sleep_rng = None
    if np is not None:
        sleep_rng = np.random.default_rng(20260407 + producer_id)

    for local_index in range(item_count):
        payload = Payload(
            producer_id=producer_id,
            local_index=local_index,
            value=producer_id * 1_000_000 + local_index,
        )
        buffer.put(payload)

        with produced_lock:
            produced_ids.append((producer_id, local_index))

        # 小幅随机睡眠用于制造并发交错，使同步逻辑可观测。
        if sleep_rng is not None:
            time.sleep(float(sleep_rng.uniform(0.0001, 0.0012)))
        else:
            time.sleep(0.0005)


def consumer_worker(
    consumer_id: int,
    buffer: BoundedBuffer,
    consumed_ids: list[tuple[int, int]],
    consumed_enqueue_seq: list[int],
    consumed_by_consumer: list[int],
    consumed_lock: threading.Lock,
) -> None:
    """单个消费者线程: 持续消费直到拿到停止哨兵。"""
    sleep_rng = None
    if np is not None:
        sleep_rng = np.random.default_rng(303000 + consumer_id)

    while True:
        item = buffer.get()
        if item.payload is None:
            break

        payload = item.payload
        with consumed_lock:
            consumed_ids.append((payload.producer_id, payload.local_index))
            consumed_enqueue_seq.append(item.enqueue_seq)
            consumed_by_consumer.append(consumer_id)

        # 模拟业务处理（非阻塞式轻计算）。
        _ = payload.value * payload.value

        if sleep_rng is not None:
            time.sleep(float(sleep_rng.uniform(0.0002, 0.0015)))
        else:
            time.sleep(0.0007)


def main() -> None:
    producer_count = 3
    consumer_count = 2
    items_per_producer = 60
    capacity = 8

    expected_total = producer_count * items_per_producer
    buffer = BoundedBuffer(capacity=capacity)

    produced_ids: list[tuple[int, int]] = []
    consumed_ids: list[tuple[int, int]] = []
    consumed_enqueue_seq: list[int] = []
    consumed_by_consumer: list[int] = []
    produced_lock = threading.Lock()
    consumed_lock = threading.Lock()

    consumers = [
        threading.Thread(
            target=consumer_worker,
            args=(
                cid,
                buffer,
                consumed_ids,
                consumed_enqueue_seq,
                consumed_by_consumer,
                consumed_lock,
            ),
            name=f"consumer-{cid}",
        )
        for cid in range(consumer_count)
    ]

    producers = [
        threading.Thread(
            target=producer_worker,
            args=(pid, items_per_producer, buffer, produced_ids, produced_lock),
            name=f"producer-{pid}",
        )
        for pid in range(producer_count)
    ]

    start_ts = time.perf_counter()

    for t in consumers:
        t.start()
    for t in producers:
        t.start()

    for t in producers:
        t.join()

    # 所有生产者结束后投递停止哨兵，保证每个消费者都能退出。
    for _ in range(consumer_count):
        buffer.put(None)

    for t in consumers:
        t.join()

    elapsed = time.perf_counter() - start_ts

    # ---------- 自动校验 ----------
    assert len(produced_ids) == expected_total, "produced count mismatch"
    assert len(consumed_ids) == expected_total, "consumed count mismatch"
    assert Counter(produced_ids) == Counter(consumed_ids), "lost or duplicated items detected"

    # FIFO 验证: 出队的全局入队序应严格递增。
    assert consumed_enqueue_seq == sorted(consumed_enqueue_seq), "FIFO property broken"

    # 单生产者内顺序验证: 每个生产者 local_index 不应逆序。
    producer_to_indices: dict[int, list[int]] = {pid: [] for pid in range(producer_count)}
    for pid, idx in consumed_ids:
        producer_to_indices[pid].append(idx)
    for pid, seq in producer_to_indices.items():
        assert seq == sorted(seq), f"producer {pid} order violated"

    assert buffer.max_size_seen <= capacity, "buffer overflow detected"

    # ---------- 汇总输出 ----------
    consumer_load = [consumed_by_consumer.count(cid) for cid in range(consumer_count)]
    throughput = expected_total / elapsed if elapsed > 0 else float("inf")

    if np is not None:
        load_std = float(np.std(np.array(consumer_load, dtype=float)))
    else:
        load_std = float(statistics.pstdev(consumer_load)) if len(consumer_load) > 1 else 0.0

    print("Producer-Consumer MVP finished.")
    print(
        f"produced={expected_total}, consumed={len(consumed_ids)}, "
        f"elapsed={elapsed:.4f}s, throughput={throughput:.2f} items/s"
    )
    print(
        f"buffer_peak={buffer.max_size_seen}/{capacity}, "
        f"put_waits={buffer.put_wait_count}, get_waits={buffer.get_wait_count}"
    )
    print(f"consumer_load={consumer_load}, load_std={load_std:.3f}")
    print("All assertions passed.")


if __name__ == "__main__":
    main()

"""Minimal runnable MVP for Deque (double-ended queue)."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from time import perf_counter
from typing import Iterable, Optional

import numpy as np
import pandas as pd


@dataclass
class Workload:
    operations: list[str]
    values: np.ndarray


class RingDeque:
    """A minimal deque using circular buffer + dynamic growth."""

    def __init__(self, init_capacity: int = 8) -> None:
        if init_capacity <= 0:
            raise ValueError("init_capacity must be positive")
        self._buffer: list[Optional[int]] = [None] * init_capacity
        self._head = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    @property
    def capacity(self) -> int:
        return len(self._buffer)

    def _ensure_capacity_for_push(self) -> None:
        if self._size == self.capacity:
            self._grow()

    def _grow(self) -> None:
        old_capacity = self.capacity
        new_capacity = old_capacity * 2
        new_buffer: list[Optional[int]] = [None] * new_capacity
        for i in range(self._size):
            new_buffer[i] = self._buffer[(self._head + i) % old_capacity]
        self._buffer = new_buffer
        self._head = 0

    def append_left(self, x: int) -> None:
        self._ensure_capacity_for_push()
        self._head = (self._head - 1) % self.capacity
        self._buffer[self._head] = x
        self._size += 1

    def append_right(self, x: int) -> None:
        self._ensure_capacity_for_push()
        tail = (self._head + self._size) % self.capacity
        self._buffer[tail] = x
        self._size += 1

    def pop_left(self) -> int:
        if self._size == 0:
            raise IndexError("pop_left from an empty deque")
        value = self._buffer[self._head]
        self._buffer[self._head] = None
        self._head = (self._head + 1) % self.capacity
        self._size -= 1
        return int(value)

    def pop_right(self) -> int:
        if self._size == 0:
            raise IndexError("pop_right from an empty deque")
        tail = (self._head + self._size - 1) % self.capacity
        value = self._buffer[tail]
        self._buffer[tail] = None
        self._size -= 1
        return int(value)

    def to_list(self) -> list[int]:
        return [int(self._buffer[(self._head + i) % self.capacity]) for i in range(self._size)]


APPEND_OPS = ("append_left", "append_right")
POP_OPS = ("pop_left", "pop_right")
ALL_OPS = APPEND_OPS + POP_OPS


def generate_workload(steps: int = 50_000, seed: int = 2026) -> Workload:
    rng = np.random.default_rng(seed)
    op_ids = rng.choice(len(ALL_OPS), size=steps, p=[0.28, 0.28, 0.22, 0.22])
    ops = [ALL_OPS[i] for i in op_ids]
    values = rng.integers(low=-1_000_000, high=1_000_000, size=steps, dtype=np.int64)
    return Workload(operations=ops, values=values)


def normalize_workload_for_nonempty_pops(workload: Workload) -> Workload:
    """Avoid popping from empty deque by rewriting invalid pops to appends."""

    normalized_ops: list[str] = []
    logical_size = 0

    for op in workload.operations:
        if op in POP_OPS and logical_size == 0:
            op = "append_right"
        normalized_ops.append(op)
        if op in APPEND_OPS:
            logical_size += 1
        else:
            logical_size -= 1

    return Workload(operations=normalized_ops, values=workload.values)


def run_ring_deque(workload: Workload) -> tuple[RingDeque, float, list[int]]:
    dq = RingDeque(init_capacity=8)
    pop_trace: list[int] = []

    t0 = perf_counter()
    for i, op in enumerate(workload.operations):
        value = int(workload.values[i])
        if op == "append_left":
            dq.append_left(value)
        elif op == "append_right":
            dq.append_right(value)
        elif op == "pop_left":
            pop_trace.append(dq.pop_left())
        elif op == "pop_right":
            pop_trace.append(dq.pop_right())
        else:
            raise ValueError(f"unknown op: {op}")
    elapsed_ms = (perf_counter() - t0) * 1000
    return dq, elapsed_ms, pop_trace


def run_reference_deque(workload: Workload) -> tuple[deque[int], float, list[int]]:
    dq: deque[int] = deque()
    pop_trace: list[int] = []

    t0 = perf_counter()
    for i, op in enumerate(workload.operations):
        value = int(workload.values[i])
        if op == "append_left":
            dq.appendleft(value)
        elif op == "append_right":
            dq.append(value)
        elif op == "pop_left":
            pop_trace.append(dq.popleft())
        elif op == "pop_right":
            pop_trace.append(dq.pop())
        else:
            raise ValueError(f"unknown op: {op}")
    elapsed_ms = (perf_counter() - t0) * 1000
    return dq, elapsed_ms, pop_trace


def summarize_operations(operations: Iterable[str]) -> pd.DataFrame:
    counter = Counter(operations)
    rows = [{"op": op, "count": counter[op]} for op in ALL_OPS]
    total = sum(counter.values())
    df = pd.DataFrame(rows)
    df["ratio"] = df["count"] / float(total)
    return df


def main() -> None:
    workload = normalize_workload_for_nonempty_pops(generate_workload(steps=60_000, seed=2026))

    ring_dq, ring_ms, ring_pop_trace = run_ring_deque(workload)
    ref_dq, ref_ms, ref_pop_trace = run_reference_deque(workload)

    ring_list = ring_dq.to_list()
    ref_list = list(ref_dq)
    op_counter = Counter(workload.operations)
    size_delta = (op_counter["append_left"] + op_counter["append_right"]) - (
        op_counter["pop_left"] + op_counter["pop_right"]
    )

    assert ring_pop_trace == ref_pop_trace, "Pop trace mismatch between RingDeque and collections.deque"
    assert ring_list == ref_list, "Final sequence mismatch between RingDeque and collections.deque"
    assert len(ring_list) == len(ref_list), "Final size mismatch"
    assert size_delta == len(ring_list), "Operation-derived size does not match final size"

    perf_df = pd.DataFrame(
        [
            {
                "implementation": "RingDeque",
                "steps": len(workload.operations),
                "elapsed_ms": round(ring_ms, 3),
                "final_size": len(ring_list),
                "size_delta": size_delta,
            },
            {
                "implementation": "collections.deque",
                "steps": len(workload.operations),
                "elapsed_ms": round(ref_ms, 3),
                "final_size": len(ref_list),
                "size_delta": size_delta,
            },
        ]
    )

    print("=== Operation Distribution ===")
    print(summarize_operations(workload.operations).to_string(index=False))
    print()

    print("=== Performance & Consistency Summary ===")
    print(perf_df.to_string(index=False))
    print()

    head_preview = ring_list[:8]
    tail_preview = ring_list[-8:] if ring_list else []
    print("=== Final Sequence Preview ===")
    print(f"head(8): {head_preview}")
    print(f"tail(8): {tail_preview}")
    print(f"total_elements: {len(ring_list)}")


if __name__ == "__main__":
    main()

"""Minimal runnable MVP for MPI communication (CS-0302).

Run:
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class CommRecord:
    """One communication operation trace record."""

    step: int
    primitive: str
    src: int
    dst: int
    tag: str
    payload_repr: str
    note: str


def summarize_payload(payload: Any) -> str:
    """Compact payload summary for logs."""
    if isinstance(payload, np.ndarray):
        if payload.size <= 6:
            return f"ndarray(shape={payload.shape}, values={payload.tolist()})"
        head = payload.ravel()[:6].tolist()
        return f"ndarray(shape={payload.shape}, head6={head}, ... )"
    if isinstance(payload, (float, int)):
        return f"{payload}"
    if isinstance(payload, dict):
        keys = sorted(payload.keys())
        return f"dict(keys={keys})"
    return repr(payload)


class MiniMPIWorld:
    """A deterministic, single-process simulator for core MPI communication semantics."""

    def __init__(self, world_size: int, root: int = 0) -> None:
        if world_size < 2:
            raise ValueError("world_size must be >= 2")
        if root < 0 or root >= world_size:
            raise ValueError("root must be within [0, world_size)")
        self.world_size = world_size
        self.root = root
        self.mailboxes: dict[tuple[int, int, str], list[Any]] = {}
        self.records: list[CommRecord] = []

    def _record(self, primitive: str, src: int, dst: int, tag: str, payload: Any, note: str) -> None:
        self.records.append(
            CommRecord(
                step=len(self.records) + 1,
                primitive=primitive,
                src=src,
                dst=dst,
                tag=tag,
                payload_repr=summarize_payload(payload),
                note=note,
            )
        )

    @staticmethod
    def _clone_payload(payload: Any) -> Any:
        if isinstance(payload, np.ndarray):
            return payload.copy()
        return payload

    def send(self, src: int, dst: int, tag: str, payload: Any, note: str = "") -> None:
        key = (src, dst, tag)
        self.mailboxes.setdefault(key, []).append(self._clone_payload(payload))
        self._record("send", src, dst, tag, payload, note)

    def recv(self, dst: int, src: int, tag: str, note: str = "") -> Any:
        key = (src, dst, tag)
        queue = self.mailboxes.get(key, [])
        if not queue:
            raise ValueError(f"No message available for recv(src={src}, dst={dst}, tag={tag})")
        payload = queue.pop(0)
        self._record("recv", src, dst, tag, payload, note)
        return payload

    def bcast(self, value: Any, tag: str = "BCAST") -> dict[int, Any]:
        received: dict[int, Any] = {self.root: self._clone_payload(value)}
        for rank in range(self.world_size):
            if rank == self.root:
                continue
            self.send(self.root, rank, tag, value, note="root broadcasts a shared config")
        for rank in range(self.world_size):
            if rank == self.root:
                continue
            received[rank] = self.recv(rank, self.root, tag, note="rank receives broadcast value")
        return received

    def scatter_array(self, array: np.ndarray, tag: str = "SCATTER") -> dict[int, np.ndarray]:
        arr = np.asarray(array)
        chunks = np.array_split(arr, self.world_size)
        local_chunks: dict[int, np.ndarray] = {self.root: chunks[self.root].copy()}

        for rank in range(self.world_size):
            if rank == self.root:
                continue
            self.send(self.root, rank, tag, chunks[rank], note="root sends one chunk")

        for rank in range(self.world_size):
            if rank == self.root:
                continue
            payload = self.recv(rank, self.root, tag, note="rank receives one chunk")
            local_chunks[rank] = np.asarray(payload, dtype=arr.dtype)

        return local_chunks

    def gather_scalars(self, local_values: dict[int, float], tag: str = "GATHER") -> np.ndarray:
        if set(local_values.keys()) != set(range(self.world_size)):
            raise ValueError("local_values must provide one scalar for each rank")

        gathered: dict[int, float] = {self.root: float(local_values[self.root])}
        for rank in range(self.world_size):
            if rank == self.root:
                continue
            self.send(rank, self.root, tag, float(local_values[rank]), note="rank sends local scalar")

        for rank in range(self.world_size):
            if rank == self.root:
                continue
            gathered[rank] = float(self.recv(self.root, rank, tag, note="root gathers one scalar"))

        return np.array([gathered[r] for r in range(self.world_size)], dtype=float)

    def allreduce_sum(self, local_values: dict[int, float]) -> np.ndarray:
        reduced = self.gather_scalars(local_values, tag="REDUCE")
        total = float(np.sum(reduced))
        broadcasted = self.bcast(total, tag="ALLREDUCE_BCAST")
        return np.array([float(broadcasted[r]) for r in range(self.world_size)], dtype=float)

    def pending_message_count(self) -> int:
        return sum(len(q) for q in self.mailboxes.values())


def run_ring_exchange(world: MiniMPIWorld, tokens: dict[int, int]) -> dict[int, int]:
    """Simulate one round ring exchange: rank i sends to rank (i+1)%P."""
    if set(tokens.keys()) != set(range(world.world_size)):
        raise ValueError("tokens must provide one integer for each rank")

    for src in range(world.world_size):
        dst = (src + 1) % world.world_size
        world.send(src, dst, "RING", int(tokens[src]), note="ring send")

    received: dict[int, int] = {}
    for dst in range(world.world_size):
        src = (dst - 1 + world.world_size) % world.world_size
        received[dst] = int(world.recv(dst, src, "RING", note="ring recv"))

    return received


def print_comm_table(records: list[CommRecord]) -> None:
    print("=== Communication Trace ===")
    header = f"{'step':<4} {'prim':<6} {'src':<3} {'dst':<3} {'tag':<16} {'payload':<48} note"
    print(header)
    print("-" * len(header))
    for rec in records:
        payload = rec.payload_repr
        if len(payload) > 48:
            payload = payload[:45] + "..."
        print(
            f"{rec.step:<4} {rec.primitive:<6} {rec.src:<3} {rec.dst:<3} "
            f"{rec.tag:<16} {payload:<48} {rec.note}"
        )


def main() -> None:
    print("=== MPI Communication MVP (Simulated Semantics) ===")

    world = MiniMPIWorld(world_size=4, root=0)

    data = np.arange(1, 17, dtype=float)
    config_by_rank = world.bcast({"scale": 1.5}, tag="BCAST_CONFIG")
    local_chunks = world.scatter_array(data, tag="SCATTER_DATA")

    local_sums: dict[int, float] = {}
    transformed_by_rank: dict[int, np.ndarray] = {}

    for rank in range(world.world_size):
        scale = float(config_by_rank[rank]["scale"])
        transformed = local_chunks[rank] * scale
        transformed_by_rank[rank] = transformed
        local_sums[rank] = float(np.sum(transformed))

    gathered = world.gather_scalars(local_sums, tag="GATHER_LOCAL_SUM")
    allreduced = world.allreduce_sum(local_sums)
    ring_received = run_ring_exchange(
        world,
        tokens={0: 100, 1: 200, 2: 300, 3: 400},
    )

    expected_chunks = {
        0: np.array([1.0, 2.0, 3.0, 4.0]),
        1: np.array([5.0, 6.0, 7.0, 8.0]),
        2: np.array([9.0, 10.0, 11.0, 12.0]),
        3: np.array([13.0, 14.0, 15.0, 16.0]),
    }
    expected_gather = np.array([15.0, 39.0, 63.0, 87.0])
    expected_total = float(np.sum(expected_gather))
    expected_allreduce = np.full((world.world_size,), expected_total)
    expected_ring = {0: 400, 1: 100, 2: 200, 3: 300}

    for rank in range(world.world_size):
        if not np.array_equal(local_chunks[rank], expected_chunks[rank]):
            raise AssertionError(f"chunk mismatch on rank {rank}: {local_chunks[rank]}")

    if not np.allclose(gathered, expected_gather):
        raise AssertionError(f"gathered values mismatch: {gathered}")

    if not np.allclose(allreduced, expected_allreduce):
        raise AssertionError(f"allreduce mismatch: {allreduced}")

    if ring_received != expected_ring:
        raise AssertionError(f"ring exchange mismatch: {ring_received}")

    if world.pending_message_count() != 0:
        raise AssertionError("mailboxes are not empty after all communication")

    send_count = sum(1 for r in world.records if r.primitive == "send")
    recv_count = sum(1 for r in world.records if r.primitive == "recv")
    if send_count != recv_count:
        raise AssertionError(f"send/recv count mismatch: send={send_count}, recv={recv_count}")

    print(f"World size: {world.world_size}")
    print(f"Data vector: {data.tolist()}")
    print("Local transformed chunks:")
    for rank in range(world.world_size):
        print(f"  rank {rank}: {transformed_by_rank[rank].tolist()} sum={local_sums[rank]:.1f}")
    print(f"Gathered local sums on root: {gathered.tolist()}")
    print(f"Allreduce(global sum) replicated to all ranks: {allreduced.tolist()}")
    print(f"Ring received tokens: {ring_received}")
    print(f"Operation count: send={send_count}, recv={recv_count}, total={len(world.records)}")

    print_comm_table(world.records)
    print("\nAll checks passed for CS-0302 (MPI通信).")


if __name__ == "__main__":
    main()

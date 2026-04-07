"""Vector clock MVP for CS-0314.

Run:
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Vector = tuple[int, ...]


def vector_leq(a: Vector, b: Vector) -> bool:
    """Return True if a is component-wise <= b."""
    va = np.asarray(a, dtype=int)
    vb = np.asarray(b, dtype=int)
    return bool(np.all(va <= vb))


def vector_lt(a: Vector, b: Vector) -> bool:
    """Return True if a < b under vector-clock partial order."""
    va = np.asarray(a, dtype=int)
    vb = np.asarray(b, dtype=int)
    return bool(np.all(va <= vb) and np.any(va < vb))


def vector_concurrent(a: Vector, b: Vector) -> bool:
    """Return True if vectors are incomparable (concurrent)."""
    return not vector_leq(a, b) and not vector_leq(b, a)


@dataclass(frozen=True)
class Message:
    """A message in flight carrying vector clock metadata."""

    msg_id: str
    src: str
    dst: str
    payload: str
    clock: Vector
    send_event_id: int


@dataclass(frozen=True)
class Event:
    """A recorded event with vector-clock snapshot."""

    event_id: int
    process_id: str
    local_index: int
    kind: str
    vector: Vector
    detail: str
    msg_id: str | None = None
    peer: str | None = None
    linked_event_id: int | None = None
    msg_clock: Vector | None = None


class VectorClockProcess:
    """Per-process state for vector-clock updates."""

    def __init__(self, process_id: str, index: int, n_processes: int) -> None:
        self.process_id = process_id
        self.index = index
        self.clock = np.zeros(n_processes, dtype=int)
        self.local_index = 0

    def tick_local_or_send(self) -> tuple[Vector, int]:
        self.clock[self.index] += 1
        self.local_index += 1
        return tuple(int(x) for x in self.clock), self.local_index

    def tick_receive(self, remote_clock: Vector) -> tuple[Vector, int]:
        remote = np.asarray(remote_clock, dtype=int)
        self.clock = np.maximum(self.clock, remote)
        self.clock[self.index] += 1
        self.local_index += 1
        return tuple(int(x) for x in self.clock), self.local_index


class VectorClockSimulator:
    """Deterministic simulation for vector-clock algorithm behavior."""

    def __init__(self, process_ids: list[str]) -> None:
        if len(set(process_ids)) != len(process_ids):
            raise ValueError(f"Duplicate process ids are not allowed: {process_ids}")

        self.process_ids = process_ids
        self.process_index = {pid: idx for idx, pid in enumerate(process_ids)}
        n = len(process_ids)
        self.processes = {
            pid: VectorClockProcess(pid, self.process_index[pid], n) for pid in process_ids
        }

        self.events: list[Event] = []
        self.hb_edges: list[tuple[int, int]] = []
        self.in_flight: dict[str, Message] = {}
        self.last_event_per_process: dict[str, int] = {}

    def _must_get_process(self, process_id: str) -> VectorClockProcess:
        if process_id not in self.processes:
            raise ValueError(f"Unknown process id: {process_id}")
        return self.processes[process_id]

    def _record_event(
        self,
        process_id: str,
        local_index: int,
        kind: str,
        vector: Vector,
        detail: str,
        msg_id: str | None = None,
        peer: str | None = None,
        linked_event_id: int | None = None,
        msg_clock: Vector | None = None,
    ) -> int:
        event_id = len(self.events)
        prev = self.last_event_per_process.get(process_id)
        if prev is not None:
            self.hb_edges.append((prev, event_id))
        self.last_event_per_process[process_id] = event_id

        self.events.append(
            Event(
                event_id=event_id,
                process_id=process_id,
                local_index=local_index,
                kind=kind,
                vector=vector,
                detail=detail,
                msg_id=msg_id,
                peer=peer,
                linked_event_id=linked_event_id,
                msg_clock=msg_clock,
            )
        )
        return event_id

    def local(self, process_id: str, detail: str) -> int:
        process = self._must_get_process(process_id)
        vector, local_index = process.tick_local_or_send()
        return self._record_event(process_id, local_index, "local", vector, detail)

    def send(self, src: str, dst: str, msg_id: str, payload: str) -> int:
        if msg_id in self.in_flight:
            raise ValueError(f"Message id already in flight: {msg_id}")

        src_process = self._must_get_process(src)
        self._must_get_process(dst)
        vector, local_index = src_process.tick_local_or_send()

        send_event_id = self._record_event(
            process_id=src,
            local_index=local_index,
            kind="send",
            vector=vector,
            detail=f"send {msg_id} -> {dst}: {payload}",
            msg_id=msg_id,
            peer=dst,
        )

        self.in_flight[msg_id] = Message(
            msg_id=msg_id,
            src=src,
            dst=dst,
            payload=payload,
            clock=vector,
            send_event_id=send_event_id,
        )
        return send_event_id

    def recv(self, dst: str, msg_id: str, detail: str) -> int:
        if msg_id not in self.in_flight:
            raise ValueError(f"Cannot receive unknown or already consumed message: {msg_id}")

        message = self.in_flight.pop(msg_id)
        if message.dst != dst:
            raise ValueError(f"Message {msg_id} is addressed to {message.dst}, not {dst}")

        dst_process = self._must_get_process(dst)
        vector, local_index = dst_process.tick_receive(message.clock)

        recv_event_id = self._record_event(
            process_id=dst,
            local_index=local_index,
            kind="recv",
            vector=vector,
            detail=f"recv {msg_id} <- {message.src}: {detail}",
            msg_id=msg_id,
            peer=message.src,
            linked_event_id=message.send_event_id,
            msg_clock=message.clock,
        )

        self.hb_edges.append((message.send_event_id, recv_event_id))
        return recv_event_id

    def assert_vector_clock_condition(self) -> None:
        """For each happens-before edge e1->e2, check VC(e1) < VC(e2)."""
        for src, dst in self.hb_edges:
            src_event = self.events[src]
            dst_event = self.events[dst]
            assert vector_lt(src_event.vector, dst_event.vector), (
                "Vector-clock condition violated on edge "
                f"{src_event.event_id + 1}->{dst_event.event_id + 1}: "
                f"{src_event.vector} !< {dst_event.vector}"
            )

    def assert_no_inflight_messages(self) -> None:
        assert not self.in_flight, f"Undelivered messages remain: {sorted(self.in_flight.keys())}"


def compute_reachability(n_events: int, edges: list[tuple[int, int]]) -> np.ndarray:
    """Compute transitive closure of happens-before graph by boolean Warshall."""
    reach = np.zeros((n_events, n_events), dtype=bool)
    for src, dst in edges:
        reach[src, dst] = True

    for mid in range(n_events):
        reach = reach | (reach[:, [mid]] & reach[[mid], :])
    return reach


def assert_hb_equivalence(events: list[Event], reach: np.ndarray) -> None:
    """Check (reachability) <=> (vector strict partial order)."""
    n = len(events)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            by_graph = bool(reach[i, j])
            by_vector = vector_lt(events[i].vector, events[j].vector)
            assert by_graph == by_vector, (
                "Causality mismatch between graph and vector clock: "
                f"eid={events[i].event_id + 1} vs eid={events[j].event_id + 1}, "
                f"graph={by_graph}, vector={by_vector}, "
                f"vi={events[i].vector}, vj={events[j].vector}"
            )


def find_one_concurrent_pair(events: list[Event], reach: np.ndarray) -> tuple[Event, Event] | None:
    """Find one pair that is concurrent both by graph and by vector relation."""
    n = len(events)
    for i in range(n):
        for j in range(i + 1, n):
            by_graph = not reach[i, j] and not reach[j, i]
            by_vector = vector_concurrent(events[i].vector, events[j].vector)
            if by_graph and by_vector:
                return events[i], events[j]
    return None


def build_demo_scenario() -> VectorClockSimulator:
    """Construct a deterministic trace with 3 processes and 4 cross-process messages."""
    sim = VectorClockSimulator(["P1", "P2", "P3"])

    sim.local("P1", "读取配置")
    sim.local("P2", "加载本地缓存")
    sim.send("P1", "P2", "m1", "同步版本号")
    sim.local("P3", "后台心跳")
    sim.recv("P2", "m1", "处理来自P1的同步")
    sim.send("P2", "P3", "m2", "转发确认")
    sim.local("P1", "继续计算任务A")
    sim.recv("P3", "m2", "接收P2确认")
    sim.send("P3", "P1", "m3", "回传聚合结果")
    sim.recv("P1", "m3", "合并来自P3结果")
    sim.local("P2", "写审计日志")
    sim.send("P1", "P2", "m4", "通知收尾")
    sim.recv("P2", "m4", "完成收尾")
    sim.local("P3", "清理临时状态")

    return sim


def _fmt_vector(v: Vector) -> str:
    return "[" + ", ".join(str(x) for x in v) + "]"


def print_event_table(events: list[Event]) -> None:
    print("=== Event Log (execution order) ===")
    header = (
        f"{'eid':<4} {'proc':<4} {'lidx':<4} {'kind':<5} "
        f"{'VC':<14} {'msg':<4} {'peer':<4} {'msgVC':<14} detail"
    )
    print(header)
    print("-" * len(header))
    for e in events:
        msg = e.msg_id or "-"
        peer = e.peer or "-"
        msg_vc = _fmt_vector(e.msg_clock) if e.msg_clock is not None else "-"
        print(
            f"{e.event_id + 1:<4} {e.process_id:<4} {e.local_index:<4} {e.kind:<5} "
            f"{_fmt_vector(e.vector):<14} {msg:<4} {peer:<4} {msg_vc:<14} {e.detail}"
        )


def print_process_clocks(sim: VectorClockSimulator) -> None:
    print("\n=== Final Vector Clocks ===")
    for pid in sim.process_ids:
        p = sim.processes[pid]
        print(f"{pid}: VC={_fmt_vector(tuple(int(x) for x in p.clock))}, local_events={p.local_index}")


def main() -> None:
    sim = build_demo_scenario()
    sim.assert_vector_clock_condition()
    sim.assert_no_inflight_messages()

    events = sim.events
    reach = compute_reachability(len(events), sim.hb_edges)
    assert_hb_equivalence(events, reach)

    print_event_table(events)
    print_process_clocks(sim)

    print("\n=== Concurrency Check ===")
    pair = find_one_concurrent_pair(events, reach)
    if pair is None:
        print("No concurrent pair found in this trace.")
    else:
        a, b = pair
        print(
            "Found concurrent events: "
            f"eid={a.event_id + 1}({a.process_id},{a.kind},VC={_fmt_vector(a.vector)}) and "
            f"eid={b.event_id + 1}({b.process_id},{b.kind},VC={_fmt_vector(b.vector)})."
        )

    print("\nAll checks passed for CS-0314 (向量时钟).")


if __name__ == "__main__":
    main()

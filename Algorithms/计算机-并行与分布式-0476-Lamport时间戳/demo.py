"""Lamport timestamp MVP for CS-0315.

Run:
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Message:
    """A message sent between processes in the simulation."""

    msg_id: str
    src: str
    dst: str
    payload: str
    send_ts: int
    send_event_id: int


@dataclass(frozen=True)
class Event:
    """A recorded event with Lamport timestamp."""

    event_id: int
    process_id: str
    local_index: int
    kind: str
    lamport_ts: int
    detail: str
    msg_id: str | None = None
    peer: str | None = None
    linked_event_id: int | None = None
    send_ts: int | None = None


class LamportProcess:
    """Per-process logical clock state."""

    def __init__(self, process_id: str) -> None:
        self.process_id = process_id
        self.clock = 0
        self.local_index = 0

    def tick_local_or_send(self) -> tuple[int, int]:
        self.clock += 1
        self.local_index += 1
        return self.clock, self.local_index

    def tick_receive(self, remote_ts: int) -> tuple[int, int]:
        self.clock = max(self.clock, remote_ts) + 1
        self.local_index += 1
        return self.clock, self.local_index


class LamportSimulator:
    """Deterministic event script runner for Lamport timestamp demonstration."""

    def __init__(self, process_ids: list[str]) -> None:
        if len(set(process_ids)) != len(process_ids):
            raise ValueError(f"Duplicate process ids are not allowed: {process_ids}")
        self.processes = {pid: LamportProcess(pid) for pid in process_ids}
        self.events: list[Event] = []
        self.in_flight: dict[str, Message] = {}
        self.hb_edges: list[tuple[int, int]] = []
        self.last_event_per_process: dict[str, int] = {}

    def _record_event(
        self,
        process_id: str,
        local_index: int,
        kind: str,
        lamport_ts: int,
        detail: str,
        msg_id: str | None = None,
        peer: str | None = None,
        linked_event_id: int | None = None,
        send_ts: int | None = None,
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
                lamport_ts=lamport_ts,
                detail=detail,
                msg_id=msg_id,
                peer=peer,
                linked_event_id=linked_event_id,
                send_ts=send_ts,
            )
        )
        return event_id

    def local(self, process_id: str, detail: str) -> int:
        process = self._must_get_process(process_id)
        ts, local_index = process.tick_local_or_send()
        return self._record_event(process_id, local_index, "local", ts, detail)

    def send(self, src: str, dst: str, msg_id: str, payload: str) -> int:
        if msg_id in self.in_flight:
            raise ValueError(f"Message id already in flight: {msg_id}")

        src_process = self._must_get_process(src)
        self._must_get_process(dst)
        ts, local_index = src_process.tick_local_or_send()

        send_event_id = self._record_event(
            process_id=src,
            local_index=local_index,
            kind="send",
            lamport_ts=ts,
            detail=f"send {msg_id} -> {dst}: {payload}",
            msg_id=msg_id,
            peer=dst,
        )

        self.in_flight[msg_id] = Message(
            msg_id=msg_id,
            src=src,
            dst=dst,
            payload=payload,
            send_ts=ts,
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
        ts, local_index = dst_process.tick_receive(message.send_ts)

        recv_event_id = self._record_event(
            process_id=dst,
            local_index=local_index,
            kind="recv",
            lamport_ts=ts,
            detail=f"recv {msg_id} <- {message.src}: {detail}",
            msg_id=msg_id,
            peer=message.src,
            linked_event_id=message.send_event_id,
            send_ts=message.send_ts,
        )

        self.hb_edges.append((message.send_event_id, recv_event_id))
        return recv_event_id

    def _must_get_process(self, process_id: str) -> LamportProcess:
        if process_id not in self.processes:
            raise ValueError(f"Unknown process id: {process_id}")
        return self.processes[process_id]

    def assert_clock_condition(self) -> None:
        """Check Lamport's clock condition on all happens-before edges."""
        for src, dst in self.hb_edges:
            src_event = self.events[src]
            dst_event = self.events[dst]
            assert src_event.lamport_ts < dst_event.lamport_ts, (
                "Lamport clock condition violated on edge "
                f"{src_event.event_id + 1}->{dst_event.event_id + 1}: "
                f"{src_event.lamport_ts} !< {dst_event.lamport_ts}"
            )

    def assert_no_inflight_messages(self) -> None:
        assert not self.in_flight, f"Undelivered messages remain: {sorted(self.in_flight.keys())}"


def compute_reachability(n_events: int, edges: list[tuple[int, int]]) -> np.ndarray:
    """Transitive closure of happens-before relation using boolean Warshall."""
    reach = np.zeros((n_events, n_events), dtype=bool)
    for src, dst in edges:
        reach[src, dst] = True

    for mid in range(n_events):
        reach = reach | (reach[:, [mid]] & reach[[mid], :])
    return reach


def find_one_concurrent_pair(events: list[Event], reach: np.ndarray) -> tuple[Event, Event] | None:
    """Find one pair of concurrent events (neither happens-before the other)."""
    n = len(events)
    for i in range(n):
        for j in range(i + 1, n):
            if not reach[i, j] and not reach[j, i]:
                return events[i], events[j]
    return None


def build_demo_scenario() -> LamportSimulator:
    """Construct a deterministic distributed trace with 3 processes and 4 messages."""
    sim = LamportSimulator(["P1", "P2", "P3"])

    sim.local("P1", "读取配置")
    sim.local("P2", "加载本地缓存")
    sim.send("P1", "P2", "m1", "同步版本号")
    sim.local("P3", "后台心跳")
    sim.recv("P2", "m1", "处理来自P1的同步")
    sim.send("P2", "P3", "m2", "转发确认")
    sim.local("P1", "继续计算任务A")
    sim.recv("P3", "m2", "接收P2确认")
    sim.send("P3", "P1", "m3", "回传聚合结果")
    sim.recv("P1", "m3", "合并来自P3的结果")
    sim.local("P2", "写审计日志")
    sim.send("P1", "P2", "m4", "通知收尾")
    sim.recv("P2", "m4", "完成收尾")

    return sim


def print_event_table(events: list[Event]) -> None:
    print("=== Event Log (execution order) ===")
    header = (
        f"{'eid':<4} {'proc':<4} {'lidx':<4} {'kind':<5} {'L':<3} "
        f"{'msg':<4} {'peer':<4} {'sendL':<5} detail"
    )
    print(header)
    print("-" * len(header))

    for e in events:
        msg = e.msg_id or "-"
        peer = e.peer or "-"
        send_l = str(e.send_ts) if e.send_ts is not None else "-"
        print(
            f"{e.event_id + 1:<4} {e.process_id:<4} {e.local_index:<4} {e.kind:<5} "
            f"{e.lamport_ts:<3} {msg:<4} {peer:<4} {send_l:<5} {e.detail}"
        )


def print_total_order(events: list[Event]) -> None:
    print("\n=== Total Order by (LamportTs, ProcessId, LocalIndex) ===")
    ordered = sorted(events, key=lambda e: (e.lamport_ts, e.process_id, e.local_index))
    for rank, e in enumerate(ordered, start=1):
        print(
            f"[{rank:02d}] eid={e.event_id + 1}, ts={e.lamport_ts}, "
            f"proc={e.process_id}, kind={e.kind}, detail={e.detail}"
        )


def print_process_clocks(sim: LamportSimulator) -> None:
    print("\n=== Final Logical Clocks ===")
    for pid in sorted(sim.processes.keys()):
        p = sim.processes[pid]
        print(f"{pid}: clock={p.clock}, local_events={p.local_index}")


def main() -> None:
    sim = build_demo_scenario()
    sim.assert_clock_condition()
    sim.assert_no_inflight_messages()

    events = sim.events
    reach = compute_reachability(len(events), sim.hb_edges)

    print_event_table(events)
    print_total_order(events)
    print_process_clocks(sim)

    pair = find_one_concurrent_pair(events, reach)
    print("\n=== Concurrency Check ===")
    if pair is None:
        print("No concurrent pair found in this trace.")
    else:
        a, b = pair
        relation = (
            "equal-ts" if a.lamport_ts == b.lamport_ts else f"ts-order({a.lamport_ts} vs {b.lamport_ts})"
        )
        print(
            "Found concurrent events: "
            f"eid={a.event_id + 1}({a.process_id},{a.kind}) and "
            f"eid={b.event_id + 1}({b.process_id},{b.kind}), {relation}."
        )
        print("This shows Lamport timestamps preserve causality but do not prove causality in reverse.")

    print("\nAll checks passed for CS-0315 (Lamport时间戳).")


if __name__ == "__main__":
    main()

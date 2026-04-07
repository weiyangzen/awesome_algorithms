"""Minimal runnable MVP for Chandy-Lamport distributed snapshot."""

from __future__ import annotations

from dataclasses import dataclass, field
from heapq import heappop, heappush
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass(frozen=True)
class AppPayload:
    transfer_id: str
    amount: int


@dataclass(frozen=True)
class Message:
    kind: str  # "APP" | "MARKER"
    src: int
    dst: int
    snapshot_id: Optional[str] = None
    payload: Optional[AppPayload] = None


@dataclass(order=True)
class Envelope:
    deliver_step: int
    seq: int
    message: Message = field(compare=False)


@dataclass
class ProcessState:
    pid: int
    balance: int


@dataclass
class SnapshotRuntime:
    snapshot_id: str
    initiator: int
    local_state: Dict[int, int]
    record_time: Dict[int, int]
    channel_state: Dict[Tuple[int, int], List[Message]]
    marker_received: Dict[int, Set[int]]
    marker_arrival_time: Dict[Tuple[int, int], int]
    marker_arrival_order: Dict[Tuple[int, int], int]
    recording_channels: Dict[int, Set[int]]
    completed: bool = False
    complete_step: Optional[int] = None


class DistributedSnapshotSimulator:
    """Discrete-time simulator with FIFO channels and Chandy-Lamport snapshot."""

    def __init__(
        self,
        initial_balances: Dict[int, int],
        directed_channels: List[Tuple[int, int]],
    ) -> None:
        if not initial_balances:
            raise ValueError("initial_balances must be non-empty")

        self.processes: Dict[int, ProcessState] = {
            pid: ProcessState(pid=pid, balance=int(balance))
            for pid, balance in sorted(initial_balances.items())
        }
        self.pids = sorted(self.processes.keys())

        self.out_neighbors: Dict[int, List[int]] = {pid: [] for pid in self.pids}
        self.in_neighbors: Dict[int, List[int]] = {pid: [] for pid in self.pids}
        self.channels: List[Tuple[int, int]] = []

        seen_channels: Set[Tuple[int, int]] = set()
        for src, dst in directed_channels:
            if src == dst:
                raise ValueError("self-channel is not allowed in this MVP")
            if src not in self.processes or dst not in self.processes:
                raise ValueError(f"invalid channel ({src}->{dst})")
            if (src, dst) in seen_channels:
                continue
            seen_channels.add((src, dst))
            self.channels.append((src, dst))
            self.out_neighbors[src].append(dst)
            self.in_neighbors[dst].append(src)

        for pid in self.pids:
            if not self.in_neighbors[pid]:
                raise ValueError(f"process {pid} has no incoming channel")
            if not self.out_neighbors[pid]:
                raise ValueError(f"process {pid} has no outgoing channel")

        self.time = 0
        self.seq_counter = 0
        self.pending: List[Envelope] = []
        self.channel_last_delivery: Dict[Tuple[int, int], int] = {}
        self.snapshot: Optional[SnapshotRuntime] = None
        self.marker_default_delay = 1
        self.marker_delay_overrides: Dict[Tuple[int, int], int] = {}
        self.dispatch_order = 0
        self.current_dispatch_order = -1

        self.initial_total = int(np.sum(np.array([p.balance for p in self.processes.values()], dtype=int)))
        self.transfer_history: Dict[str, Dict[str, int]] = {}
        self.logs: List[str] = []

    def _log(self, text: str) -> None:
        self.logs.append(f"[t={self.time}] {text}")

    def _enqueue(self, message: Message, delay: int) -> int:
        if delay < 0:
            raise ValueError("delay must be non-negative")

        channel = (message.src, message.dst)
        desired_step = self.time + delay
        last_step = self.channel_last_delivery.get(channel, -10**9)
        deliver_step = max(desired_step, last_step)
        self.channel_last_delivery[channel] = deliver_step

        envelope = Envelope(deliver_step=deliver_step, seq=self.seq_counter, message=message)
        self.seq_counter += 1
        heappush(self.pending, envelope)
        return deliver_step

    def send_app(self, src: int, dst: int, amount: int, delay: int, transfer_id: str) -> None:
        if amount <= 0:
            raise ValueError("amount must be positive")
        if transfer_id in self.transfer_history:
            raise ValueError(f"duplicate transfer_id: {transfer_id}")

        sender = self.processes[src]
        if sender.balance < amount:
            raise ValueError(
                f"insufficient balance for transfer {transfer_id}: "
                f"balance={sender.balance}, amount={amount}"
            )

        sender.balance -= amount
        payload = AppPayload(transfer_id=transfer_id, amount=amount)
        msg = Message(kind="APP", src=src, dst=dst, payload=payload)
        deliver_step = self._enqueue(msg, delay)

        self.transfer_history[transfer_id] = {
            "src": src,
            "dst": dst,
            "amount": amount,
            "send_step": self.time,
            "receive_step": -1,
            "receive_order": -1,
        }
        self._log(
            f"SEND APP {transfer_id}: P{src}->P{dst}, amount={amount}, "
            f"deliver@t={deliver_step}, sender_balance_now={sender.balance}"
        )

    def _marker_delay(self, src: int, dst: int) -> int:
        return self.marker_delay_overrides.get((src, dst), self.marker_default_delay)

    def _send_marker(self, src: int, dst: int, snapshot_id: str) -> None:
        delay = self._marker_delay(src, dst)
        marker = Message(kind="MARKER", src=src, dst=dst, snapshot_id=snapshot_id)
        deliver_step = self._enqueue(marker, delay)
        self._log(f"SEND MARKER {snapshot_id}: P{src}->P{dst}, deliver@t={deliver_step}")

    def initiate_snapshot(
        self,
        initiator: int,
        snapshot_id: str,
        marker_default_delay: int,
        marker_delay_overrides: Optional[Dict[Tuple[int, int], int]] = None,
    ) -> None:
        if self.snapshot is not None and not self.snapshot.completed:
            raise RuntimeError("another snapshot is still active")

        self.marker_default_delay = marker_default_delay
        self.marker_delay_overrides = marker_delay_overrides or {}

        runtime = SnapshotRuntime(
            snapshot_id=snapshot_id,
            initiator=initiator,
            local_state={},
            record_time={},
            channel_state={(src, dst): [] for src, dst in self.channels},
            marker_received={pid: set() for pid in self.pids},
            marker_arrival_time={},
            marker_arrival_order={},
            recording_channels={pid: set() for pid in self.pids},
        )
        self.snapshot = runtime

        self._log(f"INIT SNAPSHOT {snapshot_id} by P{initiator}")
        self._record_local_state(pid=initiator, first_marker_from=None)

    def _record_local_state(self, pid: int, first_marker_from: Optional[int]) -> None:
        runtime = self.snapshot
        if runtime is None:
            raise RuntimeError("snapshot not initialized")
        if pid in runtime.local_state:
            return

        runtime.local_state[pid] = self.processes[pid].balance
        runtime.record_time[pid] = self.time

        incoming = set(self.in_neighbors[pid])
        if first_marker_from is None:
            runtime.recording_channels[pid] = set(incoming)
            self._log(
                f"RECORD LOCAL STATE {runtime.snapshot_id}: P{pid} balance={runtime.local_state[pid]}, "
                f"open_record_channels={sorted(runtime.recording_channels[pid])}"
            )
        else:
            runtime.marker_received[pid].add(first_marker_from)
            runtime.recording_channels[pid] = set(incoming - {first_marker_from})
            runtime.channel_state[(first_marker_from, pid)] = []
            self._log(
                f"RECORD LOCAL STATE {runtime.snapshot_id}: P{pid} balance={runtime.local_state[pid]} "
                f"on first marker from P{first_marker_from}; "
                f"open_record_channels={sorted(runtime.recording_channels[pid])}"
            )

        for dst in self.out_neighbors[pid]:
            self._send_marker(src=pid, dst=dst, snapshot_id=runtime.snapshot_id)

    def _receive_app(self, message: Message) -> None:
        if message.payload is None:
            raise RuntimeError("APP message missing payload")

        receiver = self.processes[message.dst]
        receiver.balance += message.payload.amount

        transfer = self.transfer_history.get(message.payload.transfer_id)
        if transfer is None:
            raise RuntimeError(f"unknown transfer: {message.payload.transfer_id}")
        transfer["receive_step"] = self.time
        transfer["receive_order"] = self.current_dispatch_order

        recorded = False
        runtime = self.snapshot
        if runtime is not None and message.dst in runtime.local_state:
            if message.src in runtime.recording_channels[message.dst]:
                runtime.channel_state[(message.src, message.dst)].append(message)
                recorded = True

        self._log(
            f"RECV APP {message.payload.transfer_id}: P{message.src}->P{message.dst}, "
            f"amount={message.payload.amount}, receiver_balance_now={receiver.balance}, "
            f"recorded_in_snapshot={recorded}"
        )

    def _receive_marker(self, message: Message) -> None:
        runtime = self.snapshot
        if runtime is None:
            raise RuntimeError("received marker without active snapshot")
        if message.snapshot_id != runtime.snapshot_id:
            raise RuntimeError(
                f"unexpected snapshot id: got {message.snapshot_id}, expected {runtime.snapshot_id}"
            )

        src, dst = message.src, message.dst
        runtime.marker_arrival_time[(src, dst)] = self.time
        runtime.marker_arrival_order[(src, dst)] = self.current_dispatch_order

        if dst not in runtime.local_state:
            self._log(f"RECV FIRST MARKER {runtime.snapshot_id}: P{src}->P{dst}")
            self._record_local_state(pid=dst, first_marker_from=src)
        else:
            runtime.marker_received[dst].add(src)
            runtime.recording_channels[dst].discard(src)
            self._log(
                f"RECV LATER MARKER {runtime.snapshot_id}: P{src}->P{dst}, "
                f"remaining_open_channels={sorted(runtime.recording_channels[dst])}"
            )

        self._update_snapshot_completion()

    def _update_snapshot_completion(self) -> None:
        runtime = self.snapshot
        if runtime is None:
            return
        if runtime.completed:
            return
        if len(runtime.local_state) < len(self.pids):
            return

        for pid in self.pids:
            incoming = set(self.in_neighbors[pid])
            if runtime.marker_received[pid] != incoming:
                return

        runtime.completed = True
        runtime.complete_step = self.time
        self._log(f"SNAPSHOT {runtime.snapshot_id} COMPLETED at t={self.time}")

    def _dispatch(self, message: Message) -> None:
        if message.kind == "APP":
            self._receive_app(message)
            return
        if message.kind == "MARKER":
            self._receive_marker(message)
            return
        raise RuntimeError(f"unknown message kind: {message.kind}")

    def deliver_due_messages(self) -> int:
        delivered = 0
        while self.pending and self.pending[0].deliver_step <= self.time:
            envelope = heappop(self.pending)
            delivered += 1
            self.dispatch_order += 1
            self.current_dispatch_order = self.dispatch_order
            self._dispatch(envelope.message)
        return delivered

    def run_scenario(self, max_steps: int = 20) -> None:
        """Run one deterministic scenario with exactly one distributed snapshot."""
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")

        self._log("SCENARIO START")

        for step in range(max_steps):
            self.time = step

            # Step 0: one application transfer starts before snapshot.
            if step == 0:
                self.send_app(src=2, dst=1, amount=30, delay=4, transfer_id="T1")

            # Step 1: process P0 initiates snapshot S1.
            if step == 1:
                self.initiate_snapshot(
                    initiator=0,
                    snapshot_id="S1",
                    marker_default_delay=1,
                    marker_delay_overrides={(0, 2): 4},
                )

            self.deliver_due_messages()

            if self.snapshot is not None and self.snapshot.completed and not self.pending and step >= 5:
                break
        else:
            raise RuntimeError("scenario did not finish in max_steps")

    def validate_snapshot(self) -> None:
        runtime = self.snapshot
        if runtime is None:
            raise RuntimeError("snapshot not started")
        if not runtime.completed:
            raise RuntimeError("snapshot not completed")

        local_total = int(np.sum(np.array(list(runtime.local_state.values()), dtype=int)))
        channel_total = 0
        recorded_transfer_ids: Set[str] = set()

        for _channel, messages in runtime.channel_state.items():
            for msg in messages:
                if msg.kind != "APP" or msg.payload is None:
                    continue
                channel_total += msg.payload.amount
                recorded_transfer_ids.add(msg.payload.transfer_id)

                info = self.transfer_history[msg.payload.transfer_id]
                src, dst = msg.src, msg.dst
                send_step = info["send_step"]
                recv_step = info["receive_step"]
                recv_order = info["receive_order"]
                src_record = runtime.record_time[src]
                dst_record = runtime.record_time[dst]
                marker_arrival = runtime.marker_arrival_time[(src, dst)]
                marker_arrival_order = runtime.marker_arrival_order[(src, dst)]

                if not (send_step < src_record):
                    raise AssertionError(
                        f"recorded transfer {msg.payload.transfer_id} violates send<record(src)"
                    )
                received_before_marker = (recv_step < marker_arrival) or (
                    recv_step == marker_arrival and recv_order < marker_arrival_order
                )
                if not (dst_record <= recv_step and received_before_marker):
                    raise AssertionError(
                        f"recorded transfer {msg.payload.transfer_id} violates "
                        f"record(dst)<=recv<marker_arrival"
                    )

        if local_total + channel_total != self.initial_total:
            raise AssertionError(
                f"conservation broken: local={local_total}, channel={channel_total}, "
                f"initial={self.initial_total}"
            )

        expected_recorded = {"T1"}
        if recorded_transfer_ids != expected_recorded:
            raise AssertionError(
                f"unexpected recorded transfers: {recorded_transfer_ids}, "
                f"expected {expected_recorded}"
            )

    def print_summary(self) -> None:
        runtime = self.snapshot
        if runtime is None:
            raise RuntimeError("snapshot not started")

        print("=== Chandy-Lamport Distributed Snapshot MVP ===")
        print("event log:")
        for line in self.logs:
            print(" ", line)

        print("\nfinal process balances:")
        for pid in self.pids:
            print(f"  P{pid}: balance={self.processes[pid].balance}")

        print("\nsnapshot local states (recorded balances):")
        for pid in self.pids:
            rec_t = runtime.record_time[pid]
            bal = runtime.local_state[pid]
            print(f"  P{pid}: recorded_at=t{rec_t}, balance={bal}")

        print("\nsnapshot channel states (in-transit APP messages):")
        non_empty = False
        for src, dst in sorted(self.channels):
            msgs = runtime.channel_state[(src, dst)]
            if not msgs:
                continue
            non_empty = True
            parts = [f"{m.payload.transfer_id}(amount={m.payload.amount})" for m in msgs if m.payload]
            print(f"  C{src}->{dst}: {', '.join(parts)}")
        if not non_empty:
            print("  (none)")

        local_total = int(np.sum(np.array(list(runtime.local_state.values()), dtype=int)))
        channel_total = int(
            np.sum(
                np.array(
                    [
                        msg.payload.amount
                        for msgs in runtime.channel_state.values()
                        for msg in msgs
                        if msg.payload is not None
                    ],
                    dtype=int,
                )
            )
        )
        print("\nconservation check:")
        print(
            f"  local_total({local_total}) + channel_total({channel_total}) "
            f"= {local_total + channel_total}"
        )
        print(f"  initial_total = {self.initial_total}")
        print("All checks passed.")


def run_demo() -> None:
    """Run deterministic non-interactive MVP scenario."""
    initial_balances = {0: 100, 1: 100, 2: 100}
    directed_channels = [
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 2),
        (2, 0),
        (2, 1),
    ]

    sim = DistributedSnapshotSimulator(
        initial_balances=initial_balances,
        directed_channels=directed_channels,
    )
    sim.run_scenario(max_steps=20)
    sim.validate_snapshot()
    sim.print_summary()


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()

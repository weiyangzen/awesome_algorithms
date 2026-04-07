"""Minimal runnable MVP for sliding window protocol (Selective Repeat ARQ)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd


@dataclass
class ProtocolConfig:
    total_packets: int
    window_size: int
    timeout_ticks: int
    propagation_delay: int = 1
    max_ticks: int = 100


@dataclass
class TickRecord:
    tick: int
    sender_base: int
    next_seq: int
    receiver_base: int
    inflight_data: int
    inflight_acks: int
    new_tx: str
    retrans_tx: str
    acked_this_tick: str
    cumulative_retransmissions: int
    note: str


def validate_inputs(
    config: ProtocolConfig,
    data_loss_on_first_tx: Iterable[int],
    ack_loss_on_first_emit: Iterable[int],
) -> Tuple[Set[int], Set[int]]:
    """Validate config and normalize deterministic loss sets."""
    if config.total_packets <= 0:
        raise ValueError("total_packets must be > 0")
    if config.window_size <= 0:
        raise ValueError("window_size must be > 0")
    if config.timeout_ticks <= 0:
        raise ValueError("timeout_ticks must be > 0")
    if config.propagation_delay <= 0:
        raise ValueError("propagation_delay must be > 0")
    if config.max_ticks <= 0:
        raise ValueError("max_ticks must be > 0")

    data_loss = {int(x) for x in data_loss_on_first_tx}
    ack_loss = {int(x) for x in ack_loss_on_first_emit}

    upper = config.total_packets - 1
    for seq in data_loss | ack_loss:
        if seq < 0 or seq > upper:
            raise ValueError(f"loss sequence number out of range: {seq}")

    return data_loss, ack_loss


def _emit_ack(
    seq: int,
    tick: int,
    delay: int,
    ack_emit_count: np.ndarray,
    ack_loss_on_first_emit: Set[int],
    inflight_acks: List[Tuple[int, int]],
    notes: List[str],
) -> None:
    """Emit ACK for one packet with deterministic first-emit loss model."""
    ack_emit_count[seq] += 1
    if ack_emit_count[seq] == 1 and seq in ack_loss_on_first_emit:
        notes.append(f"drop ACK[{seq}] first emit")
        return
    inflight_acks.append((tick + delay, seq))
    notes.append(f"send ACK[{seq}]")


def _send_data_packet(
    seq: int,
    tick: int,
    reason: str,
    delay: int,
    data_attempt_count: np.ndarray,
    send_time: Dict[int, int],
    data_loss_on_first_tx: Set[int],
    inflight_data: List[Tuple[int, int]],
    notes: List[str],
) -> None:
    """Transmit one data packet with deterministic first-transmission loss."""
    data_attempt_count[seq] += 1
    send_time[seq] = tick
    attempt = int(data_attempt_count[seq])

    if attempt == 1 and seq in data_loss_on_first_tx:
        notes.append(f"drop DATA[{seq}] first tx")
        return

    inflight_data.append((tick + delay, seq))
    notes.append(f"{reason} DATA[{seq}] attempt={attempt}")


def simulate_sliding_window_sr(
    config: ProtocolConfig,
    data_loss_on_first_tx: Iterable[int],
    ack_loss_on_first_emit: Iterable[int],
) -> Tuple[List[TickRecord], np.ndarray, np.ndarray, np.ndarray]:
    """Simulate Selective Repeat ARQ in discrete ticks.

    Returns:
        records: timeline records
        sender_base_series: sender base after each tick
        receiver_base_series: receiver base after each tick
        retrans_series: cumulative retransmissions after each tick
    """
    data_loss, ack_loss = validate_inputs(
        config=config,
        data_loss_on_first_tx=data_loss_on_first_tx,
        ack_loss_on_first_emit=ack_loss_on_first_emit,
    )

    n = config.total_packets
    w = config.window_size
    timeout = config.timeout_ticks
    delay = config.propagation_delay

    acked = np.zeros(n, dtype=bool)
    received = np.zeros(n, dtype=bool)
    data_attempt_count = np.zeros(n, dtype=int)
    ack_emit_count = np.zeros(n, dtype=int)

    sender_base = 0
    next_seq = 0
    receiver_base = 0
    cumulative_retransmissions = 0

    send_time: Dict[int, int] = {}
    inflight_data: List[Tuple[int, int]] = []
    inflight_acks: List[Tuple[int, int]] = []
    records: List[TickRecord] = []

    for tick in range(config.max_ticks):
        notes: List[str] = []
        new_tx: List[int] = []
        retrans_tx: List[int] = []
        acked_this_tick: List[int] = []

        arrived_data = [item for item in inflight_data if item[0] == tick]
        inflight_data = [item for item in inflight_data if item[0] != tick]

        for _, seq in arrived_data:
            if seq < receiver_base:
                notes.append(f"recv old DATA[{seq}] below rcv_base")
                _emit_ack(
                    seq=seq,
                    tick=tick,
                    delay=delay,
                    ack_emit_count=ack_emit_count,
                    ack_loss_on_first_emit=ack_loss,
                    inflight_acks=inflight_acks,
                    notes=notes,
                )
                continue

            if seq >= min(receiver_base + w, n):
                notes.append(f"discard out-window DATA[{seq}]")
                continue

            if not received[seq]:
                received[seq] = True
                notes.append(f"recv DATA[{seq}]")
            else:
                notes.append(f"recv DUP DATA[{seq}]")

            _emit_ack(
                seq=seq,
                tick=tick,
                delay=delay,
                ack_emit_count=ack_emit_count,
                ack_loss_on_first_emit=ack_loss,
                inflight_acks=inflight_acks,
                notes=notes,
            )

            while receiver_base < n and received[receiver_base]:
                receiver_base += 1

        arrived_acks = [item for item in inflight_acks if item[0] == tick]
        inflight_acks = [item for item in inflight_acks if item[0] != tick]

        for _, ack_seq in arrived_acks:
            if 0 <= ack_seq < n and not acked[ack_seq]:
                acked[ack_seq] = True
                acked_this_tick.append(ack_seq)
                notes.append(f"ack DATA[{ack_seq}]")

        while sender_base < n and acked[sender_base]:
            sender_base += 1

        if sender_base >= n and not inflight_data and not inflight_acks:
            records.append(
                TickRecord(
                    tick=tick,
                    sender_base=sender_base,
                    next_seq=next_seq,
                    receiver_base=receiver_base,
                    inflight_data=0,
                    inflight_acks=0,
                    new_tx="-",
                    retrans_tx="-",
                    acked_this_tick="-",
                    cumulative_retransmissions=cumulative_retransmissions,
                    note="all packets acked; finish",
                )
            )
            break

        send_window_end = min(sender_base + w, n)
        for seq in range(sender_base, send_window_end):
            if not acked[seq] and seq in send_time and (tick - send_time[seq]) >= timeout:
                retrans_tx.append(seq)
                cumulative_retransmissions += 1
                _send_data_packet(
                    seq=seq,
                    tick=tick,
                    reason="RTX",
                    delay=delay,
                    data_attempt_count=data_attempt_count,
                    send_time=send_time,
                    data_loss_on_first_tx=data_loss,
                    inflight_data=inflight_data,
                    notes=notes,
                )

        send_window_end = min(sender_base + w, n)
        while next_seq < send_window_end:
            seq = next_seq
            if seq not in send_time:
                new_tx.append(seq)
                _send_data_packet(
                    seq=seq,
                    tick=tick,
                    reason="SEND",
                    delay=delay,
                    data_attempt_count=data_attempt_count,
                    send_time=send_time,
                    data_loss_on_first_tx=data_loss,
                    inflight_data=inflight_data,
                    notes=notes,
                )
            next_seq += 1

        records.append(
            TickRecord(
                tick=tick,
                sender_base=sender_base,
                next_seq=next_seq,
                receiver_base=receiver_base,
                inflight_data=len(inflight_data),
                inflight_acks=len(inflight_acks),
                new_tx=",".join(str(x) for x in new_tx) if new_tx else "-",
                retrans_tx=",".join(str(x) for x in retrans_tx) if retrans_tx else "-",
                acked_this_tick=",".join(str(x) for x in sorted(acked_this_tick)) if acked_this_tick else "-",
                cumulative_retransmissions=cumulative_retransmissions,
                note=" | ".join(notes) if notes else "-",
            )
        )
    else:
        raise RuntimeError("simulation did not finish within max_ticks")

    sender_base_series = np.array([r.sender_base for r in records], dtype=int)
    receiver_base_series = np.array([r.receiver_base for r in records], dtype=int)
    retrans_series = np.array([r.cumulative_retransmissions for r in records], dtype=int)

    return records, sender_base_series, receiver_base_series, retrans_series


def records_to_dataframe(records: Sequence[TickRecord]) -> pd.DataFrame:
    """Convert timeline records to a readable DataFrame."""
    return pd.DataFrame(
        {
            "tick": [r.tick for r in records],
            "sender_base": [r.sender_base for r in records],
            "next_seq": [r.next_seq for r in records],
            "receiver_base": [r.receiver_base for r in records],
            "inflight_data": [r.inflight_data for r in records],
            "inflight_acks": [r.inflight_acks for r in records],
            "new_tx": [r.new_tx for r in records],
            "retrans_tx": [r.retrans_tx for r in records],
            "acked_this_tick": [r.acked_this_tick for r in records],
            "cum_retx": [r.cumulative_retransmissions for r in records],
            "note": [r.note for r in records],
        }
    )


def run_demo() -> None:
    """Run deterministic SR sliding-window demo and assertions."""
    print("=== Sliding Window Protocol MVP (Selective Repeat ARQ) ===")

    config = ProtocolConfig(
        total_packets=8,
        window_size=4,
        timeout_ticks=3,
        propagation_delay=1,
        max_ticks=80,
    )

    data_loss_on_first_tx = {2, 6}
    ack_loss_on_first_emit = {4}

    records, sender_base_series, receiver_base_series, retrans_series = simulate_sliding_window_sr(
        config=config,
        data_loss_on_first_tx=data_loss_on_first_tx,
        ack_loss_on_first_emit=ack_loss_on_first_emit,
    )

    if sender_base_series[-1] != config.total_packets:
        raise AssertionError("sender base must reach total_packets")
    if receiver_base_series[-1] != config.total_packets:
        raise AssertionError("receiver base must reach total_packets")

    expected_sender_base = np.array([0, 0, 2, 2, 2, 4, 4, 6, 6, 6, 8], dtype=int)
    expected_receiver_base = np.array([0, 2, 2, 2, 6, 6, 6, 6, 6, 8, 8], dtype=int)
    expected_retrans = np.array([0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3], dtype=int)

    if not np.array_equal(sender_base_series, expected_sender_base):
        raise AssertionError(f"unexpected sender_base series: {sender_base_series}")
    if not np.array_equal(receiver_base_series, expected_receiver_base):
        raise AssertionError(f"unexpected receiver_base series: {receiver_base_series}")
    if not np.array_equal(retrans_series, expected_retrans):
        raise AssertionError(f"unexpected retransmission series: {retrans_series}")

    if int(retrans_series[-1]) != 3:
        raise AssertionError("expected exactly 3 retransmissions (2 data losses + 1 ACK-loss-triggered)")

    retrans_ticks = [r.tick for r in records if r.retrans_tx != "-"]
    if retrans_ticks != [3, 5, 8]:
        raise AssertionError(f"unexpected retransmission ticks: {retrans_ticks}")

    table = records_to_dataframe(records)
    print("Tick timeline:")
    with pd.option_context("display.max_colwidth", 120, "display.width", 220):
        print(table.to_string(index=False))

    print("sender_base series:", sender_base_series.tolist())
    print("receiver_base series:", receiver_base_series.tolist())
    print("cumulative_retransmissions series:", retrans_series.tolist())
    print("All checks passed.")


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()

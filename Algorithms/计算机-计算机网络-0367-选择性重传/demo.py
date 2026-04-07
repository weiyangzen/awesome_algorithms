"""Minimal runnable MVP for Selective Repeat (SR) ARQ."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2
from typing import Dict, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np


@dataclass
class SenderFrame:
    packet_id: int
    seq: int
    payload: int
    attempt: int
    last_send_time: int


@dataclass
class TxRecord:
    time: int
    packet_id: int
    seq: int
    attempt: int
    kind: str
    dropped: bool


@dataclass
class RxRecord:
    time: int
    packet_id: int
    seq: int
    attempt: int
    status: str
    delivered_now: int
    receiver_base_after: int


@dataclass
class AckRecord:
    time_emit: int
    time_arrive: int
    packet_id: int
    seq: int
    ack_index: int
    dropped: bool
    matched_sender: bool


@dataclass
class SimulationResult:
    payloads_sent: List[int]
    delivered_payloads: List[int]
    attempts_per_packet: np.ndarray
    tx_records: List[TxRecord]
    rx_records: List[RxRecord]
    ack_records: List[AckRecord]
    stats: Dict[str, float]


def validate_parameters(
    total_packets: int,
    window_size: int,
    seq_modulus: int,
    timeout: int,
    data_delay: int,
    ack_delay: int,
    max_ticks: int,
) -> None:
    """Validate SR simulation parameters."""
    if total_packets <= 0:
        raise ValueError("total_packets must be positive")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if seq_modulus <= 1:
        raise ValueError("seq_modulus must be > 1")
    if seq_modulus < 2 * window_size:
        raise ValueError("Selective Repeat requires seq_modulus >= 2 * window_size")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if data_delay <= 0 or ack_delay <= 0:
        raise ValueError("data_delay and ack_delay must be positive")
    if max_ticks <= 0:
        raise ValueError("max_ticks must be positive")


def payload_for_packet(packet_id: int) -> int:
    """Deterministically generate one-byte payload for packet_id."""
    return int((packet_id * 37 + 13) % 256)


def seq_bits(seq_modulus: int) -> int:
    """Return the minimum number of bits needed to represent sequence space."""
    return int(ceil(log2(seq_modulus)))


def transmit_frame(
    now: int,
    frame: SenderFrame,
    kind: str,
    data_drop_plan: Mapping[Tuple[int, int], bool],
    data_delay: int,
    data_events: MutableMapping[int, List[Tuple[int, int, int, int]]],
    attempts_per_packet: np.ndarray,
    tx_records: List[TxRecord],
) -> bool:
    """Transmit (or retransmit) one frame and schedule data arrival if not dropped."""
    attempts_per_packet[frame.packet_id] += 1
    dropped = bool(data_drop_plan.get((frame.packet_id, frame.attempt), False))

    tx_records.append(
        TxRecord(
            time=now,
            packet_id=frame.packet_id,
            seq=frame.seq,
            attempt=frame.attempt,
            kind=kind,
            dropped=dropped,
        )
    )

    if not dropped:
        data_events.setdefault(now + data_delay, []).append(
            (frame.packet_id, frame.seq, frame.payload, frame.attempt)
        )

    return dropped


def simulate_selective_repeat(
    total_packets: int,
    window_size: int,
    seq_modulus: int,
    timeout: int,
    data_delay: int,
    ack_delay: int,
    data_drop_plan: Mapping[Tuple[int, int], bool],
    ack_drop_plan: Mapping[Tuple[int, int], bool],
    max_ticks: int = 200,
) -> SimulationResult:
    """Simulate Selective Repeat ARQ with deterministic data/ack drops."""
    validate_parameters(
        total_packets=total_packets,
        window_size=window_size,
        seq_modulus=seq_modulus,
        timeout=timeout,
        data_delay=data_delay,
        ack_delay=ack_delay,
        max_ticks=max_ticks,
    )

    payloads_sent = [payload_for_packet(i) for i in range(total_packets)]

    sender_base = 0
    next_packet = 0
    sender_acked = np.zeros(total_packets, dtype=bool)
    attempts_per_packet = np.zeros(total_packets, dtype=int)
    outstanding: Dict[int, SenderFrame] = {}

    receiver_base = 0
    receiver_buffer: Dict[int, int] = {}
    delivered_payloads: List[int] = []

    data_events: Dict[int, List[Tuple[int, int, int, int]]] = {}
    ack_events: Dict[int, List[Tuple[int, int, int, int]]] = {}
    ack_emit_count: Dict[int, int] = {}

    tx_records: List[TxRecord] = []
    rx_records: List[RxRecord] = []
    ack_records: List[AckRecord] = []

    data_drops = 0
    ack_drops = 0
    max_receiver_buffer = 0
    completion_time = -1

    def slide_sender_base() -> None:
        nonlocal sender_base
        while sender_base < total_packets and sender_acked[sender_base]:
            sender_base += 1

    for now in range(max_ticks + 1):
        incoming_data = data_events.pop(now, [])
        for packet_id, seq, payload, attempt in incoming_data:
            send_ack = False
            delivered_now = 0

            if packet_id < receiver_base:
                status = "old_duplicate"
                send_ack = True
            elif packet_id >= receiver_base + window_size:
                status = "outside_window"
                send_ack = False
            else:
                if packet_id in receiver_buffer:
                    status = "buffered_duplicate"
                else:
                    receiver_buffer[packet_id] = payload
                    status = "buffered_new"

                while receiver_base in receiver_buffer:
                    delivered_payloads.append(receiver_buffer.pop(receiver_base))
                    receiver_base += 1
                    delivered_now += 1
                send_ack = True

            max_receiver_buffer = max(max_receiver_buffer, len(receiver_buffer))
            rx_records.append(
                RxRecord(
                    time=now,
                    packet_id=packet_id,
                    seq=seq,
                    attempt=attempt,
                    status=status,
                    delivered_now=delivered_now,
                    receiver_base_after=receiver_base,
                )
            )

            if send_ack:
                ack_index = ack_emit_count.get(packet_id, 0) + 1
                ack_emit_count[packet_id] = ack_index
                dropped = bool(ack_drop_plan.get((packet_id, ack_index), False))

                record_idx = len(ack_records)
                ack_records.append(
                    AckRecord(
                        time_emit=now,
                        time_arrive=now + ack_delay,
                        packet_id=packet_id,
                        seq=seq,
                        ack_index=ack_index,
                        dropped=dropped,
                        matched_sender=False,
                    )
                )

                if dropped:
                    ack_drops += 1
                else:
                    ack_events.setdefault(now + ack_delay, []).append(
                        (packet_id, seq, ack_index, record_idx)
                    )

        incoming_ack = ack_events.pop(now, [])
        for packet_id, seq, _ack_index, record_idx in incoming_ack:
            matched = False
            frame = outstanding.pop(seq, None)
            if frame is not None:
                if frame.packet_id == packet_id:
                    sender_acked[packet_id] = True
                    matched = True
                else:
                    # Sequence number reused with delayed stale ACK: keep current frame untouched.
                    outstanding[seq] = frame
                    matched = False
            ack_records[record_idx].matched_sender = matched
            slide_sender_base()

        for seq in sorted(outstanding.keys()):
            frame = outstanding[seq]
            if now - frame.last_send_time >= timeout:
                frame.attempt += 1
                frame.last_send_time = now
                dropped = transmit_frame(
                    now=now,
                    frame=frame,
                    kind="retransmit",
                    data_drop_plan=data_drop_plan,
                    data_delay=data_delay,
                    data_events=data_events,
                    attempts_per_packet=attempts_per_packet,
                    tx_records=tx_records,
                )
                if dropped:
                    data_drops += 1

        while next_packet < total_packets and next_packet < sender_base + window_size:
            seq = next_packet % seq_modulus
            if seq in outstanding:
                break

            frame = SenderFrame(
                packet_id=next_packet,
                seq=seq,
                payload=payloads_sent[next_packet],
                attempt=1,
                last_send_time=now,
            )
            outstanding[seq] = frame
            dropped = transmit_frame(
                now=now,
                frame=frame,
                kind="new",
                data_drop_plan=data_drop_plan,
                data_delay=data_delay,
                data_events=data_events,
                attempts_per_packet=attempts_per_packet,
                tx_records=tx_records,
            )
            if dropped:
                data_drops += 1
            next_packet += 1

        if (
            sender_base >= total_packets
            and not outstanding
            and not data_events
            and not ack_events
        ):
            completion_time = now
            break

    if completion_time < 0:
        raise RuntimeError("simulation did not finish within max_ticks")

    total_tx = len(tx_records)
    frame_bits = 8 + seq_bits(seq_modulus) + 8  # payload + seq + checksum placeholder
    info_bits = total_packets * 8
    sent_bits = total_tx * frame_bits

    stats = {
        "total_packets": float(total_packets),
        "total_tx": float(total_tx),
        "retransmissions": float(total_tx - total_packets),
        "data_drops": float(data_drops),
        "ack_drops": float(ack_drops),
        "completion_time": float(completion_time),
        "max_receiver_buffer": float(max_receiver_buffer),
        "goodput": float(info_bits / sent_bits),
    }

    return SimulationResult(
        payloads_sent=payloads_sent,
        delivered_payloads=delivered_payloads,
        attempts_per_packet=attempts_per_packet,
        tx_records=tx_records,
        rx_records=rx_records,
        ack_records=ack_records,
        stats=stats,
    )


def run_demo() -> None:
    """Run one deterministic SR-ARQ scenario with assertions."""
    print("=== Selective Repeat ARQ MVP Demo ===")

    total_packets = 8
    window_size = 4
    seq_modulus = 8
    timeout = 4
    data_delay = 1
    ack_delay = 1

    # (packet_id, transmission_attempt) -> drop data frame?
    data_drop_plan = {
        (1, 1): True,
        (4, 1): True,
    }

    # (packet_id, ack_emission_index) -> drop ACK?
    ack_drop_plan = {
        (2, 1): True,
        (5, 1): True,
    }

    result = simulate_selective_repeat(
        total_packets=total_packets,
        window_size=window_size,
        seq_modulus=seq_modulus,
        timeout=timeout,
        data_delay=data_delay,
        ack_delay=ack_delay,
        data_drop_plan=data_drop_plan,
        ack_drop_plan=ack_drop_plan,
        max_ticks=200,
    )

    if result.delivered_payloads != result.payloads_sent:
        raise AssertionError(
            "receiver delivered payload sequence mismatch\n"
            f"expected={result.payloads_sent}\n"
            f"actual={result.delivered_payloads}"
        )

    expected_attempts = np.array([1, 2, 2, 1, 2, 2, 1, 1], dtype=int)
    if not np.array_equal(result.attempts_per_packet, expected_attempts):
        raise AssertionError(
            "unexpected attempts_per_packet\n"
            f"expected={expected_attempts.tolist()}\n"
            f"actual={result.attempts_per_packet.tolist()}"
        )

    retransmitted_packet_ids = np.where(result.attempts_per_packet > 1)[0].tolist()
    if retransmitted_packet_ids != [1, 2, 4, 5]:
        raise AssertionError(
            f"unexpected retransmitted packet ids: {retransmitted_packet_ids}"
        )

    if int(result.stats["retransmissions"]) != 4:
        raise AssertionError(f"unexpected retransmissions: {result.stats['retransmissions']}")
    if int(result.stats["data_drops"]) != 2:
        raise AssertionError(f"unexpected data_drops: {result.stats['data_drops']}")
    if int(result.stats["ack_drops"]) != 2:
        raise AssertionError(f"unexpected ack_drops: {result.stats['ack_drops']}")

    print(
        "config:",
        {
            "total_packets": total_packets,
            "window_size": window_size,
            "seq_modulus": seq_modulus,
            "timeout": timeout,
            "data_delay": data_delay,
            "ack_delay": ack_delay,
        },
    )
    print("attempts_per_packet:", result.attempts_per_packet.tolist())
    print("sender tx timeline:")
    for rec in result.tx_records:
        print(
            f"  t={rec.time:02d} pkt={rec.packet_id} seq={rec.seq} "
            f"attempt={rec.attempt} kind={rec.kind} dropped={rec.dropped}"
        )

    matched_acks = sum(1 for rec in result.ack_records if rec.matched_sender)
    print(
        "stats:",
        {
            "total_tx": int(result.stats["total_tx"]),
            "retransmissions": int(result.stats["retransmissions"]),
            "data_drops": int(result.stats["data_drops"]),
            "ack_drops": int(result.stats["ack_drops"]),
            "matched_acks": matched_acks,
            "completion_time": int(result.stats["completion_time"]),
            "max_receiver_buffer": int(result.stats["max_receiver_buffer"]),
            "goodput": round(result.stats["goodput"], 4),
        },
    )
    print("All checks passed.")


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()

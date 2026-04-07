"""Minimal runnable MVP for Stop-and-Wait ARQ."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, MutableMapping, Tuple

import numpy as np


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
    expected_seq_after: int


@dataclass
class AckRecord:
    time_emit: int
    time_arrive: int
    packet_id: int
    ack_seq: int
    ack_index: int
    dropped: bool
    accepted_by_sender: bool


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
    timeout: int,
    data_delay: int,
    ack_delay: int,
    max_ticks: int,
) -> None:
    """Validate Stop-and-Wait simulation parameters."""
    if total_packets <= 0:
        raise ValueError("total_packets must be positive")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if data_delay <= 0 or ack_delay <= 0:
        raise ValueError("data_delay and ack_delay must be positive")
    if max_ticks <= 0:
        raise ValueError("max_ticks must be positive")


def payload_for_packet(packet_id: int) -> int:
    """Deterministically generate a one-byte payload."""
    return int((packet_id * 29 + 7) % 256)


def transmit_packet(
    now: int,
    packet_id: int,
    seq: int,
    payload: int,
    attempt: int,
    kind: str,
    data_drop_plan: Mapping[Tuple[int, int], bool],
    data_delay: int,
    data_events: MutableMapping[int, List[Tuple[int, int, int, int]]],
    attempts_per_packet: np.ndarray,
    tx_records: List[TxRecord],
) -> bool:
    """Transmit/retransmit one packet and schedule data arrival if not dropped."""
    attempts_per_packet[packet_id] += 1
    dropped = bool(data_drop_plan.get((packet_id, attempt), False))

    tx_records.append(
        TxRecord(
            time=now,
            packet_id=packet_id,
            seq=seq,
            attempt=attempt,
            kind=kind,
            dropped=dropped,
        )
    )

    if not dropped:
        data_events.setdefault(now + data_delay, []).append((packet_id, seq, payload, attempt))

    return dropped


def simulate_stop_and_wait(
    total_packets: int,
    timeout: int,
    data_delay: int,
    ack_delay: int,
    data_drop_plan: Mapping[Tuple[int, int], bool],
    ack_drop_plan: Mapping[Tuple[int, int], bool],
    max_ticks: int = 200,
) -> SimulationResult:
    """Simulate Stop-and-Wait ARQ with deterministic data/ack losses."""
    validate_parameters(
        total_packets=total_packets,
        timeout=timeout,
        data_delay=data_delay,
        ack_delay=ack_delay,
        max_ticks=max_ticks,
    )

    payloads_sent = [payload_for_packet(i) for i in range(total_packets)]
    delivered_payloads: List[int] = []

    attempts_per_packet = np.zeros(total_packets, dtype=int)
    tx_records: List[TxRecord] = []
    rx_records: List[RxRecord] = []
    ack_records: List[AckRecord] = []

    data_events: Dict[int, List[Tuple[int, int, int, int]]] = {}
    ack_events: Dict[int, List[Tuple[int, int, int]]] = {}
    ack_emit_count: Dict[int, int] = {}

    sender_packet = 0
    sender_seq = 0
    waiting_ack = False
    in_flight_attempt = 0
    in_flight_last_send_time = -1

    receiver_expected_seq = 0

    data_drops = 0
    ack_drops = 0
    timeout_retransmissions = 0
    completion_time = -1

    for now in range(max_ticks + 1):
        if not waiting_ack and sender_packet < total_packets:
            in_flight_attempt = int(attempts_per_packet[sender_packet] + 1)
            was_dropped = transmit_packet(
                now=now,
                packet_id=sender_packet,
                seq=sender_seq,
                payload=payloads_sent[sender_packet],
                attempt=in_flight_attempt,
                kind="new",
                data_drop_plan=data_drop_plan,
                data_delay=data_delay,
                data_events=data_events,
                attempts_per_packet=attempts_per_packet,
                tx_records=tx_records,
            )
            if was_dropped:
                data_drops += 1
            waiting_ack = True
            in_flight_last_send_time = now

        incoming_data = data_events.pop(now, [])
        for packet_id, seq, payload, attempt in incoming_data:
            delivered_now = 0
            if seq == receiver_expected_seq:
                status = "accepted_new"
                delivered_payloads.append(payload)
                delivered_now = 1
                receiver_expected_seq ^= 1
            else:
                status = "duplicate_or_unexpected"

            rx_records.append(
                RxRecord(
                    time=now,
                    packet_id=packet_id,
                    seq=seq,
                    attempt=attempt,
                    status=status,
                    delivered_now=delivered_now,
                    expected_seq_after=receiver_expected_seq,
                )
            )

            # ACK carries the sequence bit of the last in-order accepted frame.
            ack_seq = receiver_expected_seq ^ 1
            ack_index = ack_emit_count.get(packet_id, 0) + 1
            ack_emit_count[packet_id] = ack_index
            ack_dropped = bool(ack_drop_plan.get((packet_id, ack_index), False))

            ack_records.append(
                AckRecord(
                    time_emit=now,
                    time_arrive=now + ack_delay,
                    packet_id=packet_id,
                    ack_seq=ack_seq,
                    ack_index=ack_index,
                    dropped=ack_dropped,
                    accepted_by_sender=False,
                )
            )

            if ack_dropped:
                ack_drops += 1
            else:
                ack_events.setdefault(now + ack_delay, []).append((packet_id, ack_seq, ack_index))

        incoming_acks = ack_events.pop(now, [])
        for packet_id, ack_seq, ack_index in incoming_acks:
            accepted = (
                waiting_ack
                and packet_id == sender_packet
                and ack_seq == sender_seq
            )
            for record in ack_records:
                if (
                    record.packet_id == packet_id
                    and record.ack_index == ack_index
                    and record.time_arrive == now
                ):
                    record.accepted_by_sender = accepted
                    break

            if accepted:
                waiting_ack = False
                sender_packet += 1
                sender_seq ^= 1

        if waiting_ack and (now - in_flight_last_send_time >= timeout):
            timeout_retransmissions += 1
            in_flight_attempt = int(attempts_per_packet[sender_packet] + 1)
            was_dropped = transmit_packet(
                now=now,
                packet_id=sender_packet,
                seq=sender_seq,
                payload=payloads_sent[sender_packet],
                attempt=in_flight_attempt,
                kind="retransmit",
                data_drop_plan=data_drop_plan,
                data_delay=data_delay,
                data_events=data_events,
                attempts_per_packet=attempts_per_packet,
                tx_records=tx_records,
            )
            if was_dropped:
                data_drops += 1
            in_flight_last_send_time = now

        if sender_packet >= total_packets and not waiting_ack:
            completion_time = now
            break

    if completion_time < 0:
        raise RuntimeError("Simulation did not complete within max_ticks")

    total_tx = int(attempts_per_packet.sum())
    retransmissions = int(total_tx - total_packets)
    goodput = float(total_packets / total_tx)

    return SimulationResult(
        payloads_sent=payloads_sent,
        delivered_payloads=delivered_payloads,
        attempts_per_packet=attempts_per_packet,
        tx_records=tx_records,
        rx_records=rx_records,
        ack_records=ack_records,
        stats={
            "total_packets": float(total_packets),
            "total_tx": float(total_tx),
            "retransmissions": float(retransmissions),
            "timeout_retransmissions": float(timeout_retransmissions),
            "data_drops": float(data_drops),
            "ack_drops": float(ack_drops),
            "completion_time": float(completion_time),
            "goodput": goodput,
        },
    )


def run_demo() -> None:
    """Run deterministic Stop-and-Wait scenario and validate key invariants."""
    total_packets = 6
    timeout = 4
    data_delay = 1
    ack_delay = 1

    data_drop_plan = {
        (1, 1): True,
        (4, 1): True,
    }
    ack_drop_plan = {
        (2, 1): True,
        (4, 1): True,
    }

    result = simulate_stop_and_wait(
        total_packets=total_packets,
        timeout=timeout,
        data_delay=data_delay,
        ack_delay=ack_delay,
        data_drop_plan=data_drop_plan,
        ack_drop_plan=ack_drop_plan,
        max_ticks=300,
    )

    expected_attempts = np.array([1, 2, 2, 1, 3, 1], dtype=int)
    assert result.delivered_payloads == result.payloads_sent, "payload sequence mismatch"
    assert np.array_equal(result.attempts_per_packet, expected_attempts), "attempt vector mismatch"
    assert int(result.stats["retransmissions"]) == 4, "unexpected retransmission count"
    assert int(result.stats["data_drops"]) == 2, "unexpected data drop count"
    assert int(result.stats["ack_drops"]) == 2, "unexpected ack drop count"

    print("=== Stop-and-Wait ARQ Demo ===")
    print(
        f"config: total_packets={total_packets}, timeout={timeout}, "
        f"data_delay={data_delay}, ack_delay={ack_delay}"
    )
    print(f"attempts_per_packet: {result.attempts_per_packet.tolist()}")
    print(f"delivered_payloads: {result.delivered_payloads}")

    print("\n[Sender TX timeline]")
    for rec in result.tx_records:
        print(
            f"t={rec.time:>3} pkt={rec.packet_id} seq={rec.seq} "
            f"attempt={rec.attempt} kind={rec.kind:<10} dropped={rec.dropped}"
        )

    print("\n[Receiver RX timeline]")
    for rec in result.rx_records:
        print(
            f"t={rec.time:>3} pkt={rec.packet_id} seq={rec.seq} attempt={rec.attempt} "
            f"status={rec.status:<23} delivered_now={rec.delivered_now} "
            f"expected_seq_after={rec.expected_seq_after}"
        )

    print("\n[ACK timeline]")
    for rec in result.ack_records:
        print(
            f"emit={rec.time_emit:>3} arrive={rec.time_arrive:>3} pkt={rec.packet_id} "
            f"ack_seq={rec.ack_seq} ack_idx={rec.ack_index} dropped={rec.dropped} "
            f"accepted_by_sender={rec.accepted_by_sender}"
        )

    print("\n[Stats]")
    for key, value in result.stats.items():
        if key == "goodput":
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {int(value)}")
    print("All checks passed.")


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()

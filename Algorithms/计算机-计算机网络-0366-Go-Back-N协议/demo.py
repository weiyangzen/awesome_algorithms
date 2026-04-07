"""Minimal runnable MVP for the Go-Back-N ARQ protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class DataPacket:
    packet_id: int
    seq: int


@dataclass(frozen=True)
class AckPacket:
    ack_id: int
    ack_seq: int


@dataclass
class SenderState:
    base: int
    next_packet: int
    timer_start: Optional[int]


@dataclass
class ReceiverState:
    expected_packet: int
    expected_seq: int
    delivered: List[int]


@dataclass
class SlotRecord:
    slot: int
    base_before: int
    base_after: int
    next_before: int
    next_after: int
    new_sent: int
    retransmitted: int
    delivered_now: int
    discarded_now: int
    ack_received: int
    timeout_fired: bool


@dataclass
class SimulationResult:
    records: List[SlotRecord]
    delivered: List[int]
    total_data_tx: int
    total_ack_tx: int
    retransmissions: int
    data_drops: int
    ack_drops: int
    total_slots: int


def validate_params(
    total_packets: int,
    window_size: int,
    seq_mod: int,
    timeout_slots: int,
    propagation_delay: int,
    data_loss_prob: float,
    ack_loss_prob: float,
    max_slots: int,
) -> None:
    """Validate simulation parameters."""
    if total_packets <= 0:
        raise ValueError("total_packets must be positive")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if seq_mod <= window_size:
        raise ValueError("seq_mod must be greater than window_size for Go-Back-N")
    if timeout_slots <= 0:
        raise ValueError("timeout_slots must be positive")
    if propagation_delay <= 0:
        raise ValueError("propagation_delay must be positive")
    if max_slots <= 0:
        raise ValueError("max_slots must be positive")
    if not (0.0 <= data_loss_prob < 1.0):
        raise ValueError("data_loss_prob must be in [0, 1)")
    if not (0.0 <= ack_loss_prob < 1.0):
        raise ValueError("ack_loss_prob must be in [0, 1)")


def enqueue_data_packet(
    packet: DataPacket,
    arrival_slot: int,
    data_channel: List[Tuple[int, DataPacket]],
    data_loss_prob: float,
    rng: np.random.Generator,
) -> bool:
    """Try to put a data packet into the channel; return True if not dropped."""
    if rng.random() < data_loss_prob:
        return False
    data_channel.append((arrival_slot, packet))
    return True


def enqueue_ack_packet(
    ack: AckPacket,
    arrival_slot: int,
    ack_channel: List[Tuple[int, AckPacket]],
    ack_loss_prob: float,
    rng: np.random.Generator,
) -> bool:
    """Try to put an ACK packet into the channel; return True if not dropped."""
    if rng.random() < ack_loss_prob:
        return False
    ack_channel.append((arrival_slot, ack))
    return True


def split_arrivals_data(
    data_channel: Sequence[Tuple[int, DataPacket]],
    slot: int,
) -> Tuple[List[DataPacket], List[Tuple[int, DataPacket]]]:
    """Split data packets arriving at this slot and those still in flight."""
    arriving: List[DataPacket] = []
    pending: List[Tuple[int, DataPacket]] = []
    for arrive_at, packet in data_channel:
        if arrive_at == slot:
            arriving.append(packet)
        else:
            pending.append((arrive_at, packet))
    return arriving, pending


def split_arrivals_ack(
    ack_channel: Sequence[Tuple[int, AckPacket]],
    slot: int,
) -> Tuple[List[AckPacket], List[Tuple[int, AckPacket]]]:
    """Split ACK packets arriving at this slot and those still in flight."""
    arriving: List[AckPacket] = []
    pending: List[Tuple[int, AckPacket]] = []
    for arrive_at, ack in ack_channel:
        if arrive_at == slot:
            arriving.append(ack)
        else:
            pending.append((arrive_at, ack))
    return arriving, pending


def process_receiver(
    arriving_packets: Sequence[DataPacket],
    receiver: ReceiverState,
    seq_mod: int,
    ack_channel: List[Tuple[int, AckPacket]],
    slot: int,
    propagation_delay: int,
    ack_loss_prob: float,
    rng: np.random.Generator,
) -> Tuple[int, int, int, int]:
    """Process receiver behavior for this slot.

    Returns:
        delivered_now, discarded_now, ack_attempts, ack_drops
    """
    delivered_now = 0
    discarded_now = 0
    ack_attempts = 0
    ack_drops = 0

    for packet in arriving_packets:
        if packet.packet_id == receiver.expected_packet and packet.seq == receiver.expected_seq:
            receiver.delivered.append(packet.packet_id)
            receiver.expected_packet += 1
            receiver.expected_seq = (receiver.expected_seq + 1) % seq_mod
            delivered_now += 1
        else:
            discarded_now += 1

        # Receiver sends cumulative ACK for the last in-order packet.
        ack_id = receiver.expected_packet - 1
        ack_seq = ack_id % seq_mod
        ack = AckPacket(ack_id=ack_id, ack_seq=ack_seq)

        ack_attempts += 1
        kept = enqueue_ack_packet(
            ack=ack,
            arrival_slot=slot + propagation_delay,
            ack_channel=ack_channel,
            ack_loss_prob=ack_loss_prob,
            rng=rng,
        )
        if not kept:
            ack_drops += 1

    return delivered_now, discarded_now, ack_attempts, ack_drops


def process_sender_acks(
    arriving_acks: Sequence[AckPacket],
    sender: SenderState,
    slot: int,
) -> int:
    """Process cumulative ACKs at sender and return count of effective ACKs."""
    ack_received = 0
    for ack in arriving_acks:
        if ack.ack_id >= sender.base:
            sender.base = ack.ack_id + 1
            ack_received += 1
            if sender.base == sender.next_packet:
                sender.timer_start = None
            else:
                sender.timer_start = slot
    return ack_received


def simulate_go_back_n(
    total_packets: int,
    window_size: int,
    seq_mod: int,
    timeout_slots: int,
    propagation_delay: int,
    data_loss_prob: float,
    ack_loss_prob: float,
    seed: int,
    max_slots: int,
) -> SimulationResult:
    """Run a deterministic Go-Back-N simulation with lossy data/ACK channels."""
    validate_params(
        total_packets=total_packets,
        window_size=window_size,
        seq_mod=seq_mod,
        timeout_slots=timeout_slots,
        propagation_delay=propagation_delay,
        data_loss_prob=data_loss_prob,
        ack_loss_prob=ack_loss_prob,
        max_slots=max_slots,
    )

    rng = np.random.default_rng(seed)

    sender = SenderState(base=0, next_packet=0, timer_start=None)
    receiver = ReceiverState(expected_packet=0, expected_seq=0, delivered=[])

    data_channel: List[Tuple[int, DataPacket]] = []
    ack_channel: List[Tuple[int, AckPacket]] = []

    total_data_tx = 0
    total_ack_tx = 0
    retransmissions = 0
    data_drops = 0
    ack_drops = 0

    records: List[SlotRecord] = []

    for slot in range(max_slots):
        if sender.base >= total_packets:
            break

        base_before = sender.base
        next_before = sender.next_packet
        new_sent = 0
        retransmitted = 0
        timeout_fired = False

        # 1) Send new packets while window allows.
        while sender.next_packet < total_packets and sender.next_packet < sender.base + window_size:
            pid = sender.next_packet
            packet = DataPacket(packet_id=pid, seq=pid % seq_mod)
            sender.next_packet += 1

            total_data_tx += 1
            new_sent += 1
            if sender.timer_start is None:
                sender.timer_start = slot

            kept = enqueue_data_packet(
                packet=packet,
                arrival_slot=slot + propagation_delay,
                data_channel=data_channel,
                data_loss_prob=data_loss_prob,
                rng=rng,
            )
            if not kept:
                data_drops += 1

        # 2) Receiver gets data that arrives this slot.
        arriving_data, data_channel = split_arrivals_data(data_channel, slot)

        delivered_now, discarded_now, ack_attempts, ack_dropped_now = process_receiver(
            arriving_packets=arriving_data,
            receiver=receiver,
            seq_mod=seq_mod,
            ack_channel=ack_channel,
            slot=slot,
            propagation_delay=propagation_delay,
            ack_loss_prob=ack_loss_prob,
            rng=rng,
        )

        total_ack_tx += ack_attempts
        ack_drops += ack_dropped_now

        # 3) Sender gets ACKs arriving this slot.
        arriving_acks, ack_channel = split_arrivals_ack(ack_channel, slot)
        ack_received = process_sender_acks(arriving_acks=arriving_acks, sender=sender, slot=slot)

        # 4) Timeout check and Go-Back-N retransmission.
        if sender.base < sender.next_packet and sender.timer_start is not None:
            if slot - sender.timer_start >= timeout_slots:
                timeout_fired = True
                sender.timer_start = slot

                for pid in range(sender.base, sender.next_packet):
                    packet = DataPacket(packet_id=pid, seq=pid % seq_mod)
                    total_data_tx += 1
                    retransmissions += 1
                    retransmitted += 1

                    kept = enqueue_data_packet(
                        packet=packet,
                        arrival_slot=slot + propagation_delay,
                        data_channel=data_channel,
                        data_loss_prob=data_loss_prob,
                        rng=rng,
                    )
                    if not kept:
                        data_drops += 1

        records.append(
            SlotRecord(
                slot=slot,
                base_before=base_before,
                base_after=sender.base,
                next_before=next_before,
                next_after=sender.next_packet,
                new_sent=new_sent,
                retransmitted=retransmitted,
                delivered_now=delivered_now,
                discarded_now=discarded_now,
                ack_received=ack_received,
                timeout_fired=timeout_fired,
            )
        )

        # Invariant: outstanding packets never exceed sender window.
        outstanding = sender.next_packet - sender.base
        if outstanding > window_size:
            raise AssertionError(f"window violation: outstanding={outstanding}, window={window_size}")

    else:
        raise RuntimeError("simulation reached max_slots before completion")

    return SimulationResult(
        records=records,
        delivered=list(receiver.delivered),
        total_data_tx=total_data_tx,
        total_ack_tx=total_ack_tx,
        retransmissions=retransmissions,
        data_drops=data_drops,
        ack_drops=ack_drops,
        total_slots=len(records),
    )


def verify_result(result: SimulationResult, total_packets: int) -> None:
    """Run deterministic correctness checks."""
    delivered = np.array(result.delivered, dtype=int)
    expected = np.arange(total_packets, dtype=int)

    if delivered.shape != expected.shape or not np.array_equal(delivered, expected):
        raise AssertionError(f"delivery mismatch: delivered={delivered.tolist()}")

    if result.retransmissions <= 0:
        raise AssertionError("expected at least one retransmission under configured loss")

    if result.total_data_tx < total_packets:
        raise AssertionError("total data transmissions cannot be smaller than packet count")

    if result.total_slots <= 0:
        raise AssertionError("simulation must consume positive number of slots")


def run_demo() -> None:
    """Run Go-Back-N demo with fixed parameters and print timeline/stats."""
    print("=== Go-Back-N ARQ MVP Demo ===")

    cfg = {
        "total_packets": 14,
        "window_size": 4,
        "seq_mod": 8,
        "timeout_slots": 4,
        "propagation_delay": 1,
        "data_loss_prob": 0.25,
        "ack_loss_prob": 0.2,
        "seed": 7,
        "max_slots": 300,
    }

    result = simulate_go_back_n(**cfg)
    verify_result(result, total_packets=cfg["total_packets"])

    throughput = len(result.delivered) / max(result.total_slots, 1)
    efficiency = len(result.delivered) / max(result.total_data_tx, 1)

    print("Configuration:", cfg)
    print("Slot timeline (first 20 slots):")
    for rec in result.records[:20]:
        print(
            f"  t={rec.slot:3d} base={rec.base_before:2d}->{rec.base_after:2d} "
            f"next={rec.next_before:2d}->{rec.next_after:2d} "
            f"new={rec.new_sent:2d} rtx={rec.retransmitted:2d} "
            f"deliv={rec.delivered_now:2d} disc={rec.discarded_now:2d} "
            f"ack={rec.ack_received:2d} timeout={int(rec.timeout_fired)}"
        )

    print("Delivered packet IDs:", result.delivered)
    print(
        "Stats:",
        {
            "total_slots": result.total_slots,
            "total_data_tx": result.total_data_tx,
            "total_ack_tx": result.total_ack_tx,
            "retransmissions": result.retransmissions,
            "data_drops": result.data_drops,
            "ack_drops": result.ack_drops,
            "throughput_pkt_per_slot": round(float(throughput), 4),
            "delivery_efficiency": round(float(efficiency), 4),
        },
    )
    print("All checks passed.")


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()

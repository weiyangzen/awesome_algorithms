"""Minimal runnable MVP for network error control (CRC + Stop-and-Wait ARQ)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


@dataclass
class AttemptRecord:
    packet_index: int
    seq: int
    payload: int
    attempt: int
    flip_positions: Tuple[int, ...]
    accepted: bool
    tx_frame_bits: str
    rx_frame_bits: str


def bits_from_int(value: int, width: int) -> np.ndarray:
    """Convert integer to fixed-width bit array (MSB first)."""
    if width <= 0:
        raise ValueError("width must be positive")
    if value < 0 or value >= (1 << width):
        raise ValueError(f"value {value} out of range for width {width}")
    return np.array([(value >> shift) & 1 for shift in range(width - 1, -1, -1)], dtype=np.uint8)


def bits_to_int(bits: Sequence[int]) -> int:
    """Convert MSB-first bit sequence to integer."""
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def normalize_generator_bits(generator_bits: Iterable[int]) -> np.ndarray:
    """Normalize CRC generator polynomial bits to a validated numpy array."""
    bits = np.array([int(b) for b in generator_bits], dtype=np.uint8)
    if bits.ndim != 1 or bits.size < 2:
        raise ValueError("generator_bits must be a 1D bit array with length >= 2")
    if bits[0] != 1 or bits[-1] != 1:
        raise ValueError("generator polynomial must start and end with 1")
    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError("generator_bits must contain only 0/1")
    return bits


def mod2_divide(dividend: np.ndarray, divisor: np.ndarray) -> np.ndarray:
    """Return remainder of polynomial long division over GF(2)."""
    if dividend.ndim != 1 or divisor.ndim != 1:
        raise ValueError("dividend and divisor must be 1D arrays")
    if dividend.size < divisor.size:
        raise ValueError("dividend length must be >= divisor length")

    work = dividend.astype(np.uint8, copy=True)
    width = divisor.size

    for idx in range(dividend.size - width + 1):
        if work[idx] == 1:
            work[idx : idx + width] ^= divisor

    remainder_len = width - 1
    return work[-remainder_len:] if remainder_len > 0 else np.zeros(0, dtype=np.uint8)


def crc_encode(data_bits: np.ndarray, generator_bits: np.ndarray) -> np.ndarray:
    """Append CRC remainder to data bits and return full codeword."""
    zeros = np.zeros(generator_bits.size - 1, dtype=np.uint8)
    padded = np.concatenate([data_bits, zeros])
    remainder = mod2_divide(padded, generator_bits)
    return np.concatenate([data_bits, remainder])


def crc_is_valid(codeword: np.ndarray, generator_bits: np.ndarray) -> bool:
    """Check whether codeword has zero CRC syndrome under generator_bits."""
    remainder = mod2_divide(codeword, generator_bits)
    return bool(np.all(remainder == 0))


def apply_error_pattern(bits: np.ndarray, flip_positions: Iterable[int]) -> np.ndarray:
    """Flip selected bit positions to emulate channel errors."""
    out = bits.astype(np.uint8, copy=True)
    for pos in flip_positions:
        if not (0 <= pos < out.size):
            raise ValueError(f"flip position {pos} out of range [0, {out.size - 1}]")
        out[pos] ^= 1
    return out


def frame_to_string(frame: np.ndarray) -> str:
    return "".join(str(int(x)) for x in frame)


def build_frame(payload: int, seq: int, generator_bits: np.ndarray) -> np.ndarray:
    """Build frame = seq(1 bit) + payload(8 bits) + crc bits."""
    if seq not in (0, 1):
        raise ValueError("seq must be 0 or 1")
    payload_bits = bits_from_int(payload, 8)
    data_bits = np.concatenate([np.array([seq], dtype=np.uint8), payload_bits])
    return crc_encode(data_bits, generator_bits)


def parse_frame(frame: np.ndarray, generator_bits: np.ndarray) -> Tuple[int, int]:
    """Parse accepted frame to (seq, payload)."""
    crc_len = generator_bits.size - 1
    data_bits = frame[:-crc_len] if crc_len > 0 else frame
    if data_bits.size != 9:
        raise ValueError(f"unexpected data length {data_bits.size}, expected 9 bits")
    seq = int(data_bits[0])
    payload = bits_to_int(data_bits[1:])
    return seq, payload


def simulate_stop_and_wait_arq(
    payloads: Sequence[int],
    generator_bits: Iterable[int],
    error_plan: Mapping[Tuple[int, int], Sequence[int]],
    max_retries: int = 8,
) -> Tuple[List[int], List[AttemptRecord], Dict[str, float]]:
    """Simulate Stop-and-Wait ARQ with CRC-based error detection."""
    if max_retries <= 0:
        raise ValueError("max_retries must be positive")

    g = normalize_generator_bits(generator_bits)
    decoded_payloads: List[int] = []
    records: List[AttemptRecord] = []

    for packet_index, payload in enumerate(payloads):
        if payload < 0 or payload > 255:
            raise ValueError("payload must be in [0, 255]")

        seq = packet_index % 2
        tx_frame = build_frame(payload=payload, seq=seq, generator_bits=g)

        accepted = False
        for attempt in range(1, max_retries + 1):
            flips = tuple(int(x) for x in error_plan.get((packet_index, attempt), ()))
            rx_frame = apply_error_pattern(tx_frame, flips)
            ok = crc_is_valid(rx_frame, g)

            records.append(
                AttemptRecord(
                    packet_index=packet_index,
                    seq=seq,
                    payload=payload,
                    attempt=attempt,
                    flip_positions=flips,
                    accepted=ok,
                    tx_frame_bits=frame_to_string(tx_frame),
                    rx_frame_bits=frame_to_string(rx_frame),
                )
            )

            if ok:
                rx_seq, rx_payload = parse_frame(rx_frame, g)
                if rx_seq != seq:
                    raise AssertionError(
                        f"sequence mismatch: expected {seq}, got {rx_seq} at packet {packet_index}"
                    )
                decoded_payloads.append(rx_payload)
                accepted = True
                break

        if not accepted:
            raise RuntimeError(f"packet {packet_index} failed after {max_retries} attempts")

    total_packets = len(payloads)
    total_attempts = len(records)
    retransmissions = total_attempts - total_packets
    frame_bits = 1 + 8 + (g.size - 1)
    info_bits_total = total_packets * 8
    sent_bits_total = total_attempts * frame_bits

    stats = {
        "total_packets": float(total_packets),
        "total_attempts": float(total_attempts),
        "retransmissions": float(retransmissions),
        "frame_bits": float(frame_bits),
        "goodput": float(info_bits_total / sent_bits_total),
    }

    return decoded_payloads, records, stats


def run_demo() -> None:
    """Run deterministic CRC+ARQ simulation and validate expected behavior."""
    print("=== Error Control MVP: CRC + Stop-and-Wait ARQ ===")

    payloads = [0x3A, 0xC1, 0x07, 0xBE]
    generator_bits = [1, 0, 0, 1, 1]  # x^4 + x + 1, CRC length = 4

    # (packet_index, attempt) -> flipped bit positions in the transmitted frame
    error_plan: Dict[Tuple[int, int], Sequence[int]] = {
        (0, 1): [2],
        (2, 1): [10],
        (2, 2): [7],
    }

    decoded, records, stats = simulate_stop_and_wait_arq(
        payloads=payloads,
        generator_bits=generator_bits,
        error_plan=error_plan,
        max_retries=6,
    )

    if decoded != payloads:
        raise AssertionError(f"decoded payload mismatch: {decoded} vs {payloads}")

    attempts_per_packet = np.zeros(len(payloads), dtype=int)
    accepted_per_packet = np.zeros(len(payloads), dtype=int)
    for rec in records:
        attempts_per_packet[rec.packet_index] += 1
        if rec.accepted:
            accepted_per_packet[rec.packet_index] += 1

    expected_attempts = np.array([2, 1, 3, 1], dtype=int)
    if not np.array_equal(attempts_per_packet, expected_attempts):
        raise AssertionError(
            f"unexpected attempts per packet: {attempts_per_packet.tolist()}, expected {expected_attempts.tolist()}"
        )

    if not np.array_equal(accepted_per_packet, np.ones(len(payloads), dtype=int)):
        raise AssertionError(f"each packet must be accepted exactly once: {accepted_per_packet.tolist()}")

    total_attempts = int(stats["total_attempts"])
    retransmissions = int(stats["retransmissions"])
    if total_attempts != 7:
        raise AssertionError(f"unexpected total attempts: {total_attempts}")
    if retransmissions != 3:
        raise AssertionError(f"unexpected retransmissions: {retransmissions}")

    for rec in records:
        if not rec.accepted and len(rec.flip_positions) == 0:
            raise AssertionError(
                f"rejected attempt without injected channel errors: packet={rec.packet_index}, attempt={rec.attempt}"
            )

    print("Transmission timeline:")
    for rec in records:
        decision = "ACK" if rec.accepted else "NACK"
        print(
            f"  pkt={rec.packet_index} seq={rec.seq} payload=0x{rec.payload:02X} "
            f"attempt={rec.attempt} flips={list(rec.flip_positions)} -> {decision}"
        )

    print("decoded payloads:", [f"0x{x:02X}" for x in decoded])
    print(
        "stats:",
        {
            "total_packets": int(stats["total_packets"]),
            "total_attempts": int(stats["total_attempts"]),
            "retransmissions": int(stats["retransmissions"]),
            "frame_bits": int(stats["frame_bits"]),
            "goodput": round(stats["goodput"], 4),
        },
    )
    print("All checks passed.")


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()

"""Minimal runnable MVP for CRC (Cyclic Redundancy Check)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np


@dataclass
class CheckRecord:
    name: str
    flip_positions: Tuple[int, ...]
    valid: bool
    syndrome: str


def parse_generator_bits(generator_bits: str) -> np.ndarray:
    """Parse generator bits like '10011' into a validated numpy bit array."""
    if len(generator_bits) < 2:
        raise ValueError("generator_bits length must be >= 2")
    if any(ch not in {"0", "1"} for ch in generator_bits):
        raise ValueError("generator_bits must contain only '0'/'1'")
    if generator_bits[0] != "1" or generator_bits[-1] != "1":
        raise ValueError("generator polynomial must start and end with '1'")
    return np.array([int(ch) for ch in generator_bits], dtype=np.uint8)


def bits_from_bytes(payload: bytes) -> np.ndarray:
    """Convert bytes to MSB-first bits using numpy."""
    if not payload:
        raise ValueError("payload must not be empty")
    arr = np.frombuffer(payload, dtype=np.uint8)
    return np.unpackbits(arr)


def bits_to_string(bits: np.ndarray, group: int = 8) -> str:
    """Format bit array as grouped string for readable logs."""
    raw = "".join(str(int(x)) for x in bits)
    if group <= 0:
        return raw
    return "_".join(raw[idx : idx + group] for idx in range(0, len(raw), group))


def mod2_divide(dividend: np.ndarray, divisor: np.ndarray) -> np.ndarray:
    """Return GF(2) remainder from polynomial long division."""
    if dividend.ndim != 1 or divisor.ndim != 1:
        raise ValueError("dividend and divisor must be 1D arrays")
    if dividend.size < divisor.size:
        raise ValueError("dividend length must be >= divisor length")

    work = dividend.astype(np.uint8, copy=True)
    span = divisor.size

    for idx in range(dividend.size - span + 1):
        if work[idx] == 1:
            work[idx : idx + span] ^= divisor

    remainder_len = span - 1
    if remainder_len == 0:
        return np.zeros(0, dtype=np.uint8)
    return work[-remainder_len:]


def crc_encode(data_bits: np.ndarray, generator: np.ndarray) -> np.ndarray:
    """Append CRC remainder bits to data bits."""
    zeros = np.zeros(generator.size - 1, dtype=np.uint8)
    padded = np.concatenate([data_bits, zeros])
    remainder = mod2_divide(padded, generator)
    return np.concatenate([data_bits, remainder])


def crc_syndrome(codeword: np.ndarray, generator: np.ndarray) -> np.ndarray:
    """Compute CRC syndrome (remainder) for a received codeword."""
    return mod2_divide(codeword, generator)


def crc_check(codeword: np.ndarray, generator: np.ndarray) -> bool:
    """Return True when syndrome is all zeros."""
    return bool(np.all(crc_syndrome(codeword, generator) == 0))


def flip_bits(bits: np.ndarray, flip_positions: Iterable[int]) -> np.ndarray:
    """Flip selected bit positions to emulate channel errors."""
    out = bits.astype(np.uint8, copy=True)
    for pos in flip_positions:
        if pos < 0 or pos >= out.size:
            raise ValueError(f"flip position {pos} out of range [0, {out.size - 1}]")
        out[pos] ^= 1
    return out


def find_detected_two_bit_error(codeword: np.ndarray, generator: np.ndarray) -> Tuple[int, int]:
    """Find one two-bit error pattern that CRC can detect."""
    n = codeword.size
    for i in range(n):
        for j in range(i + 1, n):
            candidate = flip_bits(codeword, (i, j))
            if not crc_check(candidate, generator):
                return (i, j)
    raise RuntimeError("failed to find a detectable two-bit error")


def find_detected_burst_error(codeword: np.ndarray, generator: np.ndarray, width: int = 4) -> Tuple[int, ...]:
    """Find one detectable contiguous burst error pattern."""
    if width <= 0:
        raise ValueError("width must be positive")
    n = codeword.size
    if width > n:
        raise ValueError("width cannot exceed codeword length")

    for start in range(0, n - width + 1):
        positions = tuple(range(start, start + width))
        candidate = flip_bits(codeword, positions)
        if not crc_check(candidate, generator):
            return positions
    raise RuntimeError("failed to find a detectable burst error")


def evaluate_cases(
    codeword: np.ndarray,
    generator: np.ndarray,
    cases: Sequence[Tuple[str, Tuple[int, ...]]],
) -> List[CheckRecord]:
    """Run all error-injection scenarios and collect records."""
    records: List[CheckRecord] = []
    for name, flips in cases:
        received = flip_bits(codeword, flips)
        syndrome = crc_syndrome(received, generator)
        records.append(
            CheckRecord(
                name=name,
                flip_positions=flips,
                valid=bool(np.all(syndrome == 0)),
                syndrome=bits_to_string(syndrome, group=0),
            )
        )
    return records


def run_demo() -> None:
    """Run deterministic CRC demo and print verification report."""
    payload = b"CRC"
    generator_bits = "10011"  # x^4 + x + 1

    generator = parse_generator_bits(generator_bits)
    data_bits = bits_from_bytes(payload)
    codeword = crc_encode(data_bits, generator)

    two_bit = find_detected_two_bit_error(codeword, generator)
    burst = find_detected_burst_error(codeword, generator, width=4)

    cases: List[Tuple[str, Tuple[int, ...]]] = [
        ("no_error", tuple()),
        ("single_bit_error", (5,)),
        ("two_bit_error", two_bit),
        ("burst_error", burst),
    ]

    records = evaluate_cases(codeword, generator, cases)

    # Core assertions for MVP correctness.
    by_name = {record.name: record for record in records}
    if not by_name["no_error"].valid:
        raise AssertionError("no_error case must pass CRC check")

    for idx in range(codeword.size):
        received = flip_bits(codeword, (idx,))
        if crc_check(received, generator):
            raise AssertionError(f"single-bit error at position {idx} should be detected")

    if by_name["single_bit_error"].valid:
        raise AssertionError("single_bit_error case should fail CRC check")
    if by_name["two_bit_error"].valid:
        raise AssertionError("two_bit_error case should fail CRC check")
    if by_name["burst_error"].valid:
        raise AssertionError("burst_error case should fail CRC check")

    print("=== CRC MVP Demo ===")
    print("Generator bits:", generator_bits)
    print("Payload:", payload)
    print("Data bits:", bits_to_string(data_bits, group=8))
    print("Codeword bits:", bits_to_string(codeword, group=8))
    print("Codeword length:", codeword.size)

    print("\nCase results:")
    for record in records:
        print(
            f"  {record.name:>16} flips={list(record.flip_positions)!s:<16} "
            f"valid={record.valid!s:<5} syndrome={record.syndrome}"
        )

    print("\nAll checks passed.")


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()

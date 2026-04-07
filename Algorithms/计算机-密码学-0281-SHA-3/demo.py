"""Educational SHA-3 MVP (source-level Keccak sponge implementation).

This script implements SHA-3 digest generation without using hashlib as the
primary engine. We explicitly code:
- Keccak-f[1600] permutation (theta, rho, pi, chi, iota)
- multi-rate padding for SHA-3 domain separation
- sponge absorb/squeeze flow for SHA3-224/256/384/512

No interactive input is required.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import pandas as pd

MASK64 = (1 << 64) - 1

ROUND_CONSTANTS = [
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808A,
    0x8000000080008000,
    0x000000000000808B,
    0x0000000080000001,
    0x8000000080008081,
    0x8000000000008009,
    0x000000000000008A,
    0x0000000000000088,
    0x0000000080008009,
    0x000000008000000A,
    0x000000008000808B,
    0x800000000000008B,
    0x8000000000008089,
    0x8000000000008003,
    0x8000000000008002,
    0x8000000000000080,
    0x000000000000800A,
    0x800000008000000A,
    0x8000000080008081,
    0x8000000000008080,
    0x0000000080000001,
    0x8000000080008008,
]

# Rotation offsets indexed by [x][y].
ROTATION_OFFSETS = [
    [0, 36, 3, 41, 18],
    [1, 44, 10, 45, 2],
    [62, 6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39, 8, 14],
]

# SHA-3 standard variants: output bits -> (rate bits, hashlib name)
SHA3_SPECS = {
    224: (1152, "sha3_224"),
    256: (1088, "sha3_256"),
    384: (832, "sha3_384"),
    512: (576, "sha3_512"),
}


@dataclass
class SHA3Stats:
    digest_bits: int
    rate_bits: int
    capacity_bits: int
    message_bytes: int
    absorbed_blocks: int
    permutation_calls: int


def _rotl64(value: int, shift: int) -> int:
    """Rotate a 64-bit integer left by shift bits."""
    shift %= 64
    return ((value << shift) | (value >> (64 - shift))) & MASK64


def _keccak_f1600(state: list[int]) -> None:
    """In-place Keccak-f[1600] permutation over 25 lanes (64-bit each)."""
    for rc in ROUND_CONSTANTS:
        # Theta
        c = [0] * 5
        for x in range(5):
            c[x] = (
                state[x]
                ^ state[x + 5]
                ^ state[x + 10]
                ^ state[x + 15]
                ^ state[x + 20]
            )

        d = [0] * 5
        for x in range(5):
            d[x] = c[(x - 1) % 5] ^ _rotl64(c[(x + 1) % 5], 1)

        for y in range(5):
            for x in range(5):
                idx = x + 5 * y
                state[idx] ^= d[x]

        # Rho + Pi
        b = [0] * 25
        for x in range(5):
            for y in range(5):
                idx = x + 5 * y
                new_x = y
                new_y = (2 * x + 3 * y) % 5
                b[new_x + 5 * new_y] = _rotl64(state[idx], ROTATION_OFFSETS[x][y])

        # Chi
        for y in range(5):
            row = [b[x + 5 * y] for x in range(5)]
            for x in range(5):
                state[x + 5 * y] = (
                    row[x] ^ ((~row[(x + 1) % 5]) & row[(x + 2) % 5])
                ) & MASK64

        # Iota
        state[0] ^= rc


def _pad10star1_sha3(message: bytes, rate_bytes: int, suffix: int = 0x06) -> bytes:
    """Apply SHA-3 domain suffix + pad10*1 padding for a given rate."""
    padded = bytearray(message)
    padded.append(suffix)

    # Reserve one last byte for final 1-bit (0x80), then complete block alignment.
    zeros_needed = (rate_bytes - ((len(padded) + 1) % rate_bytes)) % rate_bytes
    padded.extend(b"\x00" * zeros_needed)
    padded.append(0x80)

    return bytes(padded)


def sha3_digest(data: bytes, digest_bits: int = 256) -> tuple[bytes, SHA3Stats]:
    """Compute SHA-3 digest from source-level Keccak sponge implementation."""
    if digest_bits not in SHA3_SPECS:
        raise ValueError(f"Unsupported SHA-3 output size: {digest_bits}")

    rate_bits, _ = SHA3_SPECS[digest_bits]
    rate_bytes = rate_bits // 8
    digest_bytes = digest_bits // 8
    capacity_bits = 1600 - rate_bits

    state = [0] * 25
    padded = _pad10star1_sha3(data, rate_bytes=rate_bytes)

    permutation_calls = 0
    absorbed_blocks = len(padded) // rate_bytes

    # Absorb phase
    for block_start in range(0, len(padded), rate_bytes):
        block = padded[block_start : block_start + rate_bytes]
        for i, byte_value in enumerate(block):
            lane_index = i // 8
            lane_shift = (i % 8) * 8
            state[lane_index] ^= (byte_value & 0xFF) << lane_shift

        _keccak_f1600(state)
        permutation_calls += 1

    # Squeeze phase
    out = bytearray()
    while len(out) < digest_bytes:
        for i in range(rate_bytes):
            lane_index = i // 8
            lane_shift = (i % 8) * 8
            out.append((state[lane_index] >> lane_shift) & 0xFF)
            if len(out) == digest_bytes:
                break

        if len(out) < digest_bytes:
            _keccak_f1600(state)
            permutation_calls += 1

    stats = SHA3Stats(
        digest_bits=digest_bits,
        rate_bits=rate_bits,
        capacity_bits=capacity_bits,
        message_bytes=len(data),
        absorbed_blocks=absorbed_blocks,
        permutation_calls=permutation_calls,
    )
    return bytes(out), stats


def sha3_hexdigest(data: bytes, digest_bits: int = 256) -> tuple[str, SHA3Stats]:
    digest, stats = sha3_digest(data, digest_bits=digest_bits)
    return digest.hex(), stats


def _self_test_known_vectors() -> None:
    """Check common SHA3-256 known vectors."""
    vectors = [
        (b"", "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a"),
        (b"abc", "3a985da74fe225b2045c172d6bd390bd855f086e3e9d525b46bfe24511431532"),
    ]
    for message, expected_hex in vectors:
        got_hex, _ = sha3_hexdigest(message, digest_bits=256)
        assert got_hex == expected_hex


def _self_test_against_hashlib() -> None:
    """Cross-check SHA3-224/256/384/512 against hashlib on deterministic samples."""
    rng = np.random.default_rng(138)
    random_payload = rng.integers(0, 256, size=257, dtype=np.uint8).tobytes()

    samples = [
        b"",
        b"abc",
        b"SHA-3 minimal MVP for CS-0138",
        b"The quick brown fox jumps over the lazy dog",
        bytes(range(256)),
        random_payload,
    ]

    for digest_bits, (_, hashlib_name) in SHA3_SPECS.items():
        for sample in samples:
            got_hex, _ = sha3_hexdigest(sample, digest_bits=digest_bits)
            expected_hex = hashlib.new(hashlib_name, sample).hexdigest()
            assert got_hex == expected_hex


def _build_digest_report(message: bytes) -> pd.DataFrame:
    """Build a pandas table that compares source-level SHA-3 with hashlib."""
    rows: list[dict[str, object]] = []

    for digest_bits, (_, hashlib_name) in SHA3_SPECS.items():
        source_hex, stats = sha3_hexdigest(message, digest_bits=digest_bits)
        hashlib_hex = hashlib.new(hashlib_name, message).hexdigest()
        rows.append(
            {
                "variant": hashlib_name,
                "digest_bits": digest_bits,
                "message_bytes": stats.message_bytes,
                "absorbed_blocks": stats.absorbed_blocks,
                "permutation_calls": stats.permutation_calls,
                "match_hashlib": source_hex == hashlib_hex,
                "digest_hex": source_hex,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    _self_test_known_vectors()
    _self_test_against_hashlib()

    message = (
        b"SHA-3 educational demo. "
        b"Digest is generated by explicit Keccak-f[1600] sponge steps."
    )

    report = _build_digest_report(message)

    print("=== SHA-3 MVP (CS-0138) ===")
    print("All built-in known vectors and hashlib cross-checks passed.")
    print()
    print("Per-variant digest report:")
    print(report.to_string(index=False))

    if not bool(report["match_hashlib"].all()):
        raise RuntimeError("SHA-3 mismatch between source-level implementation and hashlib")


if __name__ == "__main__":
    main()

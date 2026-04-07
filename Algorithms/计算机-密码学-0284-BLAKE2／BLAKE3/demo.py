"""Runnable MVP for BLAKE2/BLAKE3 (CS-0140)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import hashlib
import numpy as np
import pandas as pd

OUT_LEN = 32
BLOCK_LEN = 64
CHUNK_LEN = 1024

CHUNK_START = 1 << 0
CHUNK_END = 1 << 1
PARENT = 1 << 2
ROOT = 1 << 3

IV = [
    0x6A09E667,
    0xBB67AE85,
    0x3C6EF372,
    0xA54FF53A,
    0x510E527F,
    0x9B05688C,
    0x1F83D9AB,
    0x5BE0CD19,
]

MSG_PERMUTATION = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8]


@dataclass(frozen=True)
class Blake3Vector:
    """One official BLAKE3 test vector (first 32-byte digest only)."""

    input_len: int
    hash32_hex: str


BLAKE3_VECTOR_CASES = [
    Blake3Vector(0, "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"),
    Blake3Vector(1, "2d3adedff11b61f14c886e35afa036736dcd87a74d27b5c1510225d0f592e213"),
    Blake3Vector(2, "7b7015bb92cf0b318037702a6cdd81dee41224f734684c2c122cd6359cb1ee63"),
    Blake3Vector(3, "e1be4d7a8ab5560aa4199eea339849ba8e293d55ca0a81006726d184519e647f"),
    Blake3Vector(4, "f30f5ab28fe047904037f77b6da4fea1e27241c5d132638d8bedce9d40494f32"),
    Blake3Vector(5, "b40b44dfd97e7a84a996a91af8b85188c66c126940ba7aad2e7ae6b385402aa2"),
    Blake3Vector(63, "e9bc37a594daad83be9470df7f7b3798297c3d834ce80ba85d6e207627b7db7b"),
    Blake3Vector(64, "4eed7141ea4a5cd4b788606bd23f46e212af9cacebacdc7d1f4c6dc7f2511b98"),
    Blake3Vector(65, "de1e5fa0be70df6d2be8fffd0e99ceaa8eb6e8c93a63f2d8d1c30ecb6b263dee"),
    Blake3Vector(1023, "10108970eeda3eb932baac1428c7a2163b0e924c9a9e25b35bba72b28f70bd11"),
    Blake3Vector(1024, "42214739f095a406f3fc83deb889744ac00df831c10daa55189b5d121c855af7"),
    Blake3Vector(1025, "d00278ae47eb27b34faecf67b4fe263f82d5412916c1ffd97c8cb7fb814b8444"),
    Blake3Vector(2048, "e776b6028c7cd22a4d0ba182a8bf62205d2ef576467e838ed6f2529b85fba24a"),
]


@dataclass
class Output:
    """BLAKE3 output descriptor before choosing CV or root bytes."""

    input_chaining_value: list[int]
    block_words: list[int]
    counter: int
    block_len: int
    flags: int

    def chaining_value(self) -> list[int]:
        return _compress(
            self.input_chaining_value,
            self.block_words,
            self.counter,
            self.block_len,
            self.flags,
        )[:8]

    def root_output_bytes(self, out_len: int) -> bytes:
        if out_len <= 0:
            raise ValueError("out_len must be positive.")

        out = bytearray()
        output_block_counter = 0
        while len(out) < out_len:
            words = _compress(
                self.input_chaining_value,
                self.block_words,
                output_block_counter,
                self.block_len,
                self.flags | ROOT,
            )
            out.extend(b"".join(word.to_bytes(4, "little") for word in words))
            output_block_counter += 1
        return bytes(out[:out_len])


def _repeat_pattern_input(length: int, period: int = 251) -> bytes:
    """Official test-vector input pattern: 0..250 repeated."""
    if length < 0:
        raise ValueError("length must be non-negative.")
    return bytes(i % period for i in range(length))


def _rotr32(x: int, n: int) -> int:
    x &= 0xFFFFFFFF
    return ((x >> n) | ((x << (32 - n)) & 0xFFFFFFFF)) & 0xFFFFFFFF


def _g(state: list[int], a: int, b: int, c: int, d: int, mx: int, my: int) -> None:
    state[a] = (state[a] + state[b] + mx) & 0xFFFFFFFF
    state[d] = _rotr32(state[d] ^ state[a], 16)
    state[c] = (state[c] + state[d]) & 0xFFFFFFFF
    state[b] = _rotr32(state[b] ^ state[c], 12)
    state[a] = (state[a] + state[b] + my) & 0xFFFFFFFF
    state[d] = _rotr32(state[d] ^ state[a], 8)
    state[c] = (state[c] + state[d]) & 0xFFFFFFFF
    state[b] = _rotr32(state[b] ^ state[c], 7)


def _round(state: list[int], m: Sequence[int]) -> None:
    _g(state, 0, 4, 8, 12, m[0], m[1])
    _g(state, 1, 5, 9, 13, m[2], m[3])
    _g(state, 2, 6, 10, 14, m[4], m[5])
    _g(state, 3, 7, 11, 15, m[6], m[7])

    _g(state, 0, 5, 10, 15, m[8], m[9])
    _g(state, 1, 6, 11, 12, m[10], m[11])
    _g(state, 2, 7, 8, 13, m[12], m[13])
    _g(state, 3, 4, 9, 14, m[14], m[15])


def _permute(words: Sequence[int]) -> list[int]:
    return [words[i] for i in MSG_PERMUTATION]


def _block_words_from_bytes(block: bytes) -> list[int]:
    if len(block) > BLOCK_LEN:
        raise ValueError("block length cannot exceed 64 bytes.")
    padded = block + (b"\x00" * (BLOCK_LEN - len(block)))
    return [int.from_bytes(padded[i : i + 4], "little") for i in range(0, BLOCK_LEN, 4)]


def _compress(
    chaining_value: Sequence[int],
    block_words: Sequence[int],
    counter: int,
    block_len: int,
    flags: int,
) -> list[int]:
    if len(chaining_value) != 8:
        raise ValueError("chaining_value must contain 8 words.")
    if len(block_words) != 16:
        raise ValueError("block_words must contain 16 words.")

    counter_low = counter & 0xFFFFFFFF
    counter_high = (counter >> 32) & 0xFFFFFFFF

    state = [
        chaining_value[0],
        chaining_value[1],
        chaining_value[2],
        chaining_value[3],
        chaining_value[4],
        chaining_value[5],
        chaining_value[6],
        chaining_value[7],
        IV[0],
        IV[1],
        IV[2],
        IV[3],
        counter_low,
        counter_high,
        block_len,
        flags,
    ]

    block = list(block_words)
    for round_idx in range(7):
        _round(state, block)
        if round_idx < 6:
            block = _permute(block)

    for i in range(8):
        state[i] ^= state[i + 8]
        state[i + 8] ^= chaining_value[i]

    return [x & 0xFFFFFFFF for x in state]


def _parent_output(
    left_child_cv: Sequence[int],
    right_child_cv: Sequence[int],
    key_words: Sequence[int],
    flags: int,
) -> Output:
    block_words = list(left_child_cv) + list(right_child_cv)
    return Output(
        input_chaining_value=list(key_words),
        block_words=block_words,
        counter=0,
        block_len=BLOCK_LEN,
        flags=flags | PARENT,
    )


def _parent_cv(
    left_child_cv: Sequence[int],
    right_child_cv: Sequence[int],
    key_words: Sequence[int],
    flags: int,
) -> list[int]:
    return _parent_output(left_child_cv, right_child_cv, key_words, flags).chaining_value()


def _chunk_output(chunk: bytes, key_words: Sequence[int], chunk_counter: int, flags: int) -> Output:
    if len(chunk) > CHUNK_LEN:
        raise ValueError("one chunk cannot exceed 1024 bytes.")

    cv = list(key_words)
    blocks = [chunk[i : i + BLOCK_LEN] for i in range(0, len(chunk), BLOCK_LEN)]
    if not blocks:
        blocks = [b""]

    for block_index, block in enumerate(blocks[:-1]):
        block_flags = flags | (CHUNK_START if block_index == 0 else 0)
        cv = _compress(
            cv,
            _block_words_from_bytes(block),
            chunk_counter,
            BLOCK_LEN,
            block_flags,
        )[:8]

    last_block = blocks[-1]
    last_flags = flags | CHUNK_END
    if len(blocks) == 1:
        last_flags |= CHUNK_START

    return Output(
        input_chaining_value=cv,
        block_words=_block_words_from_bytes(last_block),
        counter=chunk_counter,
        block_len=len(last_block),
        flags=last_flags,
    )


def _add_chunk_chaining_value(
    cv_stack: list[list[int]],
    new_cv: list[int],
    total_chunks: int,
    key_words: Sequence[int],
    flags: int,
) -> None:
    while total_chunks & 1 == 0:
        left_child = cv_stack.pop()
        new_cv = _parent_cv(left_child, new_cv, key_words, flags)
        total_chunks >>= 1
    cv_stack.append(new_cv)


def blake3_hash(data: bytes, out_len: int = OUT_LEN) -> bytes:
    """Reference-style BLAKE3 implementation (regular hash mode, unkeyed)."""
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("data must be bytes-like.")
    if out_len <= 0:
        raise ValueError("out_len must be positive.")

    data_bytes = bytes(data)
    key_words = IV.copy()
    flags = 0

    chunks = [data_bytes[i : i + CHUNK_LEN] for i in range(0, len(data_bytes), CHUNK_LEN)]
    if not chunks:
        chunks = [b""]

    cv_stack: list[list[int]] = []
    last_output: Output | None = None

    for chunk_index, chunk in enumerate(chunks):
        output = _chunk_output(chunk, key_words, chunk_index, flags)
        is_last_chunk = chunk_index == len(chunks) - 1

        if is_last_chunk:
            last_output = output
        else:
            chunk_cv = output.chaining_value()
            total_chunks = chunk_index + 1
            _add_chunk_chaining_value(cv_stack, chunk_cv, total_chunks, key_words, flags)

    if last_output is None:
        raise RuntimeError("BLAKE3 internal error: no final chunk output.")

    output = last_output
    while cv_stack:
        left_child = cv_stack.pop()
        output = _parent_output(left_child, output.chaining_value(), key_words, flags)

    return output.root_output_bytes(out_len)


def blake2b_hash(
    data: bytes,
    digest_size: int = 32,
    key: bytes = b"",
    salt: bytes = b"",
    person: bytes = b"",
) -> bytes:
    return hashlib.blake2b(
        data,
        digest_size=digest_size,
        key=key,
        salt=salt,
        person=person,
    ).digest()


def blake2s_hash(
    data: bytes,
    digest_size: int = 32,
    key: bytes = b"",
    salt: bytes = b"",
    person: bytes = b"",
) -> bytes:
    return hashlib.blake2s(
        data,
        digest_size=digest_size,
        key=key,
        salt=salt,
        person=person,
    ).digest()


def verify_blake3_vectors() -> pd.DataFrame:
    """Verify implementation against official BLAKE3 test vectors."""
    records: list[dict[str, object]] = []
    all_passed = True

    for case in BLAKE3_VECTOR_CASES:
        data = _repeat_pattern_input(case.input_len)
        got = blake3_hash(data, out_len=32).hex()
        passed = got == case.hash32_hex
        all_passed = all_passed and passed

        records.append(
            {
                "input_len": case.input_len,
                "expected_32": case.hash32_hex,
                "got_32": got,
                "pass": passed,
            }
        )

    table = pd.DataFrame(records)
    if not all_passed:
        failed = table.loc[~table["pass"]]
        raise RuntimeError(f"BLAKE3 vector mismatch:\n{failed.to_string(index=False)}")

    return table


def _hamming_distance_bits(a: bytes, b: bytes) -> int:
    if len(a) != len(b):
        raise ValueError("Digest lengths must match for Hamming distance.")

    xa = np.frombuffer(a, dtype=np.uint8)
    xb = np.frombuffer(b, dtype=np.uint8)
    diff = np.bitwise_xor(xa, xb)
    return int(np.unpackbits(diff).sum())


def build_blake2_showcase(message: bytes) -> pd.DataFrame:
    """Build a compact comparison table for BLAKE2b/BLAKE2s modes."""
    rows = []

    rows.append(
        {
            "algorithm": "BLAKE2b",
            "mode": "unkeyed",
            "digest_hex": blake2b_hash(message, digest_size=32).hex(),
        }
    )
    rows.append(
        {
            "algorithm": "BLAKE2s",
            "mode": "unkeyed",
            "digest_hex": blake2s_hash(message, digest_size=32).hex(),
        }
    )
    rows.append(
        {
            "algorithm": "BLAKE2b",
            "mode": "keyed",
            "digest_hex": blake2b_hash(message, digest_size=32, key=b"demo-key").hex(),
        }
    )
    rows.append(
        {
            "algorithm": "BLAKE2s",
            "mode": "keyed",
            "digest_hex": blake2s_hash(message, digest_size=32, key=b"demo-key").hex(),
        }
    )
    rows.append(
        {
            "algorithm": "BLAKE2b",
            "mode": "salt+person",
            "digest_hex": blake2b_hash(
                message,
                digest_size=32,
                salt=b"salt-123salt-123",
                person=b"person-demo-1234",
            ).hex(),
        }
    )
    rows.append(
        {
            "algorithm": "BLAKE2s",
            "mode": "salt+person",
            "digest_hex": blake2s_hash(
                message,
                digest_size=32,
                salt=b"salt1234",
                person=b"person12",
            ).hex(),
        }
    )

    return pd.DataFrame(rows)


def build_avalanche_table(message: bytes) -> pd.DataFrame:
    """Measure bit-flip sensitivity (one-bit input perturbation)."""
    mutated = bytearray(message)
    mutated[-1] ^= 0x01
    mutated_message = bytes(mutated)

    pairs = [
        (
            "BLAKE2b-256",
            blake2b_hash(message, digest_size=32),
            blake2b_hash(mutated_message, digest_size=32),
        ),
        (
            "BLAKE2s-256",
            blake2s_hash(message, digest_size=32),
            blake2s_hash(mutated_message, digest_size=32),
        ),
        ("BLAKE3-256", blake3_hash(message, out_len=32), blake3_hash(mutated_message, out_len=32)),
    ]

    rows = []
    for name, digest_a, digest_b in pairs:
        changed_bits = _hamming_distance_bits(digest_a, digest_b)
        total_bits = len(digest_a) * 8
        rows.append(
            {
                "algorithm": name,
                "changed_bits": changed_bits,
                "total_bits": total_bits,
                "change_ratio": round(changed_bits / total_bits, 4),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    """Run deterministic, non-interactive BLAKE2/BLAKE3 MVP checks."""
    message = b"The quick brown fox jumps over the lazy dog"

    print("=== BLAKE2 Showcase ===")
    print(f"Input message: {message!r}")
    blake2_table = build_blake2_showcase(message)
    print(blake2_table.to_string(index=False))

    print("\n=== BLAKE3 Official Vector Verification (32-byte prefix) ===")
    vector_table = verify_blake3_vectors()
    print(vector_table.to_string(index=False))

    print("\n=== Avalanche Check (1-bit input flip) ===")
    avalanche_table = build_avalanche_table(message)
    print(avalanche_table.to_string(index=False))

    blake3_xof_64 = blake3_hash(message, out_len=64).hex()
    print("\n=== BLAKE3 XOF (64 bytes) ===")
    print(blake3_xof_64)

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()

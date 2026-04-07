"""Salsa20 (20 rounds) minimal runnable MVP.

This file implements Salsa20 from source-level primitives:
- quarterround
- 20-round Salsa20 hash core
- key/nonce expansion
- stream generation and XOR encryption/decryption
"""

from __future__ import annotations

from hashlib import sha256

MASK32 = 0xFFFFFFFF


def _rotl32(x: int, n: int) -> int:
    """Rotate-left for 32-bit unsigned integer."""
    return ((x << n) & MASK32) | (x >> (32 - n))


def _quarterround(y0: int, y1: int, y2: int, y3: int) -> tuple[int, int, int, int]:
    """Salsa20 quarterround operation."""
    z1 = y1 ^ _rotl32((y0 + y3) & MASK32, 7)
    z2 = y2 ^ _rotl32((z1 + y0) & MASK32, 9)
    z3 = y3 ^ _rotl32((z2 + z1) & MASK32, 13)
    z0 = y0 ^ _rotl32((z3 + z2) & MASK32, 18)
    return z0, z1, z2, z3


def _salsa20_hash(state: list[int]) -> bytes:
    """20-round Salsa20 hash on 16 little-endian 32-bit words."""
    x = state[:]
    for _ in range(10):  # 10 double-rounds = 20 rounds
        # Column round
        x[0], x[4], x[8], x[12] = _quarterround(x[0], x[4], x[8], x[12])
        x[5], x[9], x[13], x[1] = _quarterround(x[5], x[9], x[13], x[1])
        x[10], x[14], x[2], x[6] = _quarterround(x[10], x[14], x[2], x[6])
        x[15], x[3], x[7], x[11] = _quarterround(x[15], x[3], x[7], x[11])

        # Row round
        x[0], x[1], x[2], x[3] = _quarterround(x[0], x[1], x[2], x[3])
        x[5], x[6], x[7], x[4] = _quarterround(x[5], x[6], x[7], x[4])
        x[10], x[11], x[8], x[9] = _quarterround(x[10], x[11], x[8], x[9])
        x[15], x[12], x[13], x[14] = _quarterround(x[15], x[12], x[13], x[14])

    out_words = [((x[i] + state[i]) & MASK32) for i in range(16)]
    return b"".join(w.to_bytes(4, "little") for w in out_words)


def _bytes_to_words_le(data: bytes) -> list[int]:
    return [int.from_bytes(data[i : i + 4], "little") for i in range(0, len(data), 4)]


def _build_state(key: bytes, nonce: bytes, block_counter: int) -> list[int]:
    """Build Salsa20 initial 16-word state.

    Supports 16-byte and 32-byte keys, 8-byte nonce, 64-bit block counter.
    """
    if len(key) not in (16, 32):
        raise ValueError("key length must be 16 or 32 bytes")
    if len(nonce) != 8:
        raise ValueError("nonce length must be 8 bytes")
    if not (0 <= block_counter < (1 << 64)):
        raise ValueError("block_counter must be in [0, 2^64)")

    if len(key) == 32:
        constants = _bytes_to_words_le(b"expand 32-byte k")
        k = _bytes_to_words_le(key)
        k0, k1, k2, k3, k4, k5, k6, k7 = k
    else:
        constants = _bytes_to_words_le(b"expand 16-byte k")
        k = _bytes_to_words_le(key)
        k0, k1, k2, k3 = k
        k4, k5, k6, k7 = k  # repeat for 16-byte key mode

    n0, n1 = _bytes_to_words_le(nonce)
    c0, c1, c2, c3 = constants
    b0 = block_counter & MASK32
    b1 = (block_counter >> 32) & MASK32

    return [
        c0,
        k0,
        k1,
        k2,
        k3,
        c1,
        n0,
        n1,
        b0,
        b1,
        c2,
        k4,
        k5,
        k6,
        k7,
        c3,
    ]


def salsa20_keystream(key: bytes, nonce: bytes, length: int, counter: int = 0) -> bytes:
    """Generate Salsa20 keystream bytes of requested length."""
    if length < 0:
        raise ValueError("length must be non-negative")
    out = bytearray()
    block_counter = counter
    while len(out) < length:
        state = _build_state(key, nonce, block_counter)
        out.extend(_salsa20_hash(state))
        block_counter = (block_counter + 1) & ((1 << 64) - 1)
    return bytes(out[:length])


def salsa20_xor(key: bytes, nonce: bytes, data: bytes, counter: int = 0) -> bytes:
    """Encrypt/decrypt by XOR with keystream."""
    ks = salsa20_keystream(key, nonce, len(data), counter)
    return bytes(d ^ k for d, k in zip(data, ks))


def _hamming_bits(a: bytes, b: bytes) -> int:
    if len(a) != len(b):
        raise ValueError("inputs must have same length")
    return sum((x ^ y).bit_count() for x, y in zip(a, b))


def main() -> None:
    # Deterministic inputs for reproducible output.
    key = bytes(range(32))
    nonce = bytes(range(8))
    plaintext = (
        b"Salsa20 is a stream cipher. "
        b"This MVP demonstrates source-level implementation."
    )

    ciphertext = salsa20_xor(key, nonce, plaintext, counter=0)
    decrypted = salsa20_xor(key, nonce, ciphertext, counter=0)
    assert decrypted == plaintext, "round-trip failed"

    ks_a = salsa20_keystream(key, nonce, 64)
    key_flipped = bytearray(key)
    key_flipped[0] ^= 0x01  # flip one bit
    ks_b = salsa20_keystream(bytes(key_flipped), nonce, 64)
    diff_bits = _hamming_bits(ks_a, ks_b)

    print("=== Salsa20 MVP ===")
    print(f"plaintext_len={len(plaintext)}")
    print(f"ciphertext_sha256={sha256(ciphertext).hexdigest()}")
    print(f"keystream_head_16={ks_a[:16].hex()}")
    print(f"decryption_ok={decrypted == plaintext}")
    print(f"avalanche_bits_in_64B={diff_bits}/512")


if __name__ == "__main__":
    main()

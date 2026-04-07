"""Educational ChaCha20 MVP (RFC 8439 style, 96-bit nonce).

This script implements ChaCha20 from source-level operations:
- quarter round
- 20-round ChaCha block function
- stream generation and XOR encryption/decryption

No interactive input is required.
"""

from __future__ import annotations

from typing import List


def _rotl32(x: int, n: int) -> int:
    """Rotate a 32-bit integer left by n bits."""
    return ((x << n) & 0xFFFFFFFF) | (x >> (32 - n))


def _quarter_round(state: List[int], a: int, b: int, c: int, d: int) -> None:
    """In-place ChaCha quarter round on state indices (a, b, c, d)."""
    state[a] = (state[a] + state[b]) & 0xFFFFFFFF
    state[d] ^= state[a]
    state[d] = _rotl32(state[d], 16)

    state[c] = (state[c] + state[d]) & 0xFFFFFFFF
    state[b] ^= state[c]
    state[b] = _rotl32(state[b], 12)

    state[a] = (state[a] + state[b]) & 0xFFFFFFFF
    state[d] ^= state[a]
    state[d] = _rotl32(state[d], 8)

    state[c] = (state[c] + state[d]) & 0xFFFFFFFF
    state[b] ^= state[c]
    state[b] = _rotl32(state[b], 7)


def _le_bytes_to_u32_words(data: bytes) -> List[int]:
    """Interpret bytes as little-endian 32-bit words."""
    if len(data) % 4 != 0:
        raise ValueError("length must be multiple of 4")
    return [int.from_bytes(data[i : i + 4], "little") for i in range(0, len(data), 4)]


def _u32_words_to_le_bytes(words: List[int]) -> bytes:
    """Serialize 32-bit words to little-endian bytes."""
    return b"".join((w & 0xFFFFFFFF).to_bytes(4, "little") for w in words)


def chacha20_block(key: bytes, counter: int, nonce: bytes) -> bytes:
    """Return one 64-byte ChaCha20 keystream block.

    Args:
        key: 32-byte key.
        counter: 32-bit block counter.
        nonce: 12-byte nonce (IETF variant).
    """
    if len(key) != 32:
        raise ValueError("ChaCha20 key must be 32 bytes")
    if len(nonce) != 12:
        raise ValueError("ChaCha20 nonce must be 12 bytes")
    if not (0 <= counter <= 0xFFFFFFFF):
        raise ValueError("counter must fit in uint32")

    constants = _le_bytes_to_u32_words(b"expand 32-byte k")
    key_words = _le_bytes_to_u32_words(key)
    nonce_words = _le_bytes_to_u32_words(nonce)

    initial_state = constants + key_words + [counter] + nonce_words
    working_state = initial_state.copy()

    # 20 rounds = 10 column+diagonal double rounds.
    for _ in range(10):
        # Column rounds
        _quarter_round(working_state, 0, 4, 8, 12)
        _quarter_round(working_state, 1, 5, 9, 13)
        _quarter_round(working_state, 2, 6, 10, 14)
        _quarter_round(working_state, 3, 7, 11, 15)
        # Diagonal rounds
        _quarter_round(working_state, 0, 5, 10, 15)
        _quarter_round(working_state, 1, 6, 11, 12)
        _quarter_round(working_state, 2, 7, 8, 13)
        _quarter_round(working_state, 3, 4, 9, 14)

    output_state = [
        (working_state[i] + initial_state[i]) & 0xFFFFFFFF for i in range(16)
    ]
    return _u32_words_to_le_bytes(output_state)


def chacha20_encrypt(key: bytes, nonce: bytes, plaintext: bytes, counter: int = 1) -> bytes:
    """Encrypt/decrypt bytes with ChaCha20 keystream XOR."""
    if counter < 0:
        raise ValueError("counter must be non-negative")

    out = bytearray(len(plaintext))
    block_counter = counter

    for offset in range(0, len(plaintext), 64):
        block = chacha20_block(key, block_counter, nonce)
        chunk = plaintext[offset : offset + 64]
        for i, b in enumerate(chunk):
            out[offset + i] = b ^ block[i]
        block_counter += 1
        if block_counter > 0xFFFFFFFF:
            raise OverflowError("ChaCha20 counter overflow")

    return bytes(out)


def _self_test_quarter_round() -> None:
    """RFC 8439 quarter-round test vector."""
    state = [0] * 16
    state[0] = 0x11111111
    state[1] = 0x01020304
    state[2] = 0x9B8D6F43
    state[3] = 0x01234567
    _quarter_round(state, 0, 1, 2, 3)
    assert state[0] == 0xEA2A92F4
    assert state[1] == 0xCB1CF8CE
    assert state[2] == 0x4581472E
    assert state[3] == 0x5881C4BB


def _self_test_block() -> None:
    """RFC 8439 block-function test vector."""
    key = bytes.fromhex(
        "000102030405060708090a0b0c0d0e0f"
        "101112131415161718191a1b1c1d1e1f"
    )
    nonce = bytes.fromhex("000000090000004a00000000")
    counter = 1
    expected = bytes.fromhex(
        "10f1e7e4d13b5915500fdd1fa32071c4"
        "c7d1f4c733c068030422aa9ac3d46c4e"
        "d2826446079faa0914c2d705d98b02a2"
        "b5129cd1de164eb9cbd083e8a2503c4e"
    )
    got = chacha20_block(key, counter, nonce)
    assert got == expected


def _self_test_encrypt_roundtrip() -> None:
    """Round-trip test with deterministic sample data."""
    key = bytes.fromhex("00" * 32)
    nonce = bytes.fromhex("000000000000000000000002")
    plaintext = (
        b"ChaCha20 is a stream cipher: ciphertext = plaintext XOR keystream. "
        b"This demo verifies round-trip correctness."
    )
    ciphertext = chacha20_encrypt(key, nonce, plaintext, counter=1)
    recovered = chacha20_encrypt(key, nonce, ciphertext, counter=1)
    assert recovered == plaintext


def main() -> None:
    _self_test_quarter_round()
    _self_test_block()
    _self_test_encrypt_roundtrip()

    key = bytes.fromhex(
        "00112233445566778899aabbccddeeff"
        "fedcba98765432100123456789abcdef"
    )
    nonce = bytes.fromhex("aabbccddeeff001122334455")
    plaintext = (
        b"Minimal ChaCha20 MVP for CS-0130. "
        b"It performs source-level quarter rounds and block-function mixing."
    )
    ciphertext = chacha20_encrypt(key, nonce, plaintext, counter=7)
    decrypted = chacha20_encrypt(key, nonce, ciphertext, counter=7)

    print("=== ChaCha20 MVP (RFC 8439 nonce/counter layout) ===")
    print(f"plaintext bytes: {len(plaintext)}")
    print(f"ciphertext bytes: {len(ciphertext)}")
    print(f"key (hex, truncated): {key.hex()[:24]}...")
    print(f"nonce (hex): {nonce.hex()}")
    print(f"ciphertext (hex, truncated): {ciphertext.hex()[:64]}...")
    print(f"decryption ok: {decrypted == plaintext}")

    if decrypted != plaintext:
        raise RuntimeError("ChaCha20 round-trip check failed")


if __name__ == "__main__":
    main()

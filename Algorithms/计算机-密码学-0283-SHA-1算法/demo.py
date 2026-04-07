"""Educational SHA-1 MVP.

This file implements SHA-1 from source-level operations:
- message padding
- message schedule expansion (16 -> 80 words)
- 80-round compression with Boolean functions and constants

No interactive input is required.
"""

from __future__ import annotations

import hashlib


def _left_rotate_32(value: int, bits: int) -> int:
    """Left-rotate a 32-bit integer."""
    return ((value << bits) | (value >> (32 - bits))) & 0xFFFFFFFF


def _pad_sha1_message(data: bytes) -> bytes:
    """Apply SHA-1 padding and return bytes whose length is a multiple of 64."""
    bit_length = (len(data) * 8) & 0xFFFFFFFFFFFFFFFF
    padded = data + b"\x80"
    while (len(padded) % 64) != 56:
        padded += b"\x00"
    padded += bit_length.to_bytes(8, "big")
    return padded


def sha1_digest(data: bytes) -> bytes:
    """Compute SHA-1 digest bytes (20 bytes) using a source-level implementation."""
    # Initial hash values (H0..H4), defined by SHA-1 standard.
    h0 = 0x67452301
    h1 = 0xEFCDAB89
    h2 = 0x98BADCFE
    h3 = 0x10325476
    h4 = 0xC3D2E1F0

    padded = _pad_sha1_message(data)

    for offset in range(0, len(padded), 64):
        chunk = padded[offset : offset + 64]
        w = [0] * 80

        # First 16 words come directly from this 512-bit block.
        for i in range(16):
            start = i * 4
            w[i] = int.from_bytes(chunk[start : start + 4], "big")

        # Extend to 80 words via XOR + rotate-left(1).
        for i in range(16, 80):
            w[i] = _left_rotate_32(
                w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16],
                1,
            )

        a, b, c, d, e = h0, h1, h2, h3, h4

        for t in range(80):
            if 0 <= t <= 19:
                f = (b & c) | ((~b) & d)
                k = 0x5A827999
            elif 20 <= t <= 39:
                f = b ^ c ^ d
                k = 0x6ED9EBA1
            elif 40 <= t <= 59:
                f = (b & c) | (b & d) | (c & d)
                k = 0x8F1BBCDC
            else:
                f = b ^ c ^ d
                k = 0xCA62C1D6

            temp = (_left_rotate_32(a, 5) + f + e + k + w[t]) & 0xFFFFFFFF
            e = d
            d = c
            c = _left_rotate_32(b, 30)
            b = a
            a = temp

        h0 = (h0 + a) & 0xFFFFFFFF
        h1 = (h1 + b) & 0xFFFFFFFF
        h2 = (h2 + c) & 0xFFFFFFFF
        h3 = (h3 + d) & 0xFFFFFFFF
        h4 = (h4 + e) & 0xFFFFFFFF

    return b"".join(
        h.to_bytes(4, "big")
        for h in (h0, h1, h2, h3, h4)
    )


def sha1_hexdigest(data: bytes) -> str:
    """Compute SHA-1 digest hex string (40 hex chars)."""
    return sha1_digest(data).hex()


def _self_test_known_vectors() -> None:
    """Known test vectors from FIPS 180-1 style examples."""
    vectors = [
        (b"", "da39a3ee5e6b4b0d3255bfef95601890afd80709"),
        (b"abc", "a9993e364706816aba3e25717850c26c9cd0d89d"),
        (
            b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
            "84983e441c3bd26ebaae4aa1f95129e5e54670f1",
        ),
    ]
    for message, expected_hex in vectors:
        got_hex = sha1_hexdigest(message)
        assert got_hex == expected_hex


def _self_test_against_hashlib() -> None:
    """Cross-check deterministic samples against Python hashlib.sha1."""
    samples = [
        b"SHA-1 minimal MVP for CS-0139",
        b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09",
        b"The quick brown fox jumps over the lazy dog",
        b"The quick brown fox jumps over the lazy cog",
        bytes(range(256)),
    ]
    for sample in samples:
        got = sha1_hexdigest(sample)
        expected = hashlib.sha1(sample).hexdigest()
        assert got == expected


def main() -> None:
    _self_test_known_vectors()
    _self_test_against_hashlib()

    message = (
        b"SHA-1 educational demo. "
        b"This digest is produced by a source-level implementation in demo.py."
    )
    digest_hex = sha1_hexdigest(message)
    hashlib_hex = hashlib.sha1(message).hexdigest()

    print("=== SHA-1 MVP (CS-0139) ===")
    print(f"message bytes: {len(message)}")
    print(f"sha1 (source-level): {digest_hex}")
    print(f"sha1 (hashlib):      {hashlib_hex}")
    print(f"match hashlib: {digest_hex == hashlib_hex}")

    if digest_hex != hashlib_hex:
        raise RuntimeError("SHA-1 mismatch between source-level implementation and hashlib")


if __name__ == "__main__":
    main()

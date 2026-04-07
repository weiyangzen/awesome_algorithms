"""Educational SHA-256 MVP.

This file provides a source-level SHA-256 implementation (FIPS 180-4 style):
- message padding
- message schedule expansion (16 -> 64 words)
- 64-round compression function

No interactive input is required.
"""

from __future__ import annotations

import hashlib

MASK32 = 0xFFFFFFFF

# SHA-256 round constants (first 32 bits of fractional parts of cube roots of primes).
K = [
    0x428A2F98,
    0x71374491,
    0xB5C0FBCF,
    0xE9B5DBA5,
    0x3956C25B,
    0x59F111F1,
    0x923F82A4,
    0xAB1C5ED5,
    0xD807AA98,
    0x12835B01,
    0x243185BE,
    0x550C7DC3,
    0x72BE5D74,
    0x80DEB1FE,
    0x9BDC06A7,
    0xC19BF174,
    0xE49B69C1,
    0xEFBE4786,
    0x0FC19DC6,
    0x240CA1CC,
    0x2DE92C6F,
    0x4A7484AA,
    0x5CB0A9DC,
    0x76F988DA,
    0x983E5152,
    0xA831C66D,
    0xB00327C8,
    0xBF597FC7,
    0xC6E00BF3,
    0xD5A79147,
    0x06CA6351,
    0x14292967,
    0x27B70A85,
    0x2E1B2138,
    0x4D2C6DFC,
    0x53380D13,
    0x650A7354,
    0x766A0ABB,
    0x81C2C92E,
    0x92722C85,
    0xA2BFE8A1,
    0xA81A664B,
    0xC24B8B70,
    0xC76C51A3,
    0xD192E819,
    0xD6990624,
    0xF40E3585,
    0x106AA070,
    0x19A4C116,
    0x1E376C08,
    0x2748774C,
    0x34B0BCB5,
    0x391C0CB3,
    0x4ED8AA4A,
    0x5B9CCA4F,
    0x682E6FF3,
    0x748F82EE,
    0x78A5636F,
    0x84C87814,
    0x8CC70208,
    0x90BEFFFA,
    0xA4506CEB,
    0xBEF9A3F7,
    0xC67178F2,
]


def _rotr32(x: int, n: int) -> int:
    """Rotate-right for 32-bit integer."""
    return ((x >> n) | (x << (32 - n))) & MASK32


def _pad_sha256_message(data: bytes) -> bytes:
    """Apply SHA-256 padding and return bytes whose length is a multiple of 64."""
    bit_length = (len(data) * 8) & 0xFFFFFFFFFFFFFFFF
    padded = data + b"\x80"
    while (len(padded) % 64) != 56:
        padded += b"\x00"
    padded += bit_length.to_bytes(8, "big")
    return padded


def sha256_digest(data: bytes) -> bytes:
    """Compute SHA-256 digest bytes (32 bytes) via source-level implementation."""
    # Initial hash values (first 32 bits of fractional parts of square roots of primes).
    h = [
        0x6A09E667,
        0xBB67AE85,
        0x3C6EF372,
        0xA54FF53A,
        0x510E527F,
        0x9B05688C,
        0x1F83D9AB,
        0x5BE0CD19,
    ]

    padded = _pad_sha256_message(data)

    for offset in range(0, len(padded), 64):
        chunk = padded[offset : offset + 64]
        w = [0] * 64

        for i in range(16):
            start = i * 4
            w[i] = int.from_bytes(chunk[start : start + 4], "big")

        for i in range(16, 64):
            s0 = _rotr32(w[i - 15], 7) ^ _rotr32(w[i - 15], 18) ^ (w[i - 15] >> 3)
            s1 = _rotr32(w[i - 2], 17) ^ _rotr32(w[i - 2], 19) ^ (w[i - 2] >> 10)
            w[i] = (w[i - 16] + s0 + w[i - 7] + s1) & MASK32

        a, b, c, d, e, f, g, hh = h

        for i in range(64):
            sum1 = _rotr32(e, 6) ^ _rotr32(e, 11) ^ _rotr32(e, 25)
            ch = (e & f) ^ ((~e) & g)
            temp1 = (hh + sum1 + ch + K[i] + w[i]) & MASK32

            sum0 = _rotr32(a, 2) ^ _rotr32(a, 13) ^ _rotr32(a, 22)
            maj = (a & b) ^ (a & c) ^ (b & c)
            temp2 = (sum0 + maj) & MASK32

            hh = g
            g = f
            f = e
            e = (d + temp1) & MASK32
            d = c
            c = b
            b = a
            a = (temp1 + temp2) & MASK32

        h[0] = (h[0] + a) & MASK32
        h[1] = (h[1] + b) & MASK32
        h[2] = (h[2] + c) & MASK32
        h[3] = (h[3] + d) & MASK32
        h[4] = (h[4] + e) & MASK32
        h[5] = (h[5] + f) & MASK32
        h[6] = (h[6] + g) & MASK32
        h[7] = (h[7] + hh) & MASK32

    return b"".join(word.to_bytes(4, "big") for word in h)


def sha256_hexdigest(data: bytes) -> str:
    """Compute SHA-256 digest hex string (64 hex chars)."""
    return sha256_digest(data).hex()


def _self_test_known_vectors() -> None:
    """Known vectors commonly listed in FIPS 180-4 references."""
    vectors = [
        (
            b"",
            "e3b0c44298fc1c149afbf4c8996fb924"
            "27ae41e4649b934ca495991b7852b855",
        ),
        (
            b"abc",
            "ba7816bf8f01cfea414140de5dae2223"
            "b00361a396177a9cb410ff61f20015ad",
        ),
        (
            b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
            "248d6a61d20638b8e5c026930c3e6039"
            "a33ce45964ff2167f6ecedd419db06c1",
        ),
    ]
    for message, expected_hex in vectors:
        got_hex = sha256_hexdigest(message)
        assert got_hex == expected_hex


def _self_test_against_hashlib() -> None:
    """Cross-check deterministic samples against hashlib.sha256."""
    samples = [
        b"SHA-256 minimal MVP for CS-0137",
        b"The quick brown fox jumps over the lazy dog",
        b"The quick brown fox jumps over the lazy dog.",
        bytes(range(256)),
        b"\x00\x00\x00\x00\xff\xfe\xfd\xfc",
    ]
    for sample in samples:
        got = sha256_hexdigest(sample)
        expected = hashlib.sha256(sample).hexdigest()
        assert got == expected


def main() -> None:
    _self_test_known_vectors()
    _self_test_against_hashlib()

    message = (
        b"SHA-256 educational demo. "
        b"This digest is produced by a source-level implementation in demo.py."
    )
    digest_hex = sha256_hexdigest(message)
    hashlib_hex = hashlib.sha256(message).hexdigest()

    print("=== SHA-256 MVP (CS-0137) ===")
    print(f"message bytes: {len(message)}")
    print(f"sha256 (source-level): {digest_hex}")
    print(f"sha256 (hashlib):      {hashlib_hex}")
    print(f"match hashlib: {digest_hex == hashlib_hex}")

    if digest_hex != hashlib_hex:
        raise RuntimeError("SHA-256 mismatch between source-level implementation and hashlib")


if __name__ == "__main__":
    main()

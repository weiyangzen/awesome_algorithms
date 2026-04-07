"""Minimal runnable MVP for bcrypt (CS-0143).

This implementation is educational and source-visible:
- pure Python EksBlowfish key schedule + 16-round Blowfish core
- custom bcrypt base64 encoding/decoding
- bcrypt hash string generation and verification

Notes:
- It targets the bcrypt 2b string format.
- It is designed for algorithm understanding, not hardened production use.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, getcontext
import hashlib
import hmac
import math
import os
import re
import time
from functools import lru_cache

import numpy as np

BCRYPT_ALPHABET = "./ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
B64_INDEX = {ch: i for i, ch in enumerate(BCRYPT_ALPHABET)}
MASK32 = 0xFFFFFFFF
MAGIC_TEXT = b"OrpheanBeholderScryDoubt"
HASH_RE = re.compile(r"^\$(2[abxy])\$(\d\d)\$([./A-Za-z0-9]{22})([./A-Za-z0-9]{31})$")


@dataclass
class EksBlowfishState:
    p: list[int]
    s: list[list[int]]

    @classmethod
    def fresh(cls) -> "EksBlowfishState":
        words = blowfish_words_from_pi()
        p = list(words[:18])
        s = [list(words[18 + i * 256 : 18 + (i + 1) * 256]) for i in range(4)]
        return cls(p=p, s=s)

    def f(self, value: int) -> int:
        a = (value >> 24) & 0xFF
        b = (value >> 16) & 0xFF
        c = (value >> 8) & 0xFF
        d = value & 0xFF
        out = (self.s[0][a] + self.s[1][b]) & MASK32
        out ^= self.s[2][c]
        out = (out + self.s[3][d]) & MASK32
        return out

    def encipher(self, left: int, right: int) -> tuple[int, int]:
        left &= MASK32
        right &= MASK32

        left ^= self.p[0]
        for i in range(0, 16, 2):
            right ^= self.f(left) ^ self.p[i + 1]
            right &= MASK32
            left ^= self.f(right) ^ self.p[i + 2]
            left &= MASK32
        right ^= self.p[17]
        right &= MASK32

        return right, left


@lru_cache(maxsize=1)
def blowfish_words_from_pi() -> tuple[int, ...]:
    """Generate Blowfish P/S initialization words from hex digits of pi.

    Blowfish uses 18 + 4*256 = 1042 32-bit words.
    """

    total_words = 1042
    hex_digits = total_words * 8
    decimal_digits = int(math.ceil(hex_digits * math.log10(16))) + 30

    pi_value = chudnovsky_pi(decimal_digits)
    scale = Decimal(16) ** hex_digits
    scaled = int(pi_value * scale)

    hex_stream = format(scaled, "x")
    needed = 1 + hex_digits
    if len(hex_stream) < needed:
        hex_stream = hex_stream.rjust(needed, "0")
    hex_stream = hex_stream[1:needed]

    words = [int(hex_stream[i : i + 8], 16) for i in range(0, len(hex_stream), 8)]
    if len(words) != total_words:
        raise RuntimeError(f"expected {total_words} words, got {len(words)}")
    return tuple(words)


def chudnovsky_pi(decimal_digits: int) -> Decimal:
    """Compute pi with Chudnovsky formula to the requested decimal precision."""
    getcontext().prec = decimal_digits + 10

    c = 426880 * Decimal(10005).sqrt()
    m = 1
    l = 13591409
    x = 1
    k = 6
    summation = Decimal(l)

    # Each Chudnovsky term contributes about 14 decimal digits.
    for i in range(1, decimal_digits // 14 + 2):
        m = (m * (k**3 - 16 * k)) // (i**3)
        l += 545140134
        x *= -262537412640768000
        summation += Decimal(m * l) / x
        k += 12

    pi_value = c / summation
    getcontext().prec = decimal_digits
    return +pi_value


def bcrypt_b64_encode(data: bytes) -> str:
    """Encode bytes with bcrypt's custom base64 alphabet (no padding)."""
    out: list[str] = []
    i = 0
    data_len = len(data)

    while i < data_len:
        c1 = data[i]
        i += 1
        out.append(BCRYPT_ALPHABET[(c1 >> 2) & 0x3F])
        c1 = (c1 & 0x03) << 4

        if i >= data_len:
            out.append(BCRYPT_ALPHABET[c1 & 0x3F])
            break

        c2 = data[i]
        i += 1
        c1 |= (c2 >> 4) & 0x0F
        out.append(BCRYPT_ALPHABET[c1 & 0x3F])
        c1 = (c2 & 0x0F) << 2

        if i >= data_len:
            out.append(BCRYPT_ALPHABET[c1 & 0x3F])
            break

        c2 = data[i]
        i += 1
        c1 |= (c2 >> 6) & 0x03
        out.append(BCRYPT_ALPHABET[c1 & 0x3F])
        out.append(BCRYPT_ALPHABET[c2 & 0x3F])

    return "".join(out)


def bcrypt_b64_decode(text: str, max_bytes: int) -> bytes:
    """Decode bcrypt custom base64 string to bytes with explicit length bound."""
    out = bytearray()
    off = 0

    while off < len(text) and len(out) < max_bytes:
        if off + 1 >= len(text):
            break
        c1 = B64_INDEX[text[off]]
        c2 = B64_INDEX[text[off + 1]]
        off += 2

        out.append(((c1 << 2) | ((c2 & 0x30) >> 4)) & 0xFF)
        if len(out) >= max_bytes or off >= len(text):
            break

        c3 = B64_INDEX[text[off]]
        off += 1
        out.append((((c2 & 0x0F) << 4) | ((c3 & 0x3C) >> 2)) & 0xFF)
        if len(out) >= max_bytes or off >= len(text):
            break

        c4 = B64_INDEX[text[off]]
        off += 1
        out.append((((c3 & 0x03) << 6) | c4) & 0xFF)

    return bytes(out[:max_bytes])


def stream_to_word(data: bytes, offset: int) -> tuple[int, int]:
    if not data:
        raise ValueError("stream_to_word requires non-empty data")

    word = 0
    local_offset = offset
    for _ in range(4):
        word = ((word << 8) | data[local_offset]) & MASK32
        local_offset = (local_offset + 1) % len(data)
    return word, local_offset


def expand_state(state: EksBlowfishState, salt: bytes, key: bytes) -> None:
    """EksBlowfish expand with salt and key."""
    key_off = 0
    salt_off = 0

    for i in range(18):
        w, key_off = stream_to_word(key, key_off)
        state.p[i] ^= w

    left = 0
    right = 0
    for i in range(0, 18, 2):
        sw, salt_off = stream_to_word(salt, salt_off)
        left ^= sw
        sw, salt_off = stream_to_word(salt, salt_off)
        right ^= sw
        left, right = state.encipher(left, right)
        state.p[i] = left
        state.p[i + 1] = right

    for box in range(4):
        for i in range(0, 256, 2):
            sw, salt_off = stream_to_word(salt, salt_off)
            left ^= sw
            sw, salt_off = stream_to_word(salt, salt_off)
            right ^= sw
            left, right = state.encipher(left, right)
            state.s[box][i] = left
            state.s[box][i + 1] = right


def expand0_state(state: EksBlowfishState, key: bytes) -> None:
    """EksBlowfish expand with key only (zero-salt path)."""
    key_off = 0

    for i in range(18):
        w, key_off = stream_to_word(key, key_off)
        state.p[i] ^= w

    left = 0
    right = 0
    for i in range(0, 18, 2):
        left, right = state.encipher(left, right)
        state.p[i] = left
        state.p[i + 1] = right

    for box in range(4):
        for i in range(0, 256, 2):
            left, right = state.encipher(left, right)
            state.s[box][i] = left
            state.s[box][i + 1] = right


def bcrypt_raw(password: bytes, salt: bytes, cost: int) -> bytes:
    """Compute raw bcrypt checksum (23 bytes)."""
    if len(salt) != 16:
        raise ValueError(f"bcrypt salt must be 16 bytes, got {len(salt)}")
    if not (4 <= cost <= 16):
        raise ValueError(f"cost must be in [4, 16] for this MVP, got {cost}")

    key = password[:72] + b"\x00"
    state = EksBlowfishState.fresh()

    expand_state(state, salt=salt, key=key)
    rounds = 1 << cost
    for _ in range(rounds):
        expand0_state(state, key)
        expand0_state(state, salt)

    cdata = [
        int.from_bytes(MAGIC_TEXT[i : i + 4], byteorder="big")
        for i in range(0, len(MAGIC_TEXT), 4)
    ]

    for _ in range(64):
        for i in range(0, len(cdata), 2):
            cdata[i], cdata[i + 1] = state.encipher(cdata[i], cdata[i + 1])

    output = bytearray()
    for word in cdata:
        output.extend(word.to_bytes(4, byteorder="big"))

    return bytes(output[:23])


def bcrypt_hash(password: str, cost: int = 6, salt: bytes | None = None, version: str = "2b") -> str:
    if version not in {"2a", "2b", "2x", "2y"}:
        raise ValueError(f"unsupported bcrypt version: {version}")

    if salt is None:
        salt = os.urandom(16)
    if len(salt) != 16:
        raise ValueError("salt must be exactly 16 bytes")

    password_bytes = password.encode("utf-8")
    raw = bcrypt_raw(password=password_bytes, salt=salt, cost=cost)

    salt_enc = bcrypt_b64_encode(salt)
    hash_enc = bcrypt_b64_encode(raw)
    if len(salt_enc) != 22 or len(hash_enc) != 31:
        raise RuntimeError("bcrypt encoding length mismatch")

    return f"${version}${cost:02d}${salt_enc}{hash_enc}"


def parse_bcrypt_hash(hash_text: str) -> tuple[str, int, bytes, bytes]:
    m = HASH_RE.match(hash_text)
    if not m:
        raise ValueError("invalid bcrypt hash format")

    version = m.group(1)
    cost = int(m.group(2))
    salt_enc = m.group(3)
    hash_enc = m.group(4)

    salt = bcrypt_b64_decode(salt_enc, 16)
    digest = bcrypt_b64_decode(hash_enc, 23)

    if len(salt) != 16 or len(digest) != 23:
        raise ValueError("invalid bcrypt payload lengths after decode")

    return version, cost, salt, digest


def bcrypt_verify(password: str, hash_text: str) -> bool:
    version, cost, salt, _ = parse_bcrypt_hash(hash_text)
    candidate = bcrypt_hash(password=password, cost=cost, salt=salt, version=version)
    return hmac.compare_digest(candidate, hash_text)


def run_deterministic_checks() -> None:
    print("== Deterministic bcrypt checks ==")
    salt = bytes.fromhex("00112233445566778899aabbccddeeff")
    cost = 5

    h1 = bcrypt_hash(password="correct horse battery staple", cost=cost, salt=salt)
    h2 = bcrypt_hash(password="correct horse battery staple", cost=cost, salt=salt)
    h3 = bcrypt_hash(password="correct horse battery staple!", cost=cost, salt=salt)

    print(f"hash (cost={cost}): {h1}")
    print(f"stable under same input: {h1 == h2}")
    print(f"changed by password mutation: {h1 != h3}")
    print(f"verify(correct): {bcrypt_verify('correct horse battery staple', h1)}")
    print(f"verify(wrong)  : {bcrypt_verify('wrong password', h1)}")

    version, parsed_cost, parsed_salt, parsed_digest = parse_bcrypt_hash(h1)
    print(
        "parsed => "
        f"version={version}, cost={parsed_cost}, salt={parsed_salt.hex()}, "
        f"digest_prefix={parsed_digest.hex()[:16]}"
    )

    if h1 != h2:
        raise RuntimeError("determinism check failed")
    if h1 == h3:
        raise RuntimeError("password sensitivity check failed")
    if not bcrypt_verify("correct horse battery staple", h1):
        raise RuntimeError("verification should succeed for correct password")
    if bcrypt_verify("wrong password", h1):
        raise RuntimeError("verification should fail for wrong password")


def run_cost_benchmark() -> None:
    print("\n== Cost-factor benchmark (single-process) ==")
    salt = bytes.fromhex("0f1e2d3c4b5a69788796a5b4c3d2e1f0")
    password = "benchmark-password"
    costs = [4, 5, 6]

    times_ms = []
    for c in costs:
        t0 = time.perf_counter()
        _ = bcrypt_hash(password=password, cost=c, salt=salt)
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000.0
        times_ms.append(elapsed_ms)
        print(f"cost={c:02d} -> {elapsed_ms:.2f} ms")

    arr = np.array(times_ms, dtype=np.float64)
    ratios = arr[1:] / arr[:-1]

    print(f"mean time: {float(np.mean(arr)):.2f} ms")
    print(
        "adjacent ratios: "
        + ", ".join(f"{costs[i]}->{costs[i+1]}: {ratios[i]:.2f}x" for i in range(len(ratios)))
    )

    if not np.all(arr[1:] > arr[:-1]):
        raise RuntimeError("cost benchmark is not monotonic; unexpected on this run")


def run_salt_uniqueness_check() -> None:
    print("\n== Salt uniqueness check ==")
    password = "same-password"
    cost = 5

    salt_a = hashlib.sha256(b"salt-A").digest()[:16]
    salt_b = hashlib.sha256(b"salt-B").digest()[:16]

    hash_a = bcrypt_hash(password=password, cost=cost, salt=salt_a)
    hash_b = bcrypt_hash(password=password, cost=cost, salt=salt_b)
    print(f"salt A hash: {hash_a}")
    print(f"salt B hash: {hash_b}")
    print(f"different salts produce different hashes: {hash_a != hash_b}")

    if hash_a == hash_b:
        raise RuntimeError("salt uniqueness check failed")


def main() -> None:
    run_deterministic_checks()
    run_salt_uniqueness_check()
    run_cost_benchmark()
    print("\nall bcrypt checks passed")


if __name__ == "__main__":
    main()

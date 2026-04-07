"""Minimal runnable MVP for HMAC (CS-0141).

This script provides a non-black-box HMAC implementation, then validates it with:
- RFC 4231 HMAC-SHA256 vectors
- Python stdlib reference `hmac.digest`
"""

from __future__ import annotations

import hashlib
import hmac
import time


def _normalize_key(hash_name: str, key: bytes) -> bytes:
    """Normalize key to hash block size for HMAC processing."""
    if not key:
        raise ValueError("key must be non-empty for this MVP")

    block_size = hashlib.new(hash_name).block_size
    if block_size <= 0:
        raise ValueError(f"invalid block_size for hash {hash_name}")

    normalized = key
    if len(normalized) > block_size:
        normalized = hashlib.new(hash_name, normalized).digest()
    if len(normalized) < block_size:
        normalized = normalized + (b"\x00" * (block_size - len(normalized)))
    return normalized


def _xor_with_byte(data: bytes, value: int) -> bytes:
    """XOR each byte in data with the same byte value."""
    return bytes(b ^ value for b in data)


def hmac_manual(hash_name: str, key: bytes, message: bytes) -> bytes:
    """Manual HMAC implementation following RFC 2104 / FIPS 198-1."""
    key_block = _normalize_key(hash_name, key)
    inner_key = _xor_with_byte(key_block, 0x36)
    outer_key = _xor_with_byte(key_block, 0x5C)

    inner_digest = hashlib.new(hash_name, inner_key + message).digest()
    return hashlib.new(hash_name, outer_key + inner_digest).digest()


def hmac_reference(hash_name: str, key: bytes, message: bytes) -> bytes:
    """Reference implementation from Python stdlib."""
    return hmac.digest(key, message, hash_name)


def run_rfc4231_sha256_vectors() -> None:
    """Validate manual HMAC-SHA256 against RFC 4231 vectors."""
    vectors = [
        {
            "key": bytes.fromhex("0b" * 20),
            "message": b"Hi There",
            "expected_hex": (
                "b0344c61d8db38535ca8afceaf0bf12b"
                "881dc200c9833da726e9376c2e32cff7"
            ),
        },
        {
            "key": b"Jefe",
            "message": b"what do ya want for nothing?",
            "expected_hex": (
                "5bdcc146bf60754e6a042426089575c7"
                "5a003f089d2739839dec58b964ec3843"
            ),
        },
        {
            "key": bytes.fromhex("aa" * 20),
            "message": bytes.fromhex("dd" * 50),
            "expected_hex": (
                "773ea91e36800e46854db8ebd09181a7"
                "2959098b3ef8c122d9635514ced565fe"
            ),
        },
        {
            "key": bytes(range(1, 26)),
            "message": bytes.fromhex("cd" * 50),
            "expected_hex": (
                "82558a389a443c0ea4cc819899f2083a"
                "85f0faa3e578f8077a2e3ff46729665b"
            ),
        },
        {
            "key": bytes.fromhex("aa" * 131),
            "message": b"Test Using Larger Than Block-Size Key - Hash Key First",
            "expected_hex": (
                "60e431591ee0b67f0d8a26aacbf5b77f"
                "8e0bc6213728c5140546040f0ee37f54"
            ),
        },
        {
            "key": bytes.fromhex("aa" * 131),
            "message": (
                b"This is a test using a larger than block-size key and a larger "
                b"than block-size data. The key needs to be hashed before being "
                b"used by the HMAC algorithm."
            ),
            "expected_hex": (
                "9b09ffa71b942fcb27635fbcd5b0e944"
                "bfdc63644f0713938a7f51535c3a35e2"
            ),
        },
    ]

    print("== RFC 4231 vectors (HMAC-SHA256) ==")
    for idx, vector in enumerate(vectors, start=1):
        actual = hmac_manual("sha256", vector["key"], vector["message"]).hex()
        ok = actual == vector["expected_hex"]
        print(f"vector {idx}: pass={ok}")
        if not ok:
            raise RuntimeError(
                f"RFC 4231 vector {idx} mismatch: expected {vector['expected_hex']}, got {actual}"
            )


def run_cross_checks() -> None:
    """Cross-check manual HMAC against stdlib over multiple hash functions."""
    cases = [
        {
            "hash_name": "sha256",
            "key": b"demo-key-256",
            "message": b"message for hmac cross-check",
        },
        {
            "hash_name": "sha1",
            "key": b"demo-key-1",
            "message": b"legacy compatibility path",
        },
        {
            "hash_name": "sha512",
            "key": b"demo-key-512",
            "message": b"longer digest experiment",
        },
    ]

    print("\n== Cross-check vs hmac.digest ==")
    for case in cases:
        hash_name = str(case["hash_name"])
        key = bytes(case["key"])
        message = bytes(case["message"])

        t0 = time.perf_counter()
        manual = hmac_manual(hash_name, key, message)
        t1 = time.perf_counter()
        reference = hmac_reference(hash_name, key, message)
        t2 = time.perf_counter()

        same = hmac.compare_digest(manual, reference)
        changed_message = not hmac.compare_digest(
            manual, hmac_manual(hash_name, key, message + b"!")
        )
        changed_key = not hmac.compare_digest(
            manual, hmac_manual(hash_name, key + b"!", message)
        )

        manual_us = (t1 - t0) * 1e6
        reference_us = (t2 - t1) * 1e6

        print(f"hash={hash_name}, match={same}, msg_sensitive={changed_message}, key_sensitive={changed_key}")
        print(f"manual={manual.hex()}")
        print(f"stdlib={reference.hex()}")
        print(f"timing_us(manual/std)={manual_us:.1f}/{reference_us:.1f}")

        if not all([same, changed_message, changed_key]):
            raise RuntimeError(f"cross-check failed for {hash_name}")


def main() -> None:
    run_rfc4231_sha256_vectors()
    run_cross_checks()
    print("\nall HMAC checks passed")


if __name__ == "__main__":
    main()

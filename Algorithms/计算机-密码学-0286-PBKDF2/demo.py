"""Minimal runnable MVP for PBKDF2 (CS-0142).

This script implements PBKDF2-HMAC manually, then verifies it against:
- RFC 6070 known vectors (PBKDF2-HMAC-SHA1)
- Python stdlib reference `hashlib.pbkdf2_hmac`
"""

from __future__ import annotations

import hashlib
import hmac
import math
import time


def _prf_hmac(hash_name: str, key: bytes, data: bytes) -> bytes:
    """PRF primitive used by PBKDF2: HMAC(hash_name, key, data)."""
    return hmac.new(key, data, hash_name).digest()


def _xor_bytes(left: bytes, right: bytes) -> bytes:
    """XOR two equal-length byte strings."""
    if len(left) != len(right):
        raise ValueError("XOR inputs must have equal length")
    return bytes(a ^ b for a, b in zip(left, right))


def pbkdf2_hmac_manual(
    hash_name: str,
    password: bytes,
    salt: bytes,
    iterations: int,
    dklen: int,
) -> bytes:
    """Manual PBKDF2-HMAC implementation from RFC 8018."""
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if dklen <= 0:
        raise ValueError("dklen must be positive")
    if not password:
        raise ValueError("password must be non-empty for this MVP")
    if not salt:
        raise ValueError("salt must be non-empty for this MVP")

    hlen = hashlib.new(hash_name).digest_size
    blocks = math.ceil(dklen / hlen)
    if blocks > 0xFFFFFFFF:
        raise ValueError("derived key too long")

    output = bytearray()
    for block_index in range(1, blocks + 1):
        block_id = block_index.to_bytes(4, byteorder="big")
        u = _prf_hmac(hash_name, password, salt + block_id)
        t = u
        for _ in range(2, iterations + 1):
            u = _prf_hmac(hash_name, password, u)
            t = _xor_bytes(t, u)
        output.extend(t)

    return bytes(output[:dklen])


def pbkdf2_hmac_reference(
    hash_name: str,
    password: bytes,
    salt: bytes,
    iterations: int,
    dklen: int,
) -> bytes:
    """Reference PBKDF2 implementation from Python stdlib."""
    return hashlib.pbkdf2_hmac(hash_name, password, salt, iterations, dklen)


def run_rfc6070_vectors() -> None:
    """Validate manual implementation against RFC 6070 test vectors."""
    vectors = [
        {
            "password": b"password",
            "salt": b"salt",
            "iterations": 1,
            "dklen": 20,
            "expected_hex": "0c60c80f961f0e71f3a9b524af6012062fe037a6",
        },
        {
            "password": b"password",
            "salt": b"salt",
            "iterations": 2,
            "dklen": 20,
            "expected_hex": "ea6c014dc72d6f8ccd1ed92ace1d41f0d8de8957",
        },
        {
            "password": b"password",
            "salt": b"salt",
            "iterations": 4096,
            "dklen": 20,
            "expected_hex": "4b007901b765489abead49d926f721d065a429c1",
        },
        {
            "password": b"passwordPASSWORDpassword",
            "salt": b"saltSALTsaltSALTsaltSALTsaltSALTsalt",
            "iterations": 4096,
            "dklen": 25,
            "expected_hex": "3d2eec4fe41c849b80c8d83662c0e44a8b291a964cf2f07038",
        },
        {
            "password": b"pass\x00word",
            "salt": b"sa\x00lt",
            "iterations": 4096,
            "dklen": 16,
            "expected_hex": "56fa6aa75548099dcc37d7f03425e0c3",
        },
    ]

    print("== RFC 6070 vectors (PBKDF2-HMAC-SHA1) ==")
    for idx, vector in enumerate(vectors, start=1):
        actual = pbkdf2_hmac_manual(
            hash_name="sha1",
            password=vector["password"],
            salt=vector["salt"],
            iterations=int(vector["iterations"]),
            dklen=int(vector["dklen"]),
        ).hex()
        ok = actual == vector["expected_hex"]
        print(f"vector {idx}: iterations={vector['iterations']}, pass={ok}")
        if not ok:
            raise RuntimeError(
                f"RFC vector {idx} mismatch: expected {vector['expected_hex']}, got {actual}"
            )


def run_cross_checks() -> None:
    """Compare manual implementation with hashlib.pbkdf2_hmac."""
    hash_name = "sha256"
    password = b"Tr0ub4dor&3"
    salt = bytes.fromhex("00112233445566778899aabbccddeeff")
    iterations = 12000
    dklen = 32

    t0 = time.perf_counter()
    manual = pbkdf2_hmac_manual(hash_name, password, salt, iterations, dklen)
    t1 = time.perf_counter()
    reference = pbkdf2_hmac_reference(hash_name, password, salt, iterations, dklen)
    t2 = time.perf_counter()

    same = hmac.compare_digest(manual, reference)

    mutated_password = b"Tr0ub4dor&4"
    mutated_salt = salt[:-1] + b"\x00"
    changed_by_password = manual != pbkdf2_hmac_manual(
        hash_name, mutated_password, salt, iterations, dklen
    )
    changed_by_salt = manual != pbkdf2_hmac_manual(
        hash_name, password, mutated_salt, iterations, dklen
    )

    manual_ms = (t1 - t0) * 1000.0
    reference_ms = (t2 - t1) * 1000.0

    print("\n== Cross-check vs hashlib.pbkdf2_hmac ==")
    print(f"hash={hash_name}, iterations={iterations}, dklen={dklen}")
    print(f"salt(hex)={salt.hex()}")
    print(f"manual  : {manual.hex()}")
    print(f"hashlib : {reference.hex()}")
    print(f"manual == hashlib: {same}")
    print(f"password sensitivity: {changed_by_password}")
    print(f"salt sensitivity: {changed_by_salt}")
    print(f"manual time: {manual_ms:.2f} ms")
    print(f"hashlib time: {reference_ms:.2f} ms")

    if not all([same, changed_by_password, changed_by_salt]):
        raise RuntimeError("PBKDF2 cross-check failed")


def main() -> None:
    run_rfc6070_vectors()
    run_cross_checks()
    print("\nall PBKDF2 checks passed")


if __name__ == "__main__":
    main()

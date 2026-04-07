"""Minimal runnable MVP for scrypt (CS-0144).

This script demonstrates:
- deterministic key derivation for the same password/salt/params
- password verification via constant-time comparison
- sensitivity to password and salt changes

It uses Python's standard-library `hashlib.scrypt` binding.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import time


@dataclass(frozen=True)
class ScryptParams:
    """Scrypt work-factor parameters."""

    n: int
    r: int
    p: int
    dklen: int = 32


def _validate_params(params: ScryptParams) -> None:
    """Validate scrypt parameters before calling hashlib.scrypt."""
    if params.n < 2 or (params.n & (params.n - 1)) != 0:
        raise ValueError("n must be a power of two and >= 2")
    if params.r <= 0:
        raise ValueError("r must be positive")
    if params.p <= 0:
        raise ValueError("p must be positive")
    if params.dklen <= 0:
        raise ValueError("dklen must be positive")


def estimate_memory_bytes(params: ScryptParams) -> int:
    """Approximate ROMix memory usage from RFC expression 128 * r * N."""
    _validate_params(params)
    return 128 * params.r * params.n


def derive_key(password: str, salt: bytes, params: ScryptParams) -> bytes:
    """Derive key bytes using scrypt(password, salt, N, r, p, dkLen)."""
    _validate_params(params)
    if not password:
        raise ValueError("password must be non-empty for this MVP")
    if not salt:
        raise ValueError("salt must be non-empty")

    return hashlib.scrypt(
        password=password.encode("utf-8"),
        salt=salt,
        n=params.n,
        r=params.r,
        p=params.p,
        dklen=params.dklen,
    )


def build_record(password: str, salt: bytes, params: ScryptParams) -> dict[str, str | int]:
    """Create a serializable record containing params, salt, and derived key."""
    dk = derive_key(password, salt, params)
    return {
        "algorithm": "scrypt",
        "n": params.n,
        "r": params.r,
        "p": params.p,
        "dklen": params.dklen,
        "salt_hex": salt.hex(),
        "derived_key_hex": dk.hex(),
    }


def verify_password(record: dict[str, str | int], candidate_password: str) -> bool:
    """Verify password against stored scrypt record using constant-time compare."""
    params = ScryptParams(
        n=int(record["n"]),
        r=int(record["r"]),
        p=int(record["p"]),
        dklen=int(record["dklen"]),
    )
    salt = bytes.fromhex(str(record["salt_hex"]))
    expected = bytes.fromhex(str(record["derived_key_hex"]))
    actual = derive_key(candidate_password, salt, params)
    return hmac.compare_digest(actual, expected)


def bit_distance(a: bytes, b: bytes) -> int:
    """Return Hamming distance in bits for two equal-length byte strings."""
    if len(a) != len(b):
        raise ValueError("inputs must have equal length")
    return sum((x ^ y).bit_count() for x, y in zip(a, b))


def main() -> None:
    params = ScryptParams(n=1 << 14, r=8, p=1, dklen=32)
    password = "correct horse battery staple"
    salt = bytes.fromhex("00112233445566778899aabbccddeeff")

    t0 = time.perf_counter()
    dk1 = derive_key(password, salt, params)
    t1 = time.perf_counter()

    dk2 = derive_key(password, salt, params)
    dk_wrong_pw = derive_key("correct horse battery stap1e", salt, params)
    dk_wrong_salt = derive_key(password, salt[:-1] + b"\x00", params)

    record = build_record(password, salt, params)
    ok_verify = verify_password(record, password)
    ok_reject = not verify_password(record, "Tr0ub4dor&3")

    deterministic_ok = dk1 == dk2
    pw_sensitivity_ok = dk1 != dk_wrong_pw
    salt_sensitivity_ok = dk1 != dk_wrong_salt

    dist_wrong_pw = bit_distance(dk1, dk_wrong_pw)
    dist_wrong_salt = bit_distance(dk1, dk_wrong_salt)

    elapsed_ms = (t1 - t0) * 1000.0
    mem_mb = estimate_memory_bytes(params) / (1024 * 1024)

    print("=== scrypt MVP (CS-0144) ===")
    print(f"params: N={params.n}, r={params.r}, p={params.p}, dkLen={params.dklen}")
    print(f"estimated ROMix memory: {mem_mb:.2f} MiB")
    print(f"derive latency (single run): {elapsed_ms:.2f} ms")
    print(f"salt(hex): {salt.hex()}")
    print(f"derived key #1: {dk1.hex()}")
    print(f"derived key #2: {dk2.hex()}")
    print(f"deterministic check: {deterministic_ok}")
    print(f"password sensitivity check: {pw_sensitivity_ok} (bit distance={dist_wrong_pw})")
    print(f"salt sensitivity check: {salt_sensitivity_ok} (bit distance={dist_wrong_salt})")
    print(f"verify(correct password): {ok_verify}")
    print(f"verify(wrong password): {not ok_reject}")

    if not all(
        [
            deterministic_ok,
            pw_sensitivity_ok,
            salt_sensitivity_ok,
            ok_verify,
            ok_reject,
        ]
    ):
        raise RuntimeError("scrypt MVP checks failed")


if __name__ == "__main__":
    main()

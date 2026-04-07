"""Educational Argon2-style MVP (memory-hard password hashing).

This is a source-level teaching implementation that keeps Argon2's core ideas:
- parameterized memory-hard filling (m, t, p)
- lane-based block matrix (1 KiB blocks)
- PHC-style encoded output and password verification

It is intentionally simplified and is not a full RFC 9106-compatible engine.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import time
from dataclasses import dataclass

BLOCK_SIZE = 1024
ARGON2ID_TYPE_CODE = 2


@dataclass(frozen=True)
class Argon2Params:
    """Core tunable parameters for the Argon2-style MVP."""

    time_cost: int
    memory_cost_kib: int
    parallelism: int
    hash_len: int = 32
    version: int = 19
    variant: str = "argon2id"


def _le32(value: int) -> bytes:
    """Encode integer as little-endian uint32."""
    if not (0 <= value <= 0xFFFFFFFF):
        raise ValueError(f"uint32 overflow: {value}")
    return value.to_bytes(4, "little")


def _xor_bytes(a: bytes, b: bytes) -> bytes:
    """Bytewise XOR for equal-length buffers."""
    if len(a) != len(b):
        raise ValueError("xor inputs must have the same length")
    return bytes(x ^ y for x, y in zip(a, b))


def _expand_hash(data: bytes, out_len: int) -> bytes:
    """Extend BLAKE2b output to arbitrary length in a deterministic way."""
    if out_len <= 0:
        raise ValueError("out_len must be positive")

    out = bytearray()
    counter = 0
    while len(out) < out_len:
        chunk = hashlib.blake2b(
            _le32(counter) + _le32(out_len) + data,
            digest_size=64,
        ).digest()
        take = min(64, out_len - len(out))
        out.extend(chunk[:take])
        counter += 1
    return bytes(out)


def _validate_params(params: Argon2Params, salt: bytes) -> None:
    """Validate Argon2 parameters and salt constraints."""
    if params.variant != "argon2id":
        raise ValueError("this MVP currently supports variant='argon2id' only")
    if params.time_cost < 1:
        raise ValueError("time_cost must be >= 1")
    if params.parallelism < 1:
        raise ValueError("parallelism must be >= 1")
    if params.memory_cost_kib < 8 * params.parallelism:
        raise ValueError("memory_cost_kib must be >= 8 * parallelism")
    if not (4 <= params.hash_len <= 1024):
        raise ValueError("hash_len must be in [4, 1024]")
    if len(salt) < 8:
        raise ValueError("salt must be at least 8 bytes")


def _memory_layout(params: Argon2Params) -> tuple[int, int]:
    """Return (lane_length, total_blocks) aligned by parallelism."""
    lane_length = params.memory_cost_kib // params.parallelism
    if lane_length < 8:
        raise ValueError("lane_length must be >= 8 blocks")
    total_blocks = lane_length * params.parallelism
    return lane_length, total_blocks


def _initial_hash(
    password: bytes,
    salt: bytes,
    params: Argon2Params,
    total_blocks: int,
) -> bytes:
    """Compute initial H0 digest from parameters + password + salt."""
    data = b"".join(
        [
            _le32(params.parallelism),
            _le32(params.hash_len),
            _le32(total_blocks),
            _le32(params.time_cost),
            _le32(params.version),
            _le32(ARGON2ID_TYPE_CODE),
            _le32(len(password)),
            password,
            _le32(len(salt)),
            salt,
            _le32(0),  # secret length
            _le32(0),  # associated data length
        ]
    )
    return hashlib.blake2b(data, digest_size=64).digest()


def _reference_index(
    prev_block: bytes,
    pass_index: int,
    lane_index: int,
    column_index: int,
    lane_length: int,
    parallelism: int,
) -> tuple[int, int]:
    """Pick a reference block index from previous block entropy."""
    j1 = int.from_bytes(prev_block[0:4], "little")
    j2 = int.from_bytes(prev_block[4:8], "little")

    # First pass: intra-lane references only (Argon2id-like simplification).
    if pass_index == 0:
        ref_lane = lane_index
    else:
        ref_lane = j2 % parallelism

    if ref_lane == lane_index:
        # Current block is not yet available; choose in [0, column_index-1].
        upper = max(1, column_index)
    else:
        upper = lane_length

    ref_column = j1 % upper
    return ref_lane, ref_column


def _compress_block(
    prev_block: bytes,
    ref_block: bytes,
    pass_index: int,
    lane_index: int,
    column_index: int,
) -> bytes:
    """Derive a new 1KiB block from previous and reference blocks."""
    seed = b"".join(
        [
            _xor_bytes(prev_block, ref_block),
            _le32(pass_index),
            _le32(lane_index),
            _le32(column_index),
        ]
    )
    return _expand_hash(seed, BLOCK_SIZE)


def argon2_mvp_hash_raw(password: bytes, salt: bytes, params: Argon2Params) -> bytes:
    """Compute raw digest bytes with the educational Argon2-style pipeline."""
    _validate_params(params, salt)
    lane_length, total_blocks = _memory_layout(params)

    h0 = _initial_hash(password, salt, params, total_blocks)

    lanes: list[list[bytes]] = [
        [b"\x00" * BLOCK_SIZE for _ in range(lane_length)]
        for _ in range(params.parallelism)
    ]

    for lane in range(params.parallelism):
        lanes[lane][0] = _expand_hash(h0 + _le32(0) + _le32(lane), BLOCK_SIZE)
        lanes[lane][1] = _expand_hash(h0 + _le32(1) + _le32(lane), BLOCK_SIZE)

    for pass_index in range(params.time_cost):
        for column_index in range(2, lane_length):
            for lane_index in range(params.parallelism):
                prev_block = lanes[lane_index][column_index - 1]
                ref_lane, ref_col = _reference_index(
                    prev_block,
                    pass_index,
                    lane_index,
                    column_index,
                    lane_length,
                    params.parallelism,
                )
                ref_block = lanes[ref_lane][ref_col]
                new_block = _compress_block(
                    prev_block,
                    ref_block,
                    pass_index,
                    lane_index,
                    column_index,
                )
                if pass_index > 0:
                    new_block = _xor_bytes(new_block, lanes[lane_index][column_index])
                lanes[lane_index][column_index] = new_block

    final_block = lanes[0][lane_length - 1]
    for lane in range(1, params.parallelism):
        final_block = _xor_bytes(final_block, lanes[lane][lane_length - 1])

    return _expand_hash(final_block + _le32(params.hash_len), params.hash_len)


def _b64_no_pad(data: bytes) -> str:
    """Base64 text without trailing '=' for PHC-style display."""
    return base64.b64encode(data).decode("ascii").rstrip("=")


def _b64_decode_no_pad(text: str) -> bytes:
    """Decode base64 text that may omit '=' padding."""
    pad = "=" * ((4 - len(text) % 4) % 4)
    return base64.b64decode(text + pad)


def encode_phc_string(params: Argon2Params, salt: bytes, digest: bytes) -> str:
    """Encode digest into a PHC-like Argon2id string."""
    return (
        f"${params.variant}$v={params.version}"
        f"$m={params.memory_cost_kib},t={params.time_cost},p={params.parallelism}"
        f"${_b64_no_pad(salt)}"
        f"${_b64_no_pad(digest)}"
    )


def parse_phc_string(encoded: str) -> tuple[Argon2Params, bytes, bytes]:
    """Parse PHC-like encoded string back to params/salt/digest."""
    parts = encoded.split("$")
    if len(parts) != 6 or parts[0] != "":
        raise ValueError("invalid PHC string format")

    variant = parts[1]
    if variant != "argon2id":
        raise ValueError("unsupported variant")

    if not parts[2].startswith("v="):
        raise ValueError("missing version field")
    version = int(parts[2][2:])

    kv = {}
    for pair in parts[3].split(","):
        if "=" not in pair:
            raise ValueError("invalid parameter segment")
        k, v = pair.split("=", 1)
        kv[k] = int(v)

    if not {"m", "t", "p"}.issubset(kv):
        raise ValueError("missing m/t/p parameters")

    salt = _b64_decode_no_pad(parts[4])
    digest = _b64_decode_no_pad(parts[5])

    params = Argon2Params(
        time_cost=kv["t"],
        memory_cost_kib=kv["m"],
        parallelism=kv["p"],
        hash_len=len(digest),
        version=version,
        variant=variant,
    )
    return params, salt, digest


def verify_password(password: bytes, encoded: str) -> bool:
    """Recompute digest from encoded PHC string and compare safely."""
    params, salt, expected = parse_phc_string(encoded)
    got = argon2_mvp_hash_raw(password, salt, params)
    return hmac.compare_digest(got, expected)


def _self_test_deterministic() -> None:
    """Same input tuple must produce identical digest."""
    params = Argon2Params(time_cost=2, memory_cost_kib=64, parallelism=2, hash_len=32)
    password = b"correct horse battery staple"
    salt = b"demo-salt-1234"

    d1 = argon2_mvp_hash_raw(password, salt, params)
    d2 = argon2_mvp_hash_raw(password, salt, params)
    assert d1 == d2


def _self_test_verify() -> None:
    """Correct password should pass, wrong one should fail."""
    params = Argon2Params(time_cost=2, memory_cost_kib=64, parallelism=2, hash_len=32)
    password = b"argon2-password"
    salt = b"salt-2026-demo"

    digest = argon2_mvp_hash_raw(password, salt, params)
    encoded = encode_phc_string(params, salt, digest)

    assert verify_password(password, encoded)
    assert not verify_password(b"wrong-password", encoded)


def _self_test_validation() -> None:
    """Parameter checks should reject obviously weak/invalid settings."""
    try:
        bad = Argon2Params(time_cost=1, memory_cost_kib=4, parallelism=2, hash_len=32)
        _ = argon2_mvp_hash_raw(b"pw", b"12345678", bad)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for invalid memory_cost_kib")


def main() -> None:
    _self_test_deterministic()
    _self_test_verify()
    _self_test_validation()

    params = Argon2Params(time_cost=3, memory_cost_kib=96, parallelism=3, hash_len=32)
    password = b"S3curePassword!"
    salt = b"unique-demo-salt"

    t0 = time.perf_counter()
    digest = argon2_mvp_hash_raw(password, salt, params)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    encoded = encode_phc_string(params, salt, digest)
    ok = verify_password(password, encoded)
    bad = verify_password(b"S3curePassword?", encoded)

    print("=== Argon2 Educational MVP (argon2id-style) ===")
    print(
        "params: "
        f"m={params.memory_cost_kib} KiB, t={params.time_cost}, "
        f"p={params.parallelism}, hash_len={params.hash_len}"
    )
    print(f"runtime: {elapsed_ms:.2f} ms")
    print(f"digest (hex): {digest.hex()}")
    print(f"phc (truncated): {encoded[:100]}...")
    print(f"verify(correct): {ok}")
    print(f"verify(wrong): {bad}")

    if not ok or bad:
        raise RuntimeError("verification result mismatch")


if __name__ == "__main__":
    main()

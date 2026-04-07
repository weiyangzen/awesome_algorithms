"""ECC MVP: explicit finite-field curve arithmetic + ECDH + ECDSA demo.

This script is intentionally educational and auditable:
- No black-box crypto library calls for core ECC operations.
- Implements point add/double/scalar-multiplication on secp256k1.
- Demonstrates both key agreement (ECDH) and digital signature (ECDSA).
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Dict, List, Optional, Tuple

import numpy as np

Point = Optional[Tuple[int, int]]


@dataclass(frozen=True)
class Curve:
    name: str
    p: int
    a: int
    b: int
    g: Tuple[int, int]
    n: int
    h: int


@dataclass
class OpCounter:
    additions: int = 0
    doublings: int = 0
    inversions: int = 0

    @property
    def total(self) -> int:
        return self.additions + self.doublings + self.inversions


def secp256k1() -> Curve:
    return Curve(
        name="secp256k1",
        p=0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F,
        a=0,
        b=7,
        g=(
            0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
            0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8,
        ),
        n=0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141,
        h=1,
    )


def ensure_int(name: str, value: int) -> int:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be int, got {type(value)!r}.")
    return value


def mod_inv(value: int, modulus: int) -> int:
    value %= modulus
    if value == 0:
        raise ZeroDivisionError("Cannot invert zero modulo modulus.")

    t, new_t = 0, 1
    r, new_r = modulus, value
    while new_r != 0:
        q = r // new_r
        t, new_t = new_t, t - q * new_t
        r, new_r = new_r, r - q * new_r

    if r != 1:
        raise ZeroDivisionError("Input is not invertible modulo modulus.")
    return t % modulus


def is_on_curve(curve: Curve, point: Point) -> bool:
    if point is None:
        return True
    x, y = point
    if not (0 <= x < curve.p and 0 <= y < curve.p):
        return False
    return (y * y - (x * x * x + curve.a * x + curve.b)) % curve.p == 0


def point_neg(curve: Curve, point: Point) -> Point:
    if point is None:
        return None
    x, y = point
    return (x, (-y) % curve.p)


def point_add(curve: Curve, p1: Point, p2: Point, counter: Optional[OpCounter] = None) -> Point:
    if p1 is None:
        return p2
    if p2 is None:
        return p1

    x1, y1 = p1
    x2, y2 = p2
    p = curve.p

    if x1 == x2 and (y1 + y2) % p == 0:
        return None

    if p1 == p2:
        if y1 % p == 0:
            return None
        numerator = (3 * x1 * x1 + curve.a) % p
        denominator = (2 * y1) % p
        if counter is not None:
            counter.doublings += 1
    else:
        numerator = (y2 - y1) % p
        denominator = (x2 - x1) % p
        if counter is not None:
            counter.additions += 1

    slope = (numerator * mod_inv(denominator, p)) % p
    if counter is not None:
        counter.inversions += 1

    x3 = (slope * slope - x1 - x2) % p
    y3 = (slope * (x1 - x3) - y1) % p
    return (x3, y3)


def scalar_mult(curve: Curve, k: int, point: Point, counter: Optional[OpCounter] = None) -> Point:
    k = ensure_int("scalar", k)
    if k < 0:
        return scalar_mult(curve, -k, point_neg(curve, point), counter)
    if k == 0 or point is None:
        return None

    result: Point = None
    addend = point
    scalar = k
    while scalar > 0:
        if scalar & 1:
            result = point_add(curve, result, addend, counter)
        addend = point_add(curve, addend, addend, counter)
        scalar >>= 1
    return result


def validate_private_key(curve: Curve, private_key: int) -> int:
    private_key = ensure_int("private_key", private_key)
    if not (1 <= private_key < curve.n):
        raise ValueError(f"private_key must satisfy 1 <= d < n, got {private_key}.")
    return private_key


def validate_public_key(curve: Curve, public_key: Point) -> None:
    if public_key is None:
        raise ValueError("Public key cannot be infinity point.")
    if not is_on_curve(curve, public_key):
        raise ValueError("Public key is not on the curve.")
    # secp256k1 has cofactor 1, but this subgroup check still makes intent explicit.
    if scalar_mult(curve, curve.n, public_key) is not None:
        raise ValueError("Public key failed subgroup check (n * Q != O).")


def derive_public_key(curve: Curve, private_key: int, counter: Optional[OpCounter] = None) -> Point:
    private_key = validate_private_key(curve, private_key)
    public_key = scalar_mult(curve, private_key, curve.g, counter)
    if public_key is None:
        raise RuntimeError("Derived public key at infinity unexpectedly.")
    return public_key


def ecdh_shared_point(
    curve: Curve,
    local_private_key: int,
    peer_public_key: Point,
    counter: Optional[OpCounter] = None,
) -> Point:
    local_private_key = validate_private_key(curve, local_private_key)
    validate_public_key(curve, peer_public_key)
    shared = scalar_mult(curve, local_private_key, peer_public_key, counter)
    if shared is None:
        raise RuntimeError("ECDH shared point is infinity.")
    return shared


def hash_to_int(message: bytes, n: int) -> int:
    return int.from_bytes(sha256(message).digest(), "big") % n


def deterministic_nonce(curve: Curve, private_key: int, message: bytes) -> int:
    n_bytes = (curve.n.bit_length() + 7) // 8
    seed = sha256(private_key.to_bytes(n_bytes, "big") + message).digest()
    k = int.from_bytes(seed, "big") % curve.n
    return k if k != 0 else 1


def ecdsa_sign(
    curve: Curve,
    private_key: int,
    message: bytes,
    counter: Optional[OpCounter] = None,
) -> Tuple[int, int]:
    private_key = validate_private_key(curve, private_key)
    z = hash_to_int(message, curve.n)
    k = deterministic_nonce(curve, private_key, message)

    for _ in range(32):
        point_r = scalar_mult(curve, k, curve.g, counter)
        if point_r is None:
            k = (k + 1) % curve.n or 1
            continue

        r = point_r[0] % curve.n
        if r == 0:
            k = (k + 1) % curve.n or 1
            continue

        s = (mod_inv(k, curve.n) * (z + r * private_key)) % curve.n
        if s == 0:
            k = (k + 1) % curve.n or 1
            continue

        return (r, s)

    raise RuntimeError("Failed to find valid deterministic ECDSA nonce.")


def ecdsa_verify(
    curve: Curve,
    public_key: Point,
    message: bytes,
    signature: Tuple[int, int],
    counter: Optional[OpCounter] = None,
) -> bool:
    r, s = signature
    if not (1 <= r < curve.n and 1 <= s < curve.n):
        return False

    try:
        validate_public_key(curve, public_key)
    except ValueError:
        return False

    z = hash_to_int(message, curve.n)
    w = mod_inv(s, curve.n)
    u1 = (z * w) % curve.n
    u2 = (r * w) % curve.n

    p1 = scalar_mult(curve, u1, curve.g, counter)
    p2 = scalar_mult(curve, u2, public_key, counter)
    x_point = point_add(curve, p1, p2, counter)
    if x_point is None:
        return False

    return (x_point[0] % curve.n) == r


def kdf_from_point_x(curve: Curve, point: Point) -> str:
    if point is None:
        raise ValueError("Cannot derive key from infinity point.")
    byte_len = (curve.p.bit_length() + 7) // 8
    x_bytes = point[0].to_bytes(byte_len, "big")
    return sha256(x_bytes).hexdigest()


def short_hex(value: int, keep: int = 12) -> str:
    hx = f"{value:x}"
    if len(hx) <= keep:
        return hx
    return f"{hx[:keep]}...{hx[-keep:]}"


def check_group_invariants(curve: Curve) -> None:
    if not is_on_curve(curve, curve.g):
        raise RuntimeError("Generator is not on curve.")
    if scalar_mult(curve, curve.n, curve.g) is not None:
        raise RuntimeError("Generator order check failed: nG != O.")
    if scalar_mult(curve, curve.n + 1, curve.g) != curve.g:
        raise RuntimeError("Group cycle check failed: (n+1)G != G.")
    if point_add(curve, curve.g, point_neg(curve, curve.g)) is not None:
        raise RuntimeError("Additive inverse check failed: G + (-G) != O.")


def run_case(
    curve: Curve,
    case_name: str,
    alice_private: int,
    bob_private: int,
    message: bytes,
) -> Dict[str, int]:
    print(f"\n=== {case_name} ===")

    alice_pub_ops = OpCounter()
    bob_pub_ops = OpCounter()
    alice_public = derive_public_key(curve, alice_private, alice_pub_ops)
    bob_public = derive_public_key(curve, bob_private, bob_pub_ops)

    alice_shared_ops = OpCounter()
    bob_shared_ops = OpCounter()
    shared_ab = ecdh_shared_point(curve, alice_private, bob_public, alice_shared_ops)
    shared_ba = ecdh_shared_point(curve, bob_private, alice_public, bob_shared_ops)
    if shared_ab != shared_ba:
        raise RuntimeError("ECDH mismatch between Alice and Bob.")

    sign_ops = OpCounter()
    verify_ops = OpCounter()
    signature = ecdsa_sign(curve, alice_private, message, sign_ops)
    verify_ok = ecdsa_verify(curve, alice_public, message, signature, verify_ops)
    if not verify_ok:
        raise RuntimeError("ECDSA verification failed on original message.")

    tampered_ok = ecdsa_verify(curve, alice_public, message + b"!", signature)
    if tampered_ok:
        raise RuntimeError("ECDSA accepted tampered message unexpectedly.")

    malformed_pub = (alice_public[0], (alice_public[1] + 1) % curve.p)
    malformed_rejected = False
    try:
        validate_public_key(curve, malformed_pub)
    except ValueError:
        malformed_rejected = True
    if not malformed_rejected:
        raise RuntimeError("Malformed public key was not rejected.")

    session_key = kdf_from_point_x(curve, shared_ab)
    print(f"Alice pub x  : 0x{short_hex(alice_public[0])}")
    print(f"Bob   pub x  : 0x{short_hex(bob_public[0])}")
    print(f"Shared x     : 0x{short_hex(shared_ab[0])}")
    print(f"ECDSA r,s    : 0x{short_hex(signature[0], 10)}, 0x{short_hex(signature[1], 10)}")
    print(f"Session key  : {session_key}")
    print(
        "Ops | keygen(A,B) "
        f"{alice_pub_ops.total}, {bob_pub_ops.total} | "
        f"ecdh(A,B) {alice_shared_ops.total}, {bob_shared_ops.total} | "
        f"ecdsa(sign,verify) {sign_ops.total}, {verify_ops.total}"
    )

    return {
        "alice_pub_ops": alice_pub_ops.total,
        "bob_pub_ops": bob_pub_ops.total,
        "alice_shared_ops": alice_shared_ops.total,
        "bob_shared_ops": bob_shared_ops.total,
        "sign_ops": sign_ops.total,
        "verify_ops": verify_ops.total,
        "shared_bits": shared_ab[0].bit_length(),
        "session_prefix": int(session_key[:16], 16),
        "sig_prefix": signature[0] & ((1 << 64) - 1),
    }


def main() -> None:
    curve = secp256k1()
    print(f"Curve: {curve.name}")
    check_group_invariants(curve)
    print("Group invariants: PASS")

    cases = [
        (
            "Case-1 deterministic keys",
            int("1f1e1d1c1b1a19181716151413121110100f0e0d0c0b0a090807060504030201", 16),
            int("2a2b2c2d2e2f30313233343536373839404142434445464748494a4b4c4d4e4f", 16),
            b"ECC MVP demo message 1",
        ),
        (
            "Case-2 deterministic keys",
            int("a0a1a2a3a4a5a6a7a8a9aaabacadaeaf0102030405060708090a0b0c0d0e0f10", 16),
            int("0f0e0d0c0b0a090807060504030201112233445566778899aabbccddeeff0011", 16),
            b"ECC MVP demo message 2",
        ),
    ]

    stats: List[Dict[str, int]] = []
    for case_name, alice_d, bob_d, message in cases:
        stats.append(run_case(curve, case_name, alice_d, bob_d, message))

    metrics = np.array(
        [
            [
                item["alice_pub_ops"],
                item["bob_pub_ops"],
                item["alice_shared_ops"],
                item["bob_shared_ops"],
                item["sign_ops"],
                item["verify_ops"],
                item["shared_bits"],
            ]
            for item in stats
        ],
        dtype=np.int64,
    )

    mean_ops = np.mean(metrics[:, :6], axis=0)
    min_shared_bits = int(np.min(metrics[:, 6]))
    max_shared_bits = int(np.max(metrics[:, 6]))
    prefixes = np.array(
        [item["session_prefix"] ^ item["sig_prefix"] for item in stats],
        dtype=np.uint64,
    )
    fingerprint = int(np.bitwise_xor.reduce(prefixes))

    print("\n=== Summary ===")
    print(
        "Mean ops [keygenA,keygenB,ecdhA,ecdhB,sign,verify] = "
        f"{mean_ops[0]:.1f}, {mean_ops[1]:.1f}, {mean_ops[2]:.1f}, "
        f"{mean_ops[3]:.1f}, {mean_ops[4]:.1f}, {mean_ops[5]:.1f}"
    )
    print(f"Shared X bit-length range: [{min_shared_bits}, {max_shared_bits}]")
    print(f"Deterministic fingerprint: 0x{fingerprint:016x}")

    if min_shared_bits < 240:
        raise RuntimeError(f"Unexpectedly small shared X bit-length: {min_shared_bits}")
    if not np.all(metrics[:, :6] > 0):
        raise RuntimeError("Operation counters contain non-positive values.")

    print("All ECC checks passed.")


if __name__ == "__main__":
    main()

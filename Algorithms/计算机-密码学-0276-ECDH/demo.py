"""ECDH MVP on short-Weierstrass curves with explicit source-level steps."""

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


def mod_inv(value: int, p: int) -> int:
    value = value % p
    if value == 0:
        raise ZeroDivisionError("Cannot invert 0 in finite field.")

    t, new_t = 0, 1
    r, new_r = p, value
    while new_r != 0:
        q = r // new_r
        t, new_t = new_t, t - q * new_t
        r, new_r = new_r, r - q * new_r

    if r != 1:
        raise ZeroDivisionError("Input is not invertible modulo p.")
    return t % p


def is_on_curve(curve: Curve, point: Point) -> bool:
    if point is None:
        return True
    x, y = point
    if not (0 <= x < curve.p and 0 <= y < curve.p):
        return False
    left = (y * y) % curve.p
    right = (x * x * x + curve.a * x + curve.b) % curve.p
    return left == right


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

    inv_denominator = mod_inv(denominator, p)
    if counter is not None:
        counter.inversions += 1

    slope = (numerator * inv_denominator) % p
    x3 = (slope * slope - x1 - x2) % p
    y3 = (slope * (x1 - x3) - y1) % p
    return (x3, y3)


def scalar_mult(curve: Curve, k: int, point: Point, counter: Optional[OpCounter] = None) -> Point:
    k = ensure_int("k", k)
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


def validate_private_key(curve: Curve, d: int) -> int:
    d = ensure_int("private key", d)
    if not (1 <= d < curve.n):
        raise ValueError(f"Private key must satisfy 1 <= d < n, got d={d}.")
    return d


def validate_public_key(curve: Curve, point: Point) -> None:
    if point is None:
        raise ValueError("Public key cannot be point-at-infinity.")
    if not is_on_curve(curve, point):
        raise ValueError("Public key is not on the curve.")
    # Subgroup membership check: n * Q == O.
    if scalar_mult(curve, curve.n, point) is not None:
        raise ValueError("Public key failed subgroup check (n*Q != O).")


def derive_public_key(curve: Curve, private_key: int, counter: Optional[OpCounter] = None) -> Point:
    private_key = validate_private_key(curve, private_key)
    public_key = scalar_mult(curve, private_key, curve.g, counter)
    if public_key is None:
        raise ValueError("Derived invalid public key at infinity.")
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
        raise ValueError("Shared point is infinity; reject key agreement.")
    return shared


def kdf_from_x_sha256(curve: Curve, shared_point: Point) -> str:
    if shared_point is None:
        raise ValueError("Cannot derive key material from infinity point.")
    x_coord = shared_point[0]
    byte_len = (curve.p.bit_length() + 7) // 8
    x_bytes = x_coord.to_bytes(byte_len, "big")
    return sha256(x_bytes).hexdigest()


def short_hex(value: int, keep: int = 12) -> str:
    hx = f"{value:x}"
    if len(hx) <= keep:
        return hx
    return f"{hx[:keep]}...{hx[-keep:]}"


def run_case(curve: Curve, name: str, alice_d: int, bob_d: int) -> Dict[str, int]:
    print(f"\n=== {name} ===")
    alice_pub_ops = OpCounter()
    bob_pub_ops = OpCounter()

    alice_q = derive_public_key(curve, alice_d, alice_pub_ops)
    bob_q = derive_public_key(curve, bob_d, bob_pub_ops)

    alice_shared_ops = OpCounter()
    bob_shared_ops = OpCounter()
    shared_ab = ecdh_shared_point(curve, alice_d, bob_q, alice_shared_ops)
    shared_ba = ecdh_shared_point(curve, bob_d, alice_q, bob_shared_ops)

    if shared_ab != shared_ba:
        raise RuntimeError("ECDH shared points mismatch.")

    digest = kdf_from_x_sha256(curve, shared_ab)

    bad_peer = (bob_q[0], (bob_q[1] + 1) % curve.p)
    rejected = False
    try:
        _ = ecdh_shared_point(curve, alice_d, bad_peer)
    except ValueError:
        rejected = True
    if not rejected:
        raise RuntimeError("Malformed public key was not rejected.")

    print(f"Alice pubkey x: 0x{short_hex(alice_q[0])}")
    print(f"Bob   pubkey x: 0x{short_hex(bob_q[0])}")
    print(f"Shared x      : 0x{short_hex(shared_ab[0])}")
    print(f"KDF(SHA-256)  : {digest}")
    print(
        "Ops | pub(A,B) "
        f"{alice_pub_ops.total}, {bob_pub_ops.total} | "
        f"shared(A,B) {alice_shared_ops.total}, {bob_shared_ops.total}"
    )

    return {
        "alice_pub_ops": alice_pub_ops.total,
        "bob_pub_ops": bob_pub_ops.total,
        "alice_shared_ops": alice_shared_ops.total,
        "bob_shared_ops": bob_shared_ops.total,
        "shared_x_bits": shared_ab[0].bit_length(),
        "digest_prefix": int(digest[:16], 16),
    }


def check_group_invariants(curve: Curve) -> None:
    if not is_on_curve(curve, curve.g):
        raise RuntimeError("Generator is not on curve.")
    if scalar_mult(curve, curve.n, curve.g) is not None:
        raise RuntimeError("Generator order check failed: n*G != O.")
    if scalar_mult(curve, curve.n + 1, curve.g) != curve.g:
        raise RuntimeError("Group cycle check failed: (n+1)*G != G.")
    if point_add(curve, curve.g, point_neg(curve, curve.g)) is not None:
        raise RuntimeError("Additive inverse check failed: G + (-G) != O.")


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
        ),
        (
            "Case-2 deterministic keys",
            int("a0a1a2a3a4a5a6a7a8a9aaabacadaeaf0102030405060708090a0b0c0d0e0f10", 16),
            int("0f0e0d0c0b0a090807060504030201112233445566778899aabbccddeeff0011", 16),
        ),
    ]

    stats: List[Dict[str, int]] = []
    for case_name, alice_d, bob_d in cases:
        stats.append(run_case(curve, case_name, alice_d, bob_d))

    metrics = np.array(
        [
            [
                item["alice_pub_ops"],
                item["bob_pub_ops"],
                item["alice_shared_ops"],
                item["bob_shared_ops"],
                item["shared_x_bits"],
            ]
            for item in stats
        ],
        dtype=np.int64,
    )

    mean_ops = np.mean(metrics[:, :4], axis=0)
    min_shared_bits = int(np.min(metrics[:, 4]))
    max_shared_bits = int(np.max(metrics[:, 4]))
    digest_prefixes = np.array([item["digest_prefix"] for item in stats], dtype=np.uint64)
    digest_fingerprint = int(np.bitwise_xor.reduce(digest_prefixes))

    print("\n=== Summary ===")
    print(
        "Mean ops [pub_A, pub_B, shared_A, shared_B] = "
        f"{mean_ops[0]:.1f}, {mean_ops[1]:.1f}, {mean_ops[2]:.1f}, {mean_ops[3]:.1f}"
    )
    print(f"Shared X bit-length range: [{min_shared_bits}, {max_shared_bits}]")
    print(f"Deterministic fingerprint: 0x{digest_fingerprint:016x}")

    if min_shared_bits < 240:
        raise RuntimeError(f"Shared X too small in bit length: {min_shared_bits}")
    if not np.all(metrics[:, :4] > 0):
        raise RuntimeError("Operation counters contain zero values unexpectedly.")

    print("All ECDH checks passed.")


if __name__ == "__main__":
    main()

"""Educational ECDSA MVP over secp256k1.

This script implements core ECDSA operations from scratch:
- finite-field point arithmetic
- key generation
- signature generation
- signature verification

It is intentionally minimal and non-constant-time, for algorithm learning only.
"""

from __future__ import annotations

import hashlib
from typing import Optional, Tuple

# secp256k1 domain parameters
P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
A = 0
B = 7
GX = 55066263022277343669578718895168534326250603453777594175500187360389116729240
GY = 32670510020758816978083085130507043184471273380659243275938904335757337482424
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141

Point = Optional[Tuple[int, int]]
G: Point = (GX, GY)


def inv_mod(k: int, p: int) -> int:
    """Return modular inverse k^{-1} mod p."""
    if k % p == 0:
        raise ZeroDivisionError("inverse does not exist")
    return pow(k, -1, p)


def is_on_curve(point: Point) -> bool:
    """Check whether a point is on secp256k1."""
    if point is None:
        return True
    x, y = point
    return (y * y - (x * x * x + A * x + B)) % P == 0


def point_add(p1: Point, p2: Point) -> Point:
    """Elliptic-curve group addition."""
    if p1 is None:
        return p2
    if p2 is None:
        return p1

    x1, y1 = p1
    x2, y2 = p2

    if x1 == x2 and (y1 + y2) % P == 0:
        return None

    if p1 == p2:
        # tangent slope for doubling
        m = (3 * x1 * x1 + A) * inv_mod(2 * y1, P) % P
    else:
        # secant slope for addition
        m = (y2 - y1) * inv_mod(x2 - x1, P) % P

    x3 = (m * m - x1 - x2) % P
    y3 = (m * (x1 - x3) - y1) % P
    return (x3, y3)


def scalar_mult(k: int, point: Point) -> Point:
    """Double-and-add scalar multiplication k * point."""
    if k % N == 0 or point is None:
        return None

    result: Point = None
    addend: Point = point
    n = k

    while n > 0:
        if n & 1:
            result = point_add(result, addend)
        addend = point_add(addend, addend)
        n >>= 1

    return result


def hash_to_int(message: bytes) -> int:
    """Hash message to integer (SHA-256)."""
    digest = hashlib.sha256(message).digest()
    return int.from_bytes(digest, "big")


def deterministic_nonce(private_key: int, message: bytes, counter: int) -> int:
    """Small deterministic nonce generator for reproducible demo output."""
    seed = (
        private_key.to_bytes(32, "big")
        + message
        + counter.to_bytes(4, "big", signed=False)
    )
    k = int.from_bytes(hashlib.sha256(seed).digest(), "big") % N
    return k if k != 0 else 1


def make_keypair() -> tuple[int, Point]:
    """Create a stable private/public keypair for repeatable demonstrations."""
    private_seed = hashlib.sha256(b"CS-0135-ECDSA-demo").digest()
    private_key = int.from_bytes(private_seed, "big") % N
    if private_key == 0:
        private_key = 1

    public_key = scalar_mult(private_key, G)
    if public_key is None:
        raise RuntimeError("failed to create public key")
    return private_key, public_key


def sign(message: bytes, private_key: int) -> tuple[int, int]:
    """Generate an ECDSA signature (r, s)."""
    z = hash_to_int(message)
    counter = 0

    while True:
        k = deterministic_nonce(private_key, message, counter)
        counter += 1

        point = scalar_mult(k, G)
        if point is None:
            continue

        r = point[0] % N
        if r == 0:
            continue

        s = (inv_mod(k, N) * (z + r * private_key)) % N
        if s == 0:
            continue

        # Optional canonical low-s form.
        if s > N // 2:
            s = N - s

        return (r, s)


def verify(message: bytes, signature: tuple[int, int], public_key: Point) -> bool:
    """Verify an ECDSA signature."""
    if public_key is None or not is_on_curve(public_key):
        return False

    r, s = signature
    if not (1 <= r < N and 1 <= s < N):
        return False

    z = hash_to_int(message)
    w = inv_mod(s, N)
    u1 = (z * w) % N
    u2 = (r * w) % N

    p1 = scalar_mult(u1, G)
    p2 = scalar_mult(u2, public_key)
    point = point_add(p1, p2)

    if point is None:
        return False

    x, _ = point
    return (x % N) == r


def main() -> None:
    private_key, public_key = make_keypair()
    assert is_on_curve(G)
    assert is_on_curve(public_key)

    message = b"ECDSA minimal runnable MVP for CS-0135"
    signature = sign(message, private_key)

    ok_original = verify(message, signature, public_key)
    ok_tampered_message = verify(message + b"!", signature, public_key)

    r, s = signature
    tampered_signature = ((r + 1) % N, s)
    if tampered_signature[0] == 0:
        tampered_signature = (1, s)
    ok_tampered_signature = verify(message, tampered_signature, public_key)

    print("=== ECDSA MVP (secp256k1) ===")
    print(f"private_key (hex, truncated): {hex(private_key)[:18]}...")
    print(f"public_key x (hex, truncated): {hex(public_key[0])[:18]}...")
    print(f"public_key y (hex, truncated): {hex(public_key[1])[:18]}...")
    print(f"message: {message.decode('utf-8')}")
    print(f"signature.r (hex, truncated): {hex(r)[:18]}...")
    print(f"signature.s (hex, truncated): {hex(s)[:18]}...")
    print(f"verify(original): {ok_original}")
    print(f"verify(tampered message): {ok_tampered_message}")
    print(f"verify(tampered signature): {ok_tampered_signature}")

    if not (ok_original and not ok_tampered_message and not ok_tampered_signature):
        raise RuntimeError("ECDSA demo checks failed")


if __name__ == "__main__":
    main()

"""Paillier additive homomorphic encryption MVP.

This demo intentionally keeps the implementation small and explicit:
- Key generation
- Encryption / decryption
- Homomorphic ciphertext addition
- Homomorphic plaintext-scalar multiplication

Run:
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from secrets import randbelow


@dataclass(frozen=True)
class PublicKey:
    n: int
    g: int
    n_sq: int


@dataclass(frozen=True)
class PrivateKey:
    lam: int
    mu: int


@dataclass(frozen=True)
class KeyPair:
    public: PublicKey
    private: PrivateKey


def lcm(a: int, b: int) -> int:
    return a // gcd(a, b) * b


def egcd(a: int, b: int) -> tuple[int, int, int]:
    if b == 0:
        return a, 1, 0
    g, x1, y1 = egcd(b, a % b)
    return g, y1, x1 - (a // b) * y1


def modinv(a: int, m: int) -> int:
    g, x, _ = egcd(a, m)
    if g != 1:
        raise ValueError("modular inverse does not exist")
    return x % m


def is_probable_prime(n: int) -> bool:
    if n < 2:
        return False
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29)
    for p in small_primes:
        if n == p:
            return True
        if n % p == 0:
            return False

    d = n - 1
    s = 0
    while d % 2 == 0:
        s += 1
        d //= 2

    # For this MVP scale (<=64-bit primes), fixed bases are sufficient in practice.
    for a in (2, 3, 5, 7, 11, 13, 17):
        if a >= n:
            continue
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        witness_composite = True
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                witness_composite = False
                break
        if witness_composite:
            return False
    return True


def generate_prime(bits: int) -> int:
    if bits < 8:
        raise ValueError("bits must be >= 8")

    while True:
        candidate = randbelow(1 << bits)
        candidate |= 1
        candidate |= 1 << (bits - 1)
        if is_probable_prime(candidate):
            return candidate


def _l_function(u: int, n: int) -> int:
    return (u - 1) // n


def generate_keypair(bits: int = 48) -> KeyPair:
    if bits < 16:
        raise ValueError("bits must be >= 16")

    half = bits // 2
    while True:
        p = generate_prime(half)
        q = generate_prime(half)
        if p == q:
            continue

        n = p * q
        n_sq = n * n
        g = n + 1
        lam = lcm(p - 1, q - 1)

        l_value = _l_function(pow(g, lam, n_sq), n)
        if gcd(l_value, n) != 1:
            continue
        mu = modinv(l_value, n)
        return KeyPair(
            public=PublicKey(n=n, g=g, n_sq=n_sq),
            private=PrivateKey(lam=lam, mu=mu),
        )


def encrypt(pk: PublicKey, m: int) -> int:
    if not (0 <= m < pk.n):
        raise ValueError(f"plaintext out of range [0, n): m={m}, n={pk.n}")

    while True:
        r = randbelow(pk.n)
        if r == 0:
            continue
        if gcd(r, pk.n) == 1:
            break

    # c = g^m * r^n (mod n^2)
    return (pow(pk.g, m, pk.n_sq) * pow(r, pk.n, pk.n_sq)) % pk.n_sq


def decrypt(pk: PublicKey, sk: PrivateKey, c: int) -> int:
    if not (0 <= c < pk.n_sq):
        raise ValueError("ciphertext out of range [0, n^2)")

    u = pow(c, sk.lam, pk.n_sq)
    l_value = _l_function(u, pk.n)
    return (l_value * sk.mu) % pk.n


def e_add(pk: PublicKey, c1: int, c2: int) -> int:
    """Homomorphic addition of two ciphertexts: Enc(m1+m2)."""
    return (c1 * c2) % pk.n_sq


def e_mul_plain(pk: PublicKey, c: int, k: int) -> int:
    """Homomorphic multiply-by-plaintext constant: Enc(k*m)."""
    if k < 0:
        raise ValueError("k must be non-negative in this MVP")
    return pow(c, k, pk.n_sq)


def main() -> None:
    print("== Paillier Homomorphic Encryption MVP ==")

    keypair = generate_keypair(bits=48)
    pk = keypair.public
    sk = keypair.private

    print(f"Key size (n bits): {pk.n.bit_length()}")

    values = [7, 11, 23]
    ciphertexts = [encrypt(pk, m) for m in values]

    c_sum = ciphertexts[0]
    for c in ciphertexts[1:]:
        c_sum = e_add(pk, c_sum, c)
    decrypted_sum = decrypt(pk, sk, c_sum)

    expected_sum = sum(values) % pk.n
    print(f"Plain values: {values}")
    print(f"Decrypted homomorphic sum: {decrypted_sum}")
    print(f"Expected sum mod n:       {expected_sum}")
    assert decrypted_sum == expected_sum

    # Weighted linear combination: 3*v0 + 2*v1 + 1*v2
    c_weighted = e_add(
        pk,
        e_add(
            pk,
            e_mul_plain(pk, ciphertexts[0], 3),
            e_mul_plain(pk, ciphertexts[1], 2),
        ),
        ciphertexts[2],
    )
    decrypted_weighted = decrypt(pk, sk, c_weighted)
    expected_weighted = (3 * values[0] + 2 * values[1] + values[2]) % pk.n

    print(f"Decrypted weighted result: {decrypted_weighted}")
    print(f"Expected weighted mod n:   {expected_weighted}")
    assert decrypted_weighted == expected_weighted

    print("MVP run completed successfully.")


if __name__ == "__main__":
    main()

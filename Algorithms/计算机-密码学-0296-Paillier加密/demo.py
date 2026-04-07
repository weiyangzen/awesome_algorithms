"""Paillier encryption MVP (non-interactive, source-level implementation)."""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd, lcm
import secrets
from typing import List


@dataclass(frozen=True)
class PublicKey:
    n: int
    n_sq: int
    g: int


@dataclass(frozen=True)
class PrivateKey:
    lambda_param: int
    mu: int
    p: int
    q: int


def l_function(x: int, n: int) -> int:
    """Paillier L function: L(x) = (x - 1) // n, requiring x == 1 (mod n)."""
    if (x - 1) % n != 0:
        raise ValueError("invalid L-function input: x is not congruent to 1 mod n")
    return (x - 1) // n


def is_probable_prime(n: int) -> bool:
    """Deterministic Miller-Rabin for 64-bit integers."""
    if n < 2:
        return False

    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    for p in small_primes:
        if n == p:
            return True
        if n % p == 0:
            return False

    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1

    # Deterministic bases for testing n < 2^64.
    bases = [2, 325, 9375, 28178, 450775, 9780504, 1795265022]
    for a in bases:
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def generate_prime(bits: int) -> int:
    if bits < 8:
        raise ValueError("bits must be >= 8 for this demo")

    while True:
        candidate = secrets.randbits(bits)
        candidate |= 1  # odd
        candidate |= 1 << (bits - 1)  # enforce bit length
        if is_probable_prime(candidate):
            return candidate


def sample_coprime(n: int) -> int:
    while True:
        r = secrets.randbelow(n - 1) + 1
        if gcd(r, n) == 1:
            return r


def generate_keypair(bits: int = 32) -> tuple[PublicKey, PrivateKey]:
    p = generate_prime(bits)
    q = generate_prime(bits)
    while q == p:
        q = generate_prime(bits)

    n = p * q
    n_sq = n * n
    lambda_param = lcm(p - 1, q - 1)

    # Standard convenient choice for Paillier.
    g = n + 1
    u = pow(g, lambda_param, n_sq)
    l_u = l_function(u, n)
    if gcd(l_u, n) != 1:
        raise ValueError("invalid parameters: gcd(L(g^lambda mod n^2), n) != 1")
    mu = pow(l_u, -1, n)

    public_key = PublicKey(n=n, n_sq=n_sq, g=g)
    private_key = PrivateKey(lambda_param=lambda_param, mu=mu, p=p, q=q)
    return public_key, private_key


def encrypt(m: int, public_key: PublicKey) -> int:
    if not (0 <= m < public_key.n):
        raise ValueError(f"plaintext out of range: m={m}, expected 0<=m<n")

    r = sample_coprime(public_key.n)
    c1 = pow(public_key.g, m, public_key.n_sq)
    c2 = pow(r, public_key.n, public_key.n_sq)
    return (c1 * c2) % public_key.n_sq


def decrypt(c: int, public_key: PublicKey, private_key: PrivateKey) -> int:
    if not (0 <= c < public_key.n_sq):
        raise ValueError("ciphertext out of range")

    x = pow(c, private_key.lambda_param, public_key.n_sq)
    lx = l_function(x, public_key.n)
    return (lx * private_key.mu) % public_key.n


def add_ciphertexts(c1: int, c2: int, public_key: PublicKey) -> int:
    """Homomorphic addition in plaintext domain via ciphertext multiplication."""
    return (c1 * c2) % public_key.n_sq


def mul_ciphertext_by_constant(c: int, k: int, public_key: PublicKey) -> int:
    """Homomorphic scalar multiplication in plaintext domain via ciphertext exponentiation."""
    if k < 0:
        raise ValueError("k must be non-negative in this minimal demo")
    return pow(c, k, public_key.n_sq)


def aggregate_encrypted_values(ciphertexts: List[int], public_key: PublicKey) -> int:
    acc = 1
    for c in ciphertexts:
        acc = (acc * c) % public_key.n_sq
    return acc


def main() -> None:
    public_key, private_key = generate_keypair(bits=32)

    m1 = 12345
    m2 = 6789
    if max(m1, m2) >= public_key.n:
        raise RuntimeError("demo plaintext unexpectedly exceeds modulus n")

    c1 = encrypt(m1, public_key)
    c2 = encrypt(m2, public_key)

    recovered_m1 = decrypt(c1, public_key, private_key)
    recovered_m2 = decrypt(c2, public_key, private_key)

    # Probabilistic encryption check.
    c1_again = encrypt(m1, public_key)

    # Homomorphic addition.
    c_sum = add_ciphertexts(c1, c2, public_key)
    recovered_sum = decrypt(c_sum, public_key, private_key)
    expected_sum = (m1 + m2) % public_key.n

    # Homomorphic scalar multiplication.
    k = 7
    c_scaled = mul_ciphertext_by_constant(c1, k, public_key)
    recovered_scaled = decrypt(c_scaled, public_key, private_key)
    expected_scaled = (k * m1) % public_key.n

    # Privacy-preserving tally demo.
    votes = [1, 0, 1, 1, 0, 1, 1, 0]
    encrypted_votes = [encrypt(v, public_key) for v in votes]
    encrypted_tally = aggregate_encrypted_values(encrypted_votes, public_key)
    recovered_tally = decrypt(encrypted_tally, public_key, private_key)
    expected_tally = sum(votes)

    assert recovered_m1 == m1
    assert recovered_m2 == m2
    assert c1 != c1_again, "same plaintext should encrypt to different ciphertexts"
    assert recovered_sum == expected_sum
    assert recovered_scaled == expected_scaled
    assert recovered_tally == expected_tally

    print("=== Paillier Encryption MVP ===")
    print(f"Key size (n bits): {public_key.n.bit_length()}")
    print(f"m1={m1}, m2={m2}")
    print(f"decrypt(enc(m1))={recovered_m1}, decrypt(enc(m2))={recovered_m2}")
    print(f"homomorphic add -> {recovered_sum} (expected {expected_sum})")
    print(f"homomorphic scalar (k={k}) -> {recovered_scaled} (expected {expected_scaled})")
    print(f"encrypted votes={votes}, tally={recovered_tally}")
    print(f"cipher sample: c1={c1}, c2={c2}")
    print("Status: PASS")


if __name__ == "__main__":
    main()

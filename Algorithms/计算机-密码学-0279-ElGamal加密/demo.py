"""ElGamal encryption MVP (byte-oriented, non-interactive demo)."""

from __future__ import annotations

from dataclasses import dataclass
import secrets
from typing import List, Tuple


CipherPair = Tuple[int, int]


@dataclass(frozen=True)
class PublicKey:
    p: int
    g: int
    h: int


@dataclass(frozen=True)
class PrivateKey:
    p: int
    x: int


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def prime_factors(n: int) -> List[int]:
    factors: List[int] = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.append(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


def is_generator_mod_p(g: int, p: int) -> bool:
    if not (1 < g < p):
        return False
    phi = p - 1
    for q in prime_factors(phi):
        if pow(g, phi // q, p) == 1:
            return False
    return True


def generate_keypair(p: int, g: int) -> Tuple[PublicKey, PrivateKey]:
    if not is_prime(p):
        raise ValueError(f"p must be prime, got {p}")
    if not is_generator_mod_p(g, p):
        raise ValueError(f"g must be a generator modulo p, got g={g}, p={p}")
    x = secrets.randbelow(p - 2) + 1
    h = pow(g, x, p)
    return PublicKey(p=p, g=g, h=h), PrivateKey(p=p, x=x)


def encrypt_byte(m: int, public_key: PublicKey) -> CipherPair:
    p, g, h = public_key.p, public_key.g, public_key.h
    if not (0 <= m < p):
        raise ValueError(f"message byte out of range: m={m}, p={p}")
    k = secrets.randbelow(p - 2) + 1
    c1 = pow(g, k, p)
    s = pow(h, k, p)
    c2 = (m * s) % p
    return c1, c2


def decrypt_byte(cipher: CipherPair, private_key: PrivateKey) -> int:
    c1, c2 = cipher
    p, x = private_key.p, private_key.x
    s = pow(c1, x, p)
    s_inv = pow(s, p - 2, p)
    m = (c2 * s_inv) % p
    return m


def encrypt_bytes(data: bytes, public_key: PublicKey) -> List[CipherPair]:
    return [encrypt_byte(b, public_key) for b in data]


def decrypt_bytes(ciphertext: List[CipherPair], private_key: PrivateKey) -> bytes:
    plain_ints = [decrypt_byte(c, private_key) for c in ciphertext]
    return bytes(plain_ints)


def main() -> None:
    # Small teaching parameters (NOT production secure).
    p = 467
    g = 2

    public_key, private_key = generate_keypair(p=p, g=g)
    message = b"ElGamal MVP demo"

    ciphertext = encrypt_bytes(message, public_key)
    recovered = decrypt_bytes(ciphertext, private_key)

    # Probabilistic encryption check: same plaintext usually yields different ciphertext.
    c1 = encrypt_bytes(message, public_key)
    c2 = encrypt_bytes(message, public_key)

    assert recovered == message, "decryption failed to recover original message"
    assert c1 != c2, "ElGamal should be probabilistic with fresh random k"

    print("=== ElGamal Encryption MVP ===")
    print(f"Public key: p={public_key.p}, g={public_key.g}, h={public_key.h}")
    print(f"Original:  {message!r}")
    print(f"Recovered: {recovered!r}")
    print(f"Cipher sample (first 5 pairs): {ciphertext[:5]}")
    print("Status: PASS")


if __name__ == "__main__":
    main()

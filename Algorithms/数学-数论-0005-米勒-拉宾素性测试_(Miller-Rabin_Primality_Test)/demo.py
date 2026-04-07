"""Miller-Rabin primality test minimal runnable MVP."""

from __future__ import annotations

import random
from typing import Iterable, Sequence


U64_DETERMINISTIC_BASES: Sequence[int] = (
    2,
    325,
    9375,
    28178,
    450775,
    9780504,
    1795265022,
)


def mod_pow(base: int, exponent: int, modulus: int) -> int:
    """Compute (base ** exponent) % modulus via binary exponentiation."""
    result = 1
    base %= modulus
    while exponent > 0:
        if exponent & 1:
            result = (result * base) % modulus
        base = (base * base) % modulus
        exponent >>= 1
    return result


def decompose_n_minus_one(n: int) -> tuple[int, int]:
    """Return (s, d) such that n - 1 = 2**s * d and d is odd."""
    d = n - 1
    s = 0
    while d % 2 == 0:
        s += 1
        d //= 2
    return s, d


def is_strong_probable_prime_to_base(n: int, base: int, s: int, d: int) -> bool:
    """Single Miller-Rabin witness test for one base."""
    x = mod_pow(base, d, n)
    if x in (1, n - 1):
        return True

    for _ in range(s - 1):
        x = (x * x) % n
        if x == n - 1:
            return True
    return False


def _candidate_bases(n: int, rounds: int, rng: random.Random) -> Iterable[int]:
    if n < (1 << 64):
        # Deterministic for 64-bit integers.
        for base in U64_DETERMINISTIC_BASES:
            yield base
    else:
        for _ in range(rounds):
            yield rng.randrange(2, n - 1)


def is_probable_prime(n: int, rounds: int = 12, seed: int = 0) -> bool:
    """Miller-Rabin primality check.

    - Deterministic for n < 2^64 via fixed bases.
    - Probabilistic for larger n with error <= (1/4)^rounds.
    """
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False

    # Fast small-prime rejection to reduce expensive modular exponentiation.
    small_primes = (3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for p in small_primes:
        if n == p:
            return True
        if n % p == 0:
            return False

    s, d = decompose_n_minus_one(n)
    rng = random.Random(seed)

    for base in _candidate_bases(n, rounds, rng):
        a = base % n
        if a in (0, 1):
            continue
        if not is_strong_probable_prime_to_base(n, a, s, d):
            return False

    return True


def fermat_test_base_2(n: int) -> bool:
    """Very weak primality indicator used only for contrast in the demo."""
    if n <= 2 or n % 2 == 0:
        return n == 2
    return mod_pow(2, n - 1, n) == 1


def main() -> None:
    print("=== Miller-Rabin MVP Demo ===")

    known_cases = [
        (2, True),
        (3, True),
        (4, False),
        (5, True),
        (17, True),
        (19, True),
        (21, False),
        (97, True),
        (221, False),
        (341, False),
        (561, False),
        (1105, False),
        (2147483647, True),
        (2305843009213693951, True),  # 2^61 - 1
        (2305843009213693953, False),
    ]

    print("\n[1] Known cases")
    print("n\texpected\tmiller_rabin\tpass")
    for n, expected in known_cases:
        got = is_probable_prime(n)
        print(f"{n}\t{expected}\t{got}\t{got == expected}")

    print("\n[2] Carmichael contrast: Fermat(base=2) vs Miller-Rabin")
    for n in (341, 561, 1105, 1729):
        fermat = fermat_test_base_2(n)
        mr = is_probable_prime(n)
        print(f"n={n}: fermat_base2={fermat}, miller_rabin={mr}")

    print("\n[3] Random 128-bit odd candidates (probabilistic mode)")
    rng = random.Random(2026)
    for idx in range(1, 6):
        candidate = rng.getrandbits(128) | 1
        result = is_probable_prime(candidate, rounds=12, seed=idx)
        print(f"sample#{idx}: n={candidate}, probable_prime={result}")


if __name__ == "__main__":
    main()

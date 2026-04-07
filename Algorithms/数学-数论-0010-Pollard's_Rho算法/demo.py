"""Pollard's Rho MVP for integer factorization.

Run:
    python3 demo.py
"""

from __future__ import annotations

import math
import random
import time
from typing import Dict, List


def is_probable_prime(n: int) -> bool:
    """Deterministic Miller-Rabin for 64-bit integers."""
    if n < 2:
        return False
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for p in small_primes:
        if n % p == 0:
            return n == p

    d = n - 1
    s = 0
    while d % 2 == 0:
        s += 1
        d //= 2

    # Deterministic bases for unsigned 64-bit range.
    bases = (2, 325, 9375, 28178, 450775, 9780504, 1795265022)
    for a in bases:
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        witness_found = True
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                witness_found = False
                break
        if witness_found:
            return False
    return True


def pollards_rho(n: int, rng: random.Random, max_inner_steps: int = 200_000) -> int:
    """Return a non-trivial factor of n using Pollard's Rho with Floyd cycle finding."""
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3

    while True:
        x = rng.randrange(2, n - 1)
        y = x
        c = rng.randrange(1, n - 1)
        d = 1

        def f(v: int) -> int:
            return (v * v + c) % n

        steps = 0
        while d == 1 and steps < max_inner_steps:
            x = f(x)
            y = f(f(y))
            d = math.gcd(abs(x - y), n)
            steps += 1

        if 1 < d < n:
            return d
        # d == n or timed out: restart with a new random polynomial/seed.


def factorize(n: int, rng: random.Random, factors: Dict[int, int]) -> None:
    """Fill factors dict with prime exponents of n."""
    if n == 1:
        return
    if is_probable_prime(n):
        factors[n] = factors.get(n, 0) + 1
        return

    d = pollards_rho(n, rng)
    factorize(d, rng, factors)
    factorize(n // d, rng, factors)


def factor_list(n: int, seed: int = 42) -> List[int]:
    """Return sorted prime factors with multiplicity."""
    if n < 2:
        raise ValueError("n must be >= 2")

    rng = random.Random(seed)
    factors: Dict[int, int] = {}
    factorize(n, rng, factors)

    out: List[int] = []
    for p in sorted(factors):
        out.extend([p] * factors[p])
    return out


def verify_factorization(n: int, factors: List[int]) -> bool:
    product = 1
    for v in factors:
        product *= v
    return product == n and all(is_probable_prime(v) for v in factors)


def main() -> None:
    samples = [
        8051,  # 83 * 97
        10403,  # 101 * 103
        99991 * 100003,  # semiprime ~1e10
        600851475143,  # Project Euler classic composite
        (2**4) * (3**2) * 101 * 1009,  # repeated small factors
    ]

    print("Pollard's Rho integer factorization demo")
    print("-" * 54)

    for n in samples:
        t0 = time.perf_counter()
        factors = factor_list(n, seed=42)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        ok = verify_factorization(n, factors)
        print(f"n = {n}")
        print(f"factors = {factors}")
        print(f"verified = {ok}, time = {elapsed_ms:.3f} ms")
        print("-" * 54)


if __name__ == "__main__":
    main()

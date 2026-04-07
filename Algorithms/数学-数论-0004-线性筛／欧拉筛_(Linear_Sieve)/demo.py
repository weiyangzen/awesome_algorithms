"""Linear sieve (Euler sieve) minimal runnable MVP.

This script computes all primes up to N in O(N) time,
builds a least-prime-factor table, validates correctness
on a small range, and prints a concise demo output.
"""

from __future__ import annotations

from typing import List, Tuple


def linear_sieve(n: int) -> Tuple[List[int], List[int]]:
    """Return (primes up to n, least_prime_factor table [0..n])."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n < 2:
        return [], [0] * (n + 1)

    is_composite = [False] * (n + 1)
    least_prime_factor = [0] * (n + 1)
    primes: List[int] = []

    for i in range(2, n + 1):
        if not is_composite[i]:
            primes.append(i)
            least_prime_factor[i] = i

        for p in primes:
            x = i * p
            if x > n:
                break
            is_composite[x] = True
            least_prime_factor[x] = p
            # Key Euler-sieve cut: each composite is marked once
            # by its smallest prime factor.
            if i % p == 0:
                break

    return primes, least_prime_factor


def is_prime_slow(x: int) -> bool:
    """Simple O(sqrt(n)) primality check for validation."""
    if x < 2:
        return False
    d = 2
    while d * d <= x:
        if x % d == 0:
            return False
        d += 1
    return True


def factorize_by_lpf(x: int, least_prime_factor: List[int]) -> List[Tuple[int, int]]:
    """Prime factorization using least-prime-factor table."""
    if x < 1:
        raise ValueError("x must be >= 1")
    if x == 1:
        return []
    if x >= len(least_prime_factor):
        raise ValueError("x exceeds precomputed sieve limit")

    factors: List[Tuple[int, int]] = []
    while x > 1:
        p = least_prime_factor[x]
        exp = 0
        while x % p == 0:
            x //= p
            exp += 1
        factors.append((p, exp))
    return factors


def validate_small_range(limit: int, primes: List[int], least_prime_factor: List[int]) -> None:
    """Cross-check sieve results against a slow primality test."""
    prime_set = set(primes)
    for x in range(2, limit + 1):
        slow = is_prime_slow(x)
        fast = x in prime_set
        if slow != fast:
            raise AssertionError(f"prime mismatch at {x}: slow={slow}, fast={fast}")

        if not slow and least_prime_factor[x] == 0:
            raise AssertionError(f"missing least-prime-factor for composite {x}")

        if least_prime_factor[x] != 0:
            p = least_prime_factor[x]
            if x % p != 0:
                raise AssertionError(f"invalid least-prime-factor: lpf[{x}]={p}")
            if not is_prime_slow(p):
                raise AssertionError(f"least-prime-factor is not prime: lpf[{x}]={p}")


def format_factorization(x: int, least_prime_factor: List[int]) -> str:
    factors = factorize_by_lpf(x, least_prime_factor)
    if not factors:
        return "1"
    parts = [f"{p}^{e}" if e > 1 else str(p) for p, e in factors]
    return " * ".join(parts)


def main() -> None:
    n = 100_000
    primes, least_prime_factor = linear_sieve(n)

    validate_small_range(limit=500, primes=primes, least_prime_factor=least_prime_factor)

    # Known value: pi(100000) = 9592.
    expected_prime_count = 9_592
    if len(primes) != expected_prime_count:
        raise AssertionError(
            f"unexpected prime count up to {n}: got {len(primes)}, expected {expected_prime_count}"
        )

    print(f"Linear sieve computed primes up to n={n}")
    print(f"Prime count: {len(primes)}")
    print(f"First 15 primes: {primes[:15]}")
    print(f"Last 5 primes: {primes[-5:]}")

    samples = [84, 97, 99_991, 100_000]
    print("Sample factorizations:")
    for x in samples:
        print(f"  {x} = {format_factorization(x, least_prime_factor)}")

    print("Validation checks passed.")


if __name__ == "__main__":
    main()

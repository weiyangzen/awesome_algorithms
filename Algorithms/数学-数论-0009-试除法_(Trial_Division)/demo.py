"""Trial Division MVP.

This script demonstrates:
1) deterministic primality testing by trial division,
2) integer factorization by trial division,
3) lightweight runtime/cost profiling.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import List, Sequence, Tuple


@dataclass
class TrialStats:
    """Instrumentation for algorithm cost visibility."""

    mod_checks: int = 0


def is_prime_trial(n: int, stats: TrialStats | None = None) -> bool:
    """Return True if n is prime, otherwise False.

    Uses trial division with odd divisors only.
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if stats is not None:
        stats.mod_checks += 1
    if n % 2 == 0:
        return False

    d = 3
    while d * d <= n:
        if stats is not None:
            stats.mod_checks += 1
        if n % d == 0:
            return False
        d += 2
    return True


def factorize_trial(n: int, stats: TrialStats | None = None) -> List[Tuple[int, int]]:
    """Return prime factorization as a list of (prime, exponent)."""
    if n == 0:
        raise ValueError("0 has no finite prime factorization.")

    factors: List[Tuple[int, int]] = []
    x = n

    if x < 0:
        factors.append((-1, 1))
        x = -x

    if x == 1:
        return factors + [(1, 1)]

    exponent = 0
    while x % 2 == 0:
        if stats is not None:
            stats.mod_checks += 1
        x //= 2
        exponent += 1
    if exponent > 0:
        factors.append((2, exponent))
    if stats is not None and x > 1:
        stats.mod_checks += 1

    d = 3
    while d * d <= x:
        exponent = 0
        while x % d == 0:
            if stats is not None:
                stats.mod_checks += 1
            x //= d
            exponent += 1
        if exponent > 0:
            factors.append((d, exponent))
        d += 2
        if stats is not None and d * d <= x:
            stats.mod_checks += 1

    if x > 1:
        factors.append((x, 1))

    return factors


def format_factorization(factors: Sequence[Tuple[int, int]]) -> str:
    """Convert factor tuples to a human-readable string."""
    parts = []
    for prime, exp in factors:
        if exp == 1:
            parts.append(str(prime))
        else:
            parts.append(f"{prime}^{exp}")
    return " * ".join(parts)


def primality_demo(numbers: Sequence[int]) -> None:
    """Run primality checks and print metrics."""
    print("=== Primality Demo (Trial Division) ===")
    print(f"{'n':>12} | {'is_prime':>8} | {'mod_checks':>10} | {'elapsed_ms':>10}")
    print("-" * 52)
    for n in numbers:
        stats = TrialStats()
        t0 = perf_counter()
        result = is_prime_trial(n, stats)
        elapsed_ms = (perf_counter() - t0) * 1_000
        print(f"{n:>12} | {str(result):>8} | {stats.mod_checks:>10} | {elapsed_ms:>10.4f}")
    print()


def factorization_demo(numbers: Sequence[int]) -> None:
    """Run factorization and print metrics."""
    print("=== Factorization Demo (Trial Division) ===")
    print(f"{'n':>12} | {'factorization':<28} | {'mod_checks':>10} | {'elapsed_ms':>10}")
    print("-" * 72)
    for n in numbers:
        stats = TrialStats()
        t0 = perf_counter()
        factors = factorize_trial(n, stats)
        elapsed_ms = (perf_counter() - t0) * 1_000
        factor_text = format_factorization(factors)
        print(f"{n:>12} | {factor_text:<28} | {stats.mod_checks:>10} | {elapsed_ms:>10.4f}")
    print()


def scaling_demo() -> None:
    """Show growth trend with larger near-prime odd integers."""
    print("=== Scaling Snapshot ===")
    candidates = [101, 1009, 10007, 100003, 1000003]
    print(f"{'candidate':>12} | {'is_prime':>8} | {'mod_checks':>10}")
    print("-" * 38)
    for n in candidates:
        stats = TrialStats()
        result = is_prime_trial(n, stats)
        print(f"{n:>12} | {str(result):>8} | {stats.mod_checks:>10}")
    print()


def main() -> None:
    primality_numbers = [2, 3, 4, 17, 97, 221, 9973, 99991, 1000003, 2147483647]
    factor_numbers = [84, 221, 360, 8051, 123456, 100160063, -756]

    primality_demo(primality_numbers)
    factorization_demo(factor_numbers)
    scaling_demo()


if __name__ == "__main__":
    main()

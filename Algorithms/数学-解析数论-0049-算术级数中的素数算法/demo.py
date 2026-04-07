"""MVP: primes in arithmetic progressions.

This script demonstrates how to count and list primes in residue classes
`p ≡ a (mod q)` using a sieve table, and compares exact counts with a
simple analytic-number-theory estimate.
"""

from __future__ import annotations

import math
from typing import Dict, List


def sieve_eratosthenes(limit: int) -> bytearray:
    """Return a primality table for 0..limit using Eratosthenes sieve."""
    if limit < 1:
        return bytearray(limit + 1)

    is_prime = bytearray(b"\x01") * (limit + 1)
    is_prime[0] = 0
    is_prime[1] = 0

    bound = int(limit ** 0.5)
    for p in range(2, bound + 1):
        if not is_prime[p]:
            continue
        start = p * p
        step = p
        count = ((limit - start) // step) + 1
        is_prime[start : limit + 1 : step] = b"\x00" * count
    return is_prime


def euler_phi(n: int) -> int:
    """Compute Euler's totient function phi(n)."""
    if n <= 0:
        raise ValueError("n must be positive")

    result = n
    x = n
    p = 2
    while p * p <= x:
        if x % p == 0:
            while x % p == 0:
                x //= p
            result -= result // p
        p += 1
    if x > 1:
        result -= result // x
    return result


def logarithmic_integral_discrete(limit: int, start: int = 2) -> float:
    """A discrete approximation of li(limit): sum_{n=start..limit} 1/log(n)."""
    if limit < start:
        return 0.0

    total = 0.0
    for n in range(max(start, 2), limit + 1):
        total += 1.0 / math.log(n)
    return total


def normalize_residue(a: int, q: int) -> int:
    """Normalize residue to [0, q-1]."""
    if q <= 0:
        raise ValueError("modulus q must be positive")
    return a % q


def primes_in_arithmetic_progression(
    limit: int,
    a: int,
    q: int,
    prime_table: bytearray | None = None,
) -> List[int]:
    """List all primes p <= limit such that p ≡ a (mod q)."""
    if limit < 2:
        return []
    if q <= 0:
        raise ValueError("modulus q must be positive")

    a = normalize_residue(a, q)
    is_prime = prime_table if prime_table is not None else sieve_eratosthenes(limit)
    if len(is_prime) <= limit:
        raise ValueError("prime_table length is smaller than required limit")

    first = a
    if first < 2:
        first += ((2 - first + q - 1) // q) * q

    result: List[int] = []
    for n in range(first, limit + 1, q):
        if is_prime[n]:
            result.append(n)
    return result


def coprime_residues(q: int) -> List[int]:
    """Return residues a in [0, q-1] with gcd(a, q)=1."""
    if q <= 0:
        raise ValueError("modulus q must be positive")
    return [a for a in range(q) if math.gcd(a, q) == 1]


def prime_count_estimate_ap(limit: int, q: int) -> float:
    """Estimate pi(limit; q, a) for gcd(a,q)=1 via li(limit)/phi(q)."""
    if limit < 2:
        return 0.0
    phi_q = euler_phi(q)
    if phi_q <= 0:
        raise ValueError("invalid phi(q)")
    return logarithmic_integral_discrete(limit, start=2) / float(phi_q)


def count_by_residue_classes(
    limit: int,
    q: int,
    prime_table: bytearray | None = None,
) -> Dict[int, int]:
    """Count primes in each reduced residue class modulo q."""
    is_prime = prime_table if prime_table is not None else sieve_eratosthenes(limit)
    counts: Dict[int, int] = {}
    for a in coprime_residues(q):
        counts[a] = len(primes_in_arithmetic_progression(limit, a, q, is_prime))
    return counts


def run_single_experiment(limit: int, q: int, residues: List[int], prime_table: bytearray) -> None:
    """Print exact and estimated counts for selected residue classes."""
    phi_q = euler_phi(q)
    estimate = prime_count_estimate_ap(limit, q)

    print(f"\n=== Experiment: limit={limit}, q={q}, phi(q)={phi_q} ===")
    print("Dirichlet classes (gcd(a,q)=1) should be roughly balanced as limit grows.")

    for raw_a in residues:
        a = normalize_residue(raw_a, q)
        primes = primes_in_arithmetic_progression(limit, a, q, prime_table)
        coprime = math.gcd(a, q) == 1
        est_text = f"{estimate:.2f}" if coprime else "N/A (gcd(a,q) != 1)"
        print(
            f"a={raw_a} -> normalized={a}: count={len(primes):5d}, "
            f"estimate={est_text}, first_10={primes[:10]}"
        )


def main() -> None:
    limit = 200_000
    prime_table = sieve_eratosthenes(limit)

    print("Primes in Arithmetic Progressions - MVP")
    print(f"Global setting: limit={limit}, total_primes={sum(prime_table)}")

    run_single_experiment(limit, q=4, residues=[1, 3], prime_table=prime_table)
    run_single_experiment(limit, q=10, residues=[1, 3, 7, 9], prime_table=prime_table)
    run_single_experiment(limit, q=12, residues=[1, 5, 7, 11, 6], prime_table=prime_table)

    counts_mod_10 = count_by_residue_classes(limit, q=10, prime_table=prime_table)
    print("\nReduced residue class distribution for q=10:")
    for a in sorted(counts_mod_10):
        print(f"  class {a:2d}: {counts_mod_10[a]} primes")


if __name__ == "__main__":
    main()

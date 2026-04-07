"""MVP: prime k-tuple search with an admissible-pattern check.

This script demonstrates a small but complete workflow:
1) define k-tuple patterns via offsets,
2) check admissibility in number-theoretic sense,
3) enumerate tuples up to a bound using a sieve-backed prime table,
4) compare observed counts with a Hardy-Littlewood style estimate.
"""

from __future__ import annotations

from math import isqrt, log
from typing import Iterable, Sequence


def canonicalize_pattern(offsets: Iterable[int]) -> tuple[int, ...]:
    """Normalize a pattern and validate basic constraints."""
    normalized = tuple(sorted(set(int(x) for x in offsets)))
    if not normalized:
        raise ValueError("pattern must contain at least one offset")
    if normalized[0] < 0:
        raise ValueError("offsets must be non-negative")
    if len(normalized) != len(tuple(int(x) for x in offsets)):
        raise ValueError("offsets must be unique")
    return normalized


def build_prime_table(limit: int) -> bytearray:
    """Return primality table in [0, limit] using sieve of Eratosthenes."""
    if limit < 0:
        raise ValueError("limit must be >= 0")

    is_prime = bytearray(b"\x01") * (limit + 1)
    if limit >= 0:
        is_prime[0] = 0
    if limit >= 1:
        is_prime[1] = 0

    for p in range(2, isqrt(limit) + 1):
        if is_prime[p]:
            start = p * p
            step = p
            span = ((limit - start) // step) + 1
            is_prime[start : limit + 1 : step] = b"\x00" * span
    return is_prime


def list_primes_up_to(limit: int) -> list[int]:
    table = build_prime_table(limit)
    return [x for x in range(2, limit + 1) if table[x]]


def is_admissible_pattern(offsets: Sequence[int]) -> bool:
    """Check if a k-tuple pattern is admissible.

    A pattern H={h_i} is admissible if for every prime p,
    residues {-h_i mod p} do not cover all residue classes modulo p.

    For p > k (k=len(H)), full coverage is impossible, so it is enough
    to test primes p <= k.
    """
    pattern = canonicalize_pattern(offsets)
    k = len(pattern)
    small_primes = list_primes_up_to(k)

    for p in small_primes:
        forbidden_residues = {(-h) % p for h in pattern}
        if len(forbidden_residues) == p:
            return False
    return True


def find_prime_k_tuples(
    upper_bound: int,
    offsets: Sequence[int],
    prime_table: Sequence[int] | None = None,
) -> list[tuple[int, ...]]:
    """Enumerate prime tuples with all values <= upper_bound."""
    if upper_bound < 2:
        return []

    pattern = canonicalize_pattern(offsets)
    max_offset = pattern[-1]

    if prime_table is None:
        prime_table = build_prime_table(upper_bound)

    required_len = upper_bound + 1
    if len(prime_table) < required_len:
        raise ValueError(
            f"prime_table length {len(prime_table)} is too short for upper_bound {upper_bound}"
        )

    results: list[tuple[int, ...]] = []
    for base in range(2, upper_bound - max_offset + 1):
        ok = True
        for d in pattern:
            if not prime_table[base + d]:
                ok = False
                break
        if ok:
            results.append(tuple(base + d for d in pattern))
    return results


def singular_series_approx(offsets: Sequence[int], prime_cutoff: int = 2_000) -> float:
    """Compute truncated Hardy-Littlewood singular series for the pattern."""
    if prime_cutoff < 2:
        raise ValueError("prime_cutoff must be >= 2")

    pattern = canonicalize_pattern(offsets)
    k = len(pattern)
    series = 1.0

    for p in list_primes_up_to(prime_cutoff):
        residues = {(-h) % p for h in pattern}
        nu_p = len(residues)
        if nu_p == p:
            return 0.0
        local_factor = (1.0 - nu_p / p) / ((1.0 - 1.0 / p) ** k)
        series *= local_factor
    return series


def logarithmic_integral_k(upper_bound: int, k: int) -> float:
    """Simple discrete approximation of integral dt/(log t)^k on [2, upper_bound]."""
    total = 0.0
    for n in range(2, upper_bound + 1):
        total += 1.0 / (log(n) ** k)
    return total


def hardy_littlewood_estimate(
    upper_bound: int, offsets: Sequence[int], prime_cutoff: int = 2_000
) -> float:
    """Return HL-style heuristic estimate for count of tuples <= upper_bound."""
    pattern = canonicalize_pattern(offsets)
    k = len(pattern)
    return singular_series_approx(pattern, prime_cutoff) * logarithmic_integral_k(
        upper_bound, k
    )


def main() -> None:
    upper_bound = 200_000
    patterns: list[tuple[str, tuple[int, ...]]] = [
        ("孪生素数", (0, 2)),
        ("素数三元组", (0, 2, 6)),
        ("素数四元组", (0, 2, 6, 8)),
    ]

    max_offset = max(max(pat) for _, pat in patterns)
    prime_table = build_prime_table(upper_bound + max_offset)

    print(f"Prime k-tuple demo, upper_bound={upper_bound}")
    print("=" * 72)

    for name, pattern in patterns:
        admissible = is_admissible_pattern(pattern)
        print(f"模式: {name}, offsets={pattern}, admissible={admissible}")
        if not admissible:
            print("  该模式不可容许，按理论只会出现有限次或为空。")
            print("-" * 72)
            continue

        tuples = find_prime_k_tuples(upper_bound, pattern, prime_table)
        estimate = hardy_littlewood_estimate(upper_bound, pattern, prime_cutoff=3_000)

        sample = tuples[:8]
        print(f"  实际计数: {len(tuples)}")
        print(f"  近似计数(HL截断): {estimate:.2f}")
        print(f"  前 {len(sample)} 个样例: {sample}")
        print("-" * 72)


if __name__ == "__main__":
    main()

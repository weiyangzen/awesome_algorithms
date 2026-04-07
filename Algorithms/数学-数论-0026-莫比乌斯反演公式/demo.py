"""Minimal runnable MVP for Möbius inversion formula.

This script demonstrates:
1) linear sieve construction of Möbius function mu(n)
2) divisor-sum convolution f(n) = sum_{d|n} g(d)
3) Möbius inversion recovery g(n) = sum_{d|n} mu(d) * f(n/d)
4) a classical application: counting coprime ordered pairs in [1, N]^2
"""

from __future__ import annotations

from math import gcd


def linear_sieve_mobius(limit: int) -> list[int]:
    """Compute mu[0..limit] with O(limit) linear sieve."""
    if limit < 1:
        return [0] * (limit + 1)

    mu = [0] * (limit + 1)
    is_composite = [False] * (limit + 1)
    primes: list[int] = []
    mu[1] = 1

    for i in range(2, limit + 1):
        if not is_composite[i]:
            primes.append(i)
            mu[i] = -1

        for p in primes:
            v = i * p
            if v > limit:
                break
            is_composite[v] = True
            if i % p == 0:
                mu[v] = 0
                break
            mu[v] = -mu[i]

    return mu


def divisor_sum_convolution(g: list[int]) -> list[int]:
    """Build f where f(n) = sum_{d|n} g(d), 1-indexed semantic on g."""
    n = len(g) - 1
    f = [0] * (n + 1)
    for d in range(1, n + 1):
        gd = g[d]
        for multiple in range(d, n + 1, d):
            f[multiple] += gd
    return f


def mobius_inversion(f: list[int], mu: list[int]) -> list[int]:
    """Recover g from f by g(n) = sum_{d|n} mu(d) * f(n/d)."""
    n = len(f) - 1
    g = [0] * (n + 1)
    for d in range(1, n + 1):
        mud = mu[d]
        if mud == 0:
            continue
        for multiple in range(d, n + 1, d):
            g[multiple] += mud * f[multiple // d]
    return g


def count_coprime_pairs_ordered(n: int, mu: list[int]) -> int:
    """Count ordered pairs (a, b), 1<=a,b<=n, with gcd(a,b)=1."""
    return sum(mu[k] * (n // k) ** 2 for k in range(1, n + 1))


def brute_force_coprime_pairs_ordered(n: int) -> int:
    total = 0
    for a in range(1, n + 1):
        for b in range(1, n + 1):
            if gcd(a, b) == 1:
                total += 1
    return total


def main() -> None:
    n = 30
    mu = linear_sieve_mobius(n)

    # Ground-truth g(n) used for inversion demo.
    g_true = [0] * (n + 1)
    for i in range(1, n + 1):
        g_true[i] = i * i + 3 * i + 1

    f = divisor_sum_convolution(g_true)
    g_recovered = mobius_inversion(f, mu)

    assert g_recovered == g_true, "Möbius inversion failed to recover original sequence"

    print("Möbius inversion recovery check: PASS")
    print("n | mu(n) | g_true(n) | f(n)=sum_{d|n}g(d) | g_recovered(n)")
    print("-" * 62)
    for i in range(1, 13):
        print(
            f"{i:2d} | {mu[i]:5d} | {g_true[i]:9d} | {f[i]:18d} | {g_recovered[i]:13d}"
        )

    test_n = 20
    formula_count = count_coprime_pairs_ordered(test_n, mu)
    brute_count = brute_force_coprime_pairs_ordered(test_n)
    assert formula_count == brute_count, "Coprime pair counting mismatch"

    print("\nCoprime ordered-pair counting check: PASS")
    print(f"N = {test_n}, formula = {formula_count}, brute_force = {brute_count}")


if __name__ == "__main__":
    main()

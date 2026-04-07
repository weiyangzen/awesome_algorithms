"""Minimal runnable MVP for Fermat Primality Test (MATH-0007)."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Optional


@dataclass
class FermatResult:
    n: int
    rounds: int
    probable_prime: bool
    witness: Optional[int]
    reason: str
    tested_bases: list[int]


def is_prime_trial(n: int) -> bool:
    """Deterministic primality check for small/medium integers."""
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False

    limit = math.isqrt(n)
    d = 3
    while d <= limit:
        if n % d == 0:
            return False
        d += 2
    return True


def fermat_primality_test(n: int, rounds: int = 8, seed: int = 2026) -> FermatResult:
    """Return composite/probable-prime decision by Fermat test."""
    if n < 2:
        return FermatResult(
            n=n,
            rounds=0,
            probable_prime=False,
            witness=None,
            reason="n < 2 is not prime",
            tested_bases=[],
        )
    if n in (2, 3):
        return FermatResult(
            n=n,
            rounds=0,
            probable_prime=True,
            witness=None,
            reason="small known prime",
            tested_bases=[],
        )
    if n % 2 == 0:
        return FermatResult(
            n=n,
            rounds=0,
            probable_prime=False,
            witness=2,
            reason="even number greater than 2",
            tested_bases=[],
        )

    rng = random.Random(seed)
    tested_bases: list[int] = []

    for _ in range(rounds):
        a = rng.randrange(2, n - 1)
        tested_bases.append(a)

        g = math.gcd(a, n)
        if g != 1:
            return FermatResult(
                n=n,
                rounds=len(tested_bases),
                probable_prime=False,
                witness=a,
                reason=f"gcd({a}, {n}) = {g} > 1",
                tested_bases=tested_bases,
            )

        if pow(a, n - 1, n) != 1:
            return FermatResult(
                n=n,
                rounds=len(tested_bases),
                probable_prime=False,
                witness=a,
                reason=f"{a}^(n-1) mod n != 1",
                tested_bases=tested_bases,
            )

    return FermatResult(
        n=n,
        rounds=rounds,
        probable_prime=True,
        witness=None,
        reason="all sampled bases passed Fermat congruence",
        tested_bases=tested_bases,
    )


def fermat_fixed_base_check(n: int, a: int) -> bool:
    """Single-base Fermat congruence check (without randomness)."""
    if n < 2:
        return False
    if math.gcd(a, n) != 1:
        return False
    return pow(a, n - 1, n) == 1


def liar_profile_among_coprimes(n: int) -> tuple[int, int, float]:
    """Count coprime bases and Fermat-liar ratio in [2, n-2]."""
    coprime_count = 0
    liar_count = 0

    for a in range(2, n - 1):
        if math.gcd(a, n) == 1:
            coprime_count += 1
            if pow(a, n - 1, n) == 1:
                liar_count += 1

    ratio = (liar_count / coprime_count) if coprime_count else 0.0
    return coprime_count, liar_count, ratio


def print_decision_table() -> None:
    candidates = [
        2,
        3,
        5,
        17,
        97,
        341,   # 11 * 31, base-2 Fermat pseudoprime
        561,   # Carmichael
        1105,  # Carmichael
        1729,  # Carmichael
        2047,  # 23 * 89, often fools weak tests for some bases
        1009,
        1024,
    ]

    print("=== Fermat Primality Test: sample decisions ===")
    print(" n    | exact_prime | fermat_result   | rounds_used | witness")
    print("------|-------------|-----------------|-------------|--------")
    for n in candidates:
        exact = is_prime_trial(n)
        result = fermat_primality_test(n, rounds=8, seed=2026 + n)
        verdict = "probable-prime" if result.probable_prime else "composite"
        witness = str(result.witness) if result.witness is not None else "-"
        print(f"{n:>5} | {str(exact):>11} | {verdict:>15} | {result.rounds:>11} | {witness:>7}")


def print_failure_analysis() -> None:
    print("\n=== Failure analysis ===")

    n = 341
    a = 2
    passed = fermat_fixed_base_check(n=n, a=a)
    print(f"Single-base check: n={n}, a={a}, passes={passed} (341 is actually composite).")

    carmichael_n = 561
    test_bases = [2, 50, 101, 103, 256]
    base_results = [fermat_fixed_base_check(carmichael_n, b) for b in test_bases]
    print(f"Carmichael check: n={carmichael_n}, bases={test_bases}, passes={base_results}")

    coprime_count, liar_count, ratio = liar_profile_among_coprimes(carmichael_n)
    print(
        "Among coprime bases in [2, n-2], "
        f"liars={liar_count}/{coprime_count}, ratio={ratio:.3f}."
    )


def main() -> None:
    print_decision_table()
    print_failure_analysis()


if __name__ == "__main__":
    main()

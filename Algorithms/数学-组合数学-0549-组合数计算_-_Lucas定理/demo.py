"""MVP: compute C(n, k) mod p using Lucas theorem (p must be prime)."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Iterable


def _is_prime(n: int) -> bool:
    """Return True iff n is a prime number (trial division, deterministic)."""
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False

    i = 5
    step = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += step
        step = 6 - step
    return True


def _validate_inputs(n: int, k: int, p: int) -> None:
    """Validate Lucas theorem inputs."""
    if not isinstance(n, int) or not isinstance(k, int) or not isinstance(p, int):
        raise TypeError("n, k, p must all be integers")
    if n < 0 or k < 0:
        raise ValueError("n and k must be non-negative")
    if p <= 1:
        raise ValueError("p must be > 1")
    if not _is_prime(p):
        raise ValueError("Lucas theorem requires p to be prime")


@lru_cache(maxsize=32)
def _factorials_mod_prime(p: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Precompute factorials and inverse factorials modulo prime p."""
    fact = [1] * p
    for i in range(1, p):
        fact[i] = (fact[i - 1] * i) % p

    inv_fact = [1] * p
    inv_fact[p - 1] = pow(fact[p - 1], p - 2, p)  # Fermat inverse.
    for i in range(p - 2, -1, -1):
        inv_fact[i] = (inv_fact[i + 1] * (i + 1)) % p

    return tuple(fact), tuple(inv_fact)


def _small_comb_mod_prime(n: int, k: int, p: int) -> int:
    """Compute C(n, k) mod p for 0 <= n, k < p."""
    if k < 0 or k > n:
        return 0
    fact, inv_fact = _factorials_mod_prime(p)
    return (fact[n] * inv_fact[k] % p) * inv_fact[n - k] % p


def lucas_binom_mod(n: int, k: int, p: int) -> int:
    """Compute C(n, k) mod p by Lucas theorem.

    Lucas theorem (prime p):
    C(n, k) ≡ Π C(n_i, k_i) (mod p),
    where n_i, k_i are base-p digits of n and k.
    """
    _validate_inputs(n, k, p)

    if k > n:
        return 0

    result = 1
    while n > 0 or k > 0:
        n_i = n % p
        k_i = k % p
        if k_i > n_i:
            return 0
        result = (result * _small_comb_mod_prime(n_i, k_i, p)) % p
        n //= p
        k //= p

    return result


def lucas_trace(n: int, k: int, p: int) -> list[tuple[int, int, int, int, int]]:
    """Return per-digit trace: (digit_idx, n_i, k_i, term, cumulative)."""
    _validate_inputs(n, k, p)

    if k > n:
        return []

    rows: list[tuple[int, int, int, int, int]] = []
    cumulative = 1
    idx = 0
    while n > 0 or k > 0:
        n_i = n % p
        k_i = k % p
        term = 0 if k_i > n_i else _small_comb_mod_prime(n_i, k_i, p)
        cumulative = (cumulative * term) % p
        rows.append((idx, n_i, k_i, term, cumulative))
        if term == 0:
            break
        n //= p
        k //= p
        idx += 1

    if not rows:
        rows.append((0, 0, 0, 1, 1))
    return rows


def _check_small_against_math_comb(max_n: int, primes: Iterable[int]) -> None:
    """Validate Lucas implementation against exact math.comb on small range."""
    if max_n < 0:
        raise ValueError("max_n must be non-negative")

    for p in primes:
        for n in range(max_n + 1):
            for k in range(n + 1):
                got = lucas_binom_mod(n, k, p)
                ref = math.comb(n, k) % p
                if got != ref:
                    raise AssertionError(
                        f"mismatch: p={p}, n={n}, k={k}, got={got}, ref={ref}"
                    )

        # Out-of-range k should return 0.
        if lucas_binom_mod(max_n, max_n + 1, p) != 0:
            raise AssertionError(f"k>n boundary failed for p={p}")


def _print_case_table(cases: Iterable[tuple[int, int, int]], direct_check_max_n: int) -> None:
    """Print deterministic demo cases for Lucas theorem."""
    print("=== Binomial Coefficient Mod Prime via Lucas Theorem ===")
    print(f"{'n':>22} {'k':>22} {'p':>5} {'C(n,k) mod p':>14} {'check':>10}")

    for n, k, p in cases:
        value = lucas_binom_mod(n, k, p)

        check = "SKIP"
        if n <= direct_check_max_n:
            ref = math.comb(n, k) % p if k <= n else 0
            check = "PASS" if ref == value else "FAIL"

        print(f"{n:22d} {k:22d} {p:5d} {value:14d} {check:>10}")


def _print_trace(n: int, k: int, p: int) -> None:
    """Print digit-level Lucas decomposition trace for one fixed example."""
    rows = lucas_trace(n, k, p)
    print("\n--- Lucas Digit Trace ---")
    print(f"n={n}, k={k}, p={p}")
    print(f"{'idx':>4} {'n_i':>5} {'k_i':>5} {'term':>8} {'cum':>8}")
    for idx, n_i, k_i, term, cum in rows:
        print(f"{idx:4d} {n_i:5d} {k_i:5d} {term:8d} {cum:8d}")


def main() -> None:
    # 1) Deterministic correctness checks on a small exhaustive range.
    _check_small_against_math_comb(max_n=40, primes=[2, 3, 5, 7, 11, 13, 17, 19])

    # 2) Fixed showcase cases, including very large n/k.
    cases = [
        (10, 3, 7),
        (20, 10, 13),
        (100, 50, 13),
        (1000, 123, 17),
        (10**6 + 123, 10**4 + 77, 97),
        (10**18 + 12345, 10**12 + 567, 97),
        (10**18 + 5, 10**18 + 6, 97),  # k > n boundary.
    ]
    _print_case_table(cases, direct_check_max_n=300)

    # 3) One transparent digit-level trace.
    _print_trace(n=1000, k=456, p=13)

    print("\nExhaustive checks on n<=40 for selected primes: PASS")
    print("Demo finished successfully.")


if __name__ == "__main__":
    main()

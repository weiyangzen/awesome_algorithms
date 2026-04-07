"""Minimal runnable MVP for Cantor-Zassenhaus factorization over GF(p).

This script factors square-free monic polynomials over prime fields GF(p)
using a transparent implementation:
1) Distinct-Degree Factorization (DDF)
2) Equal-Degree Factorization (EDF, Cantor-Zassenhaus randomized split)

No interactive input is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import List, Sequence, Tuple

import numpy as np


Poly = List[int]  # Coefficients in ascending order: [a0, a1, ..., an]


def poly_trim(a: Sequence[int], p: int) -> Poly:
    arr = [x % p for x in a]
    while len(arr) > 1 and arr[-1] == 0:
        arr.pop()
    return arr or [0]


def poly_is_zero(a: Sequence[int]) -> bool:
    return len(a) == 1 and a[0] == 0


def poly_deg(a: Sequence[int]) -> int:
    return -1 if poly_is_zero(a) else len(a) - 1


def poly_monic(a: Sequence[int], p: int) -> Poly:
    a = poly_trim(a, p)
    if poly_is_zero(a):
        return [0]
    inv = pow(a[-1], p - 2, p)
    return [(c * inv) % p for c in a]


def poly_add(a: Sequence[int], b: Sequence[int], p: int) -> Poly:
    n = max(len(a), len(b))
    out = [0] * n
    for i in range(n):
        ai = a[i] if i < len(a) else 0
        bi = b[i] if i < len(b) else 0
        out[i] = (ai + bi) % p
    return poly_trim(out, p)


def poly_sub(a: Sequence[int], b: Sequence[int], p: int) -> Poly:
    n = max(len(a), len(b))
    out = [0] * n
    for i in range(n):
        ai = a[i] if i < len(a) else 0
        bi = b[i] if i < len(b) else 0
        out[i] = (ai - bi) % p
    return poly_trim(out, p)


def poly_mul(a: Sequence[int], b: Sequence[int], p: int) -> Poly:
    if poly_is_zero(a) or poly_is_zero(b):
        return [0]
    out = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        if ai == 0:
            continue
        for j, bj in enumerate(b):
            if bj == 0:
                continue
            out[i + j] = (out[i + j] + ai * bj) % p
    return poly_trim(out, p)


def poly_divmod(a: Sequence[int], b: Sequence[int], p: int) -> Tuple[Poly, Poly]:
    a = poly_trim(a, p)
    b = poly_trim(b, p)
    if poly_is_zero(b):
        raise ZeroDivisionError("polynomial division by zero")

    if poly_deg(a) < poly_deg(b):
        return [0], a

    r = a[:]
    db = poly_deg(b)
    inv_lc_b = pow(b[-1], p - 2, p)
    q = [0] * (poly_deg(a) - db + 1)

    while not poly_is_zero(r) and poly_deg(r) >= db:
        dr = poly_deg(r)
        shift = dr - db
        coeff = (r[-1] * inv_lc_b) % p
        q[shift] = coeff
        for i in range(db + 1):
            r[shift + i] = (r[shift + i] - coeff * b[i]) % p
        r = poly_trim(r, p)

    return poly_trim(q, p), poly_trim(r, p)


def poly_div_exact(a: Sequence[int], b: Sequence[int], p: int) -> Poly:
    q, r = poly_divmod(a, b, p)
    if not poly_is_zero(r):
        raise ValueError("expected exact division but remainder is non-zero")
    return q


def poly_mod(a: Sequence[int], m: Sequence[int], p: int) -> Poly:
    return poly_divmod(a, m, p)[1]


def poly_mul_mod(a: Sequence[int], b: Sequence[int], m: Sequence[int], p: int) -> Poly:
    return poly_mod(poly_mul(a, b, p), m, p)


def poly_pow_mod(base: Sequence[int], exp: int, mod_poly: Sequence[int], p: int) -> Poly:
    if exp < 0:
        raise ValueError("negative exponent is not supported")
    result: Poly = [1]
    x = poly_mod(base, mod_poly, p)
    e = exp
    while e > 0:
        if e & 1:
            result = poly_mul_mod(result, x, mod_poly, p)
        x = poly_mul_mod(x, x, mod_poly, p)
        e >>= 1
    return poly_trim(result, p)


def poly_gcd(a: Sequence[int], b: Sequence[int], p: int) -> Poly:
    a = poly_trim(a, p)
    b = poly_trim(b, p)
    while not poly_is_zero(b):
        _, r = poly_divmod(a, b, p)
        a, b = b, r
    return poly_monic(a, p)


def poly_derivative(a: Sequence[int], p: int) -> Poly:
    if len(a) <= 1:
        return [0]
    out = [(i * a[i]) % p for i in range(1, len(a))]
    return poly_trim(out, p)


def poly_equal(a: Sequence[int], b: Sequence[int], p: int) -> bool:
    return poly_trim(a, p) == poly_trim(b, p)


def poly_to_str(a: Sequence[int], p: int) -> str:
    a = poly_trim(a, p)
    if poly_is_zero(a):
        return "0"

    terms: List[str] = []
    for i in range(len(a) - 1, -1, -1):
        c = a[i] % p
        if c == 0:
            continue
        if i == 0:
            terms.append(f"{c}")
        elif i == 1:
            terms.append("x" if c == 1 else f"{c}*x")
        else:
            terms.append(f"x^{i}" if c == 1 else f"{c}*x^{i}")
    return " + ".join(terms) if terms else "0"


def random_poly(max_degree: int, p: int, rng: np.random.Generator) -> Poly:
    coeffs = rng.integers(low=0, high=p, size=max_degree + 1, dtype=np.int64)
    return poly_trim(coeffs.tolist(), p)


def square_free_check(f: Sequence[int], p: int) -> None:
    f = poly_trim(f, p)
    if poly_deg(f) <= 0:
        raise ValueError("polynomial degree must be at least 1")
    df = poly_derivative(f, p)
    if poly_is_zero(df):
        raise ValueError("derivative is zero: p-th power case is outside this MVP")
    g = poly_gcd(f, df, p)
    if poly_deg(g) > 0:
        raise ValueError("input polynomial must be square-free")


def distinct_degree_factorization(f: Sequence[int], p: int) -> List[Tuple[int, Poly]]:
    """Return list of (d, g_d) where each g_d contains irreducible factors of degree d."""
    f = poly_monic(f, p)
    x_poly: Poly = [0, 1]
    g = f[:]
    h = x_poly[:]
    out: List[Tuple[int, Poly]] = []

    d = 1
    while 2 * d <= poly_deg(g):
        # Frobenius map in GF(p): h <- h^p mod g
        h = poly_pow_mod(h, p, g, p)
        t = poly_gcd(g, poly_sub(h, x_poly, p), p)
        if poly_deg(t) > 0:
            out.append((d, t))
            g = poly_div_exact(g, t, p)
            h = poly_mod(h, g, p) if poly_deg(g) >= 1 else [0]
        d += 1

    if poly_deg(g) > 0:
        out.append((poly_deg(g), g))

    return out


def equal_degree_factorization(
    f: Sequence[int], d: int, p: int, rng: np.random.Generator, max_trials: int = 128
) -> List[Poly]:
    """Cantor-Zassenhaus EDF for odd prime fields GF(p), square-free input."""
    f = poly_monic(f, p)
    n = poly_deg(f)
    if n == d:
        return [f]

    if n % d != 0:
        raise ValueError("degree is not divisible by target equal degree")
    if p == 2:
        raise ValueError("this MVP implements EDF for odd prime p only")

    exponent = (p**d - 1) // 2

    for _ in range(max_trials):
        a = random_poly(n - 1, p, rng)
        if poly_deg(a) <= 0:
            continue

        g = poly_gcd(a, f, p)
        if 0 < poly_deg(g) < n:
            left = equal_degree_factorization(g, d, p, rng, max_trials)
            right = equal_degree_factorization(poly_div_exact(f, g, p), d, p, rng, max_trials)
            return left + right

        h = poly_pow_mod(a, exponent, f, p)
        g = poly_gcd(poly_sub(h, [1], p), f, p)
        if 0 < poly_deg(g) < n:
            left = equal_degree_factorization(g, d, p, rng, max_trials)
            right = equal_degree_factorization(poly_div_exact(f, g, p), d, p, rng, max_trials)
            return left + right

    raise RuntimeError("EDF failed to split polynomial within trial budget")


def factor_square_free_monic(f: Sequence[int], p: int, rng: np.random.Generator) -> List[Poly]:
    f = poly_monic(f, p)
    square_free_check(f, p)

    factors: List[Poly] = []
    for d, block in distinct_degree_factorization(f, p):
        factors.extend(equal_degree_factorization(block, d, p, rng))

    factors = [poly_monic(g, p) for g in factors]
    factors.sort(key=lambda g: (poly_deg(g), tuple(g)))
    return factors


def poly_from_factors(factors: Sequence[Sequence[int]], p: int) -> Poly:
    out: Poly = [1]
    for g in factors:
        out = poly_mul(out, g, p)
    return poly_monic(out, p)


def is_irreducible_bruteforce(f: Sequence[int], p: int) -> bool:
    """Small-degree verifier for demo only (exhaustive divisor test)."""
    f = poly_monic(f, p)
    n = poly_deg(f)
    if n <= 1:
        return True

    for d in range(1, n // 2 + 1):
        for low in product(range(p), repeat=d):
            g = list(low) + [1]  # monic degree d
            if poly_is_zero(poly_mod(f, g, p)):
                return False
    return True


@dataclass(frozen=True)
class Case:
    name: str
    p: int
    expected_factors: Tuple[Tuple[int, ...], ...]  # each factor is ascending coeff tuple


def normalize_factor_list(factors: Sequence[Sequence[int]], p: int) -> List[Tuple[int, ...]]:
    norm = [tuple(poly_monic(g, p)) for g in factors]
    norm.sort(key=lambda g: (len(g), g))
    return norm


def run_case(case: Case, rng: np.random.Generator) -> None:
    p = case.p
    expected = [list(g) for g in case.expected_factors]
    f = poly_from_factors(expected, p)
    found = factor_square_free_monic(f, p, rng)

    rebuilt = poly_from_factors(found, p)
    expected_norm = normalize_factor_list(expected, p)
    found_norm = normalize_factor_list(found, p)
    all_irreducible = all(is_irreducible_bruteforce(g, p) for g in found)

    print("=" * 72)
    print(f"Case: {case.name}")
    print(f"Field: GF({p})")
    print(f"Input  f(x): {poly_to_str(f, p)}")
    for i, g in enumerate(found, start=1):
        print(f"  factor[{i}] (deg {poly_deg(g)}): {poly_to_str(g, p)}")
    print(f"Rebuild check: {poly_equal(rebuilt, f, p)}")
    print(f"Irreducible check (bruteforce): {all_irreducible}")

    if not poly_equal(rebuilt, f, p):
        raise AssertionError("product of factors does not match input polynomial")
    if found_norm != expected_norm:
        raise AssertionError("factorization does not match expected factors")
    if not all_irreducible:
        raise AssertionError("some reported factors are reducible")


def main() -> None:
    rng = np.random.default_rng(20260407)

    cases = [
        Case(
            name="GF(5) mixed degrees: 1 + 2 + 2",
            p=5,
            expected_factors=(
                (1, 1),      # x + 1
                (2, 0, 1),   # x^2 + 2
                (3, 0, 1),   # x^2 + 3
            ),
        ),
        Case(
            name="GF(5) equal-degree block: 3 + 3",
            p=5,
            expected_factors=(
                (1, 2, 0, 1),  # x^3 + 2x + 1
                (1, 1, 0, 1),  # x^3 + x + 1
            ),
        ),
    ]

    for case in cases:
        run_case(case, rng)

    print("=" * 72)
    print("All checks passed.")


if __name__ == "__main__":
    main()

"""Polynomial factorization MVP for MATH-0084.

This script factors univariate integer polynomials over Q using:
1) content extraction (gcd of coefficients),
2) rational root theorem,
3) synthetic division,
4) quadratic discriminant fallback.

It is intentionally compact and non-interactive.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd, isqrt
from typing import List, Sequence, Tuple


@dataclass
class FactorizationResult:
    content: Fraction
    linear_factors: List[Fraction]
    irreducible_remainder: List[Fraction]


def trim_leading_zeros(coeffs: Sequence[Fraction]) -> List[Fraction]:
    out = list(coeffs)
    i = 0
    while i < len(out) - 1 and out[i] == 0:
        i += 1
    return out[i:]


def to_fraction_coeffs(coeffs: Sequence[int | Fraction]) -> List[Fraction]:
    return trim_leading_zeros([Fraction(c) for c in coeffs])


def poly_degree(coeffs: Sequence[Fraction]) -> int:
    return len(coeffs) - 1


def poly_eval(coeffs: Sequence[Fraction], x: Fraction) -> Fraction:
    value = Fraction(0)
    for c in coeffs:
        value = value * x + c
    return value


def synthetic_division(coeffs: Sequence[Fraction], root: Fraction) -> Tuple[List[Fraction], Fraction]:
    n = len(coeffs)
    if n == 0:
        return [], Fraction(0)
    b: List[Fraction] = [coeffs[0]]
    for i in range(1, n):
        b.append(coeffs[i] + b[-1] * root)
    return b[:-1], b[-1]


def divisors(n: int) -> List[int]:
    n = abs(n)
    if n == 0:
        return [0]
    ds: List[int] = []
    for d in range(1, isqrt(n) + 1):
        if n % d == 0:
            ds.append(d)
            if d * d != n:
                ds.append(n // d)
    return sorted(ds)


def rational_root_candidates(coeffs: Sequence[Fraction]) -> List[Fraction]:
    if not coeffs:
        return []
    a0 = coeffs[0]
    an = coeffs[-1]
    if a0.denominator != 1 or an.denominator != 1:
        return []
    lead = int(a0)
    const = int(an)

    if const == 0:
        return [Fraction(0)]

    p_list = divisors(const)
    q_list = divisors(lead)
    candidates = set()
    for p in p_list:
        if p == 0:
            continue
        for q in q_list:
            if q == 0:
                continue
            candidates.add(Fraction(p, q))
            candidates.add(Fraction(-p, q))
    return sorted(candidates)


def gcd_of_ints(values: Sequence[int]) -> int:
    g = 0
    for v in values:
        g = gcd(g, abs(v))
    return g


def extract_content(coeffs: Sequence[Fraction]) -> Tuple[Fraction, List[Fraction]]:
    if all(c == 0 for c in coeffs):
        return Fraction(0), [Fraction(0)]

    if any(c.denominator != 1 for c in coeffs):
        return Fraction(1), list(coeffs)

    ints = [int(c) for c in coeffs]
    g = gcd_of_ints(ints)
    primitive = [Fraction(v // g) for v in ints]

    if primitive[0] < 0:
        g = -g
        primitive = [-c for c in primitive]

    return Fraction(g), primitive


def is_perfect_square(n: int) -> bool:
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n


def factor_quadratic_over_q(coeffs: Sequence[Fraction]) -> List[Fraction] | None:
    if len(coeffs) != 3:
        return None
    a, b, c = coeffs
    if any(x.denominator != 1 for x in coeffs):
        return None
    ai, bi, ci = int(a), int(b), int(c)
    disc = bi * bi - 4 * ai * ci
    if not is_perfect_square(disc):
        return None
    s = isqrt(disc)
    r1 = Fraction(-bi + s, 2 * ai)
    r2 = Fraction(-bi - s, 2 * ai)
    return [r1, r2]


def factor_polynomial_over_q(coeffs: Sequence[int | Fraction]) -> FactorizationResult:
    poly = to_fraction_coeffs(coeffs)
    if not poly:
        return FactorizationResult(Fraction(0), [], [Fraction(0)])

    content, poly = extract_content(poly)
    roots: List[Fraction] = []

    while poly_degree(poly) > 0:
        if poly[-1] == 0:
            roots.append(Fraction(0))
            poly, rem = synthetic_division(poly, Fraction(0))
            if rem != 0:
                break
            poly = trim_leading_zeros(poly)
            continue

        candidates = rational_root_candidates(poly)
        found = None
        for r in candidates:
            if poly_eval(poly, r) == 0:
                found = r
                break

        if found is not None:
            roots.append(found)
            poly, rem = synthetic_division(poly, found)
            if rem != 0:
                break
            poly = trim_leading_zeros(poly)
            continue

        if poly_degree(poly) == 2:
            quad_roots = factor_quadratic_over_q(poly)
            if quad_roots is not None:
                roots.extend(quad_roots)
                poly = [Fraction(1)]
        break

    if poly_degree(poly) == 0:
        content *= poly[0]
        poly = [Fraction(1)]

    return FactorizationResult(content=content, linear_factors=roots, irreducible_remainder=poly)


def format_fraction(v: Fraction) -> str:
    if v.denominator == 1:
        return str(v.numerator)
    return f"{v.numerator}/{v.denominator}"


def format_poly(coeffs: Sequence[Fraction]) -> str:
    coeffs = trim_leading_zeros(coeffs)
    deg = len(coeffs) - 1
    if deg < 0:
        return "0"

    terms: List[str] = []
    for i, c in enumerate(coeffs):
        if c == 0:
            continue
        power = deg - i
        sign = "-" if c < 0 else "+"
        abs_c = -c if c < 0 else c

        if power == 0:
            core = format_fraction(abs_c)
        elif power == 1:
            if abs_c == 1:
                core = "x"
            else:
                core = f"{format_fraction(abs_c)}*x"
        else:
            if abs_c == 1:
                core = f"x^{power}"
            else:
                core = f"{format_fraction(abs_c)}*x^{power}"

        if not terms:
            terms.append(core if sign == "+" else f"-{core}")
        else:
            terms.append(f" {sign} {core}")

    return "".join(terms) if terms else "0"


def format_factorization(result: FactorizationResult) -> str:
    parts: List[str] = []

    if result.content != 1 or (not result.linear_factors and result.irreducible_remainder == [Fraction(1)]):
        parts.append(format_fraction(result.content))

    for r in result.linear_factors:
        if r == 0:
            parts.append("x")
        elif r > 0:
            parts.append(f"(x - {format_fraction(r)})")
        else:
            parts.append(f"(x + {format_fraction(-r)})")

    rem = trim_leading_zeros(result.irreducible_remainder)
    if rem != [Fraction(1)]:
        parts.append(f"({format_poly(rem)})")

    return " * ".join(parts) if parts else "1"


def main() -> None:
    examples = [
        [1, -6, 11, -6],          # (x-1)(x-2)(x-3)
        [2, 3, -11, -6],          # (x-2)(x+3)(2x+1)
        [1, 0, -5, 0, 4],         # (x-2)(x-1)(x+1)(x+2)
        [3, 0, 0, -12],           # 3x^3-12 -> 3(x-2)(x^2+2x+4)
        [1, 0, 1],                # x^2+1 irreducible over Q
    ]

    print("MATH-0084 多项式因式分解算法 Demo")
    for idx, coeffs in enumerate(examples, start=1):
        frac_coeffs = to_fraction_coeffs(coeffs)
        result = factor_polynomial_over_q(coeffs)
        print(f"\n[{idx}] f(x) = {format_poly(frac_coeffs)}")
        print(f"    因式分解: {format_factorization(result)}")


if __name__ == "__main__":
    main()

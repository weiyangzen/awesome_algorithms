"""Subresultant chain (subresultant PRS) minimal runnable MVP.

This script implements a transparent univariate polynomial subresultant chain
algorithm over Q using exact rational arithmetic (`fractions.Fraction`).
No symbolic CAS black-box is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import List, Sequence, Tuple


Poly = List[Fraction]  # Dense coefficients, highest degree first.


@dataclass
class Case:
    name: str
    f: Sequence[int]
    g: Sequence[int]
    expected_gcd: Sequence[int]


@dataclass
class PRSStats:
    iterations: int = 0
    pseudo_division_steps: int = 0


def as_fraction_poly(coeffs: Sequence[int | Fraction]) -> Poly:
    poly = [c if isinstance(c, Fraction) else Fraction(c) for c in coeffs]
    return trim(poly)


def trim(p: Poly) -> Poly:
    i = 0
    while i < len(p) - 1 and p[i] == 0:
        i += 1
    return p[i:]


def is_zero(p: Poly) -> bool:
    q = trim(p)
    return len(q) == 1 and q[0] == 0


def degree(p: Poly) -> int:
    q = trim(p)
    return -1 if is_zero(q) else len(q) - 1


def lc(p: Poly) -> Fraction:
    return trim(p)[0]


def poly_add(a: Poly, b: Poly) -> Poly:
    if len(a) < len(b):
        a = [Fraction(0)] * (len(b) - len(a)) + a
    if len(b) < len(a):
        b = [Fraction(0)] * (len(a) - len(b)) + b
    return trim([x + y for x, y in zip(a, b)])


def poly_sub(a: Poly, b: Poly) -> Poly:
    if len(a) < len(b):
        a = [Fraction(0)] * (len(b) - len(a)) + a
    if len(b) < len(a):
        b = [Fraction(0)] * (len(a) - len(b)) + b
    return trim([x - y for x, y in zip(a, b)])


def poly_mul_scalar(a: Poly, c: Fraction) -> Poly:
    return trim([x * c for x in a])


def poly_mul_xk(a: Poly, k: int) -> Poly:
    if is_zero(a):
        return [Fraction(0)]
    return a + [Fraction(0)] * k


def poly_mul(a: Poly, b: Poly) -> Poly:
    if is_zero(a) or is_zero(b):
        return [Fraction(0)]
    out = [Fraction(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += x * y
    return trim(out)


def poly_pseudo_remainder(a: Poly, b: Poly, stats: PRSStats | None = None) -> Poly:
    """Compute prem(a, b) in K[x].

    Repeatedly applies:
      R <- lc(b) * R - lc(R) * x^t * b, t = deg(R)-deg(b)
    until deg(R) < deg(b).
    """
    if is_zero(b):
        raise ZeroDivisionError("pseudo remainder with zero divisor polynomial")

    r = trim(a[:])
    n = degree(b)
    c = lc(b)

    while (not is_zero(r)) and degree(r) >= n:
        if stats is not None:
            stats.pseudo_division_steps += 1
        t = degree(r) - n
        s = lc(r)
        r = poly_sub(poly_mul_scalar(r, c), poly_mul_scalar(poly_mul_xk(b, t), s))

    return trim(r)


def field_remainder(a: Poly, b: Poly) -> Poly:
    """Euclidean remainder over Q[x], used only for correctness checks."""
    if is_zero(b):
        raise ZeroDivisionError("field remainder with zero divisor polynomial")

    r = trim(a[:])
    db = degree(b)
    lb = lc(b)

    while (not is_zero(r)) and degree(r) >= db:
        t = degree(r) - db
        coeff = lc(r) / lb
        r = poly_sub(r, poly_mul_scalar(poly_mul_xk(b, t), coeff))

    return trim(r)


def make_monic(p: Poly) -> Poly:
    if is_zero(p):
        return [Fraction(0)]
    leading = lc(p)
    return trim([c / leading for c in p])


def subresultant_chain(f: Poly, g: Poly) -> Tuple[List[Poly], PRSStats]:
    """Compute the subresultant PRS chain for univariate polynomials.

    Recurrence (exact rational arithmetic):
      P0 = f, P1 = g
      delta_i = deg(P_{i-1}) - deg(P_i)
      alpha_i = lc(P_i)^(delta_i + 1)
      R_i = prem(alpha_i * P_{i-1}, P_i)
      beta_1 = (-1)^(delta_1 + 1)
      beta_i = -lc(P_{i-1}) * psi_i^(delta_i), i>=2
      psi_1 = -1
      psi_{i+1} = (-lc(P_i))^(delta_i) / psi_i^(delta_i - 1)
      P_{i+1} = R_i / beta_i

    The chain ends when a zero polynomial appears.
    """
    a = trim(f[:])
    b = trim(g[:])

    if is_zero(a) and is_zero(b):
        raise ValueError("both input polynomials are zero")
    if degree(a) < degree(b):
        a, b = b, a

    chain: List[Poly] = [a]
    if is_zero(b):
        return chain, PRSStats(iterations=0, pseudo_division_steps=0)

    chain.append(b)
    stats = PRSStats()

    psi = Fraction(-1)
    i = 1

    while not is_zero(chain[-1]):
        p_prev = chain[-2]
        p_curr = chain[-1]

        delta = degree(p_prev) - degree(p_curr)
        alpha = lc(p_curr) ** (delta + 1)

        r = poly_pseudo_remainder(poly_mul_scalar(p_prev, alpha), p_curr, stats)

        if i == 1:
            beta = Fraction((-1) ** (delta + 1))
        else:
            beta = -lc(p_prev) * (psi ** delta)

        next_poly = [Fraction(0)] if is_zero(r) else trim([c / beta for c in r])
        chain.append(next_poly)
        stats.iterations += 1

        if is_zero(next_poly):
            break

        psi = ((-lc(p_curr)) ** delta) / (psi ** (delta - 1))
        i += 1

    return chain, stats


def gcd_from_chain(chain: Sequence[Poly]) -> Poly:
    if not chain:
        return [Fraction(0)]
    if len(chain) == 1:
        return make_monic(chain[0])
    last_nonzero = chain[-2] if is_zero(chain[-1]) else chain[-1]
    return make_monic(last_nonzero)


def poly_divides(divisor: Poly, target: Poly) -> bool:
    if is_zero(divisor):
        return is_zero(target)
    return is_zero(field_remainder(target, divisor))


def frac_to_str(v: Fraction) -> str:
    return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"


def poly_to_str(p: Poly) -> str:
    q = trim(p)
    if is_zero(q):
        return "0"

    d = degree(q)
    chunks: List[str] = []
    for i, c in enumerate(q):
        if c == 0:
            continue
        power = d - i
        sign = "-" if c < 0 else "+"
        mag = -c if c < 0 else c

        if power == 0:
            body = frac_to_str(mag)
        elif power == 1:
            body = "x" if mag == 1 else f"{frac_to_str(mag)}*x"
        else:
            body = f"x^{power}" if mag == 1 else f"{frac_to_str(mag)}*x^{power}"

        if not chunks:
            chunks.append(body if sign == "+" else f"-{body}")
        else:
            chunks.append(f" {sign} {body}")

    return "".join(chunks) if chunks else "0"


def run_case(case: Case) -> None:
    print(f"=== {case.name} ===")
    f = as_fraction_poly(case.f)
    g = as_fraction_poly(case.g)
    expected = make_monic(as_fraction_poly(case.expected_gcd))

    chain, stats = subresultant_chain(f, g)
    gcd_poly = gcd_from_chain(chain)

    print(f"f(x) = {poly_to_str(f)}")
    print(f"g(x) = {poly_to_str(g)}")
    print("subresultant chain:")
    for idx, poly in enumerate(chain):
        print(f"  P{idx}(x) = {poly_to_str(poly)}")
    print(f"computed gcd (monic) = {poly_to_str(gcd_poly)}")
    print(f"expected gcd (monic) = {poly_to_str(expected)}")
    print(
        f"stats: iterations={stats.iterations}, "
        f"pseudo_division_steps={stats.pseudo_division_steps}"
    )

    assert gcd_poly == expected, "gcd mismatch with expected case value"
    assert poly_divides(gcd_poly, f), "computed gcd does not divide f"
    assert poly_divides(gcd_poly, g), "computed gcd does not divide g"
    print("checks: PASS\n")


def main() -> None:
    cases = [
        Case(
            name="nontrivial-gcd",
            # (x^2-1)(x-2), (x^2-1)(x+3)
            f=[1, -2, -1, 2],
            g=[1, 3, -1, -3],
            expected_gcd=[1, 0, -1],
        ),
        Case(
            name="coprime",
            # gcd(x^3 + x + 1, x^2 - 1) = 1
            f=[1, 0, 1, 1],
            g=[1, 0, -1],
            expected_gcd=[1],
        ),
    ]

    for case in cases:
        run_case(case)


if __name__ == "__main__":
    main()

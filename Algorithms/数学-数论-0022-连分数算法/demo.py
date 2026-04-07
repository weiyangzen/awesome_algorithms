"""Continued fraction algorithm MVP (number theory oriented).

This demo covers:
1) Finite continued fraction for rational numbers.
2) Convergents construction and exact reconstruction.
3) Periodic continued fraction for sqrt(D) (non-square D).
4) Approximation quality via Pell-type identities for sqrt(2).
"""

from __future__ import annotations

from fractions import Fraction
from math import isqrt, sqrt
from typing import List, Sequence, Tuple


def continued_fraction_rational(numer: int, denom: int) -> List[int]:
    """Return the finite continued-fraction coefficients of numer/denom.

    Uses Euclidean-division recursion:
    x = a0 + 1/(a1 + 1/(a2 + ...)).
    """
    if denom == 0:
        raise ValueError("denom must be non-zero")

    if denom < 0:
        numer, denom = -numer, -denom

    coeffs: List[int] = []
    x, y = numer, denom
    while y != 0:
        a = x // y
        coeffs.append(a)
        x, y = y, x - a * y
    return coeffs


def convergents_from_coeffs(coeffs: Sequence[int]) -> List[Fraction]:
    """Build all convergents p_k / q_k from continued-fraction coefficients."""
    if not coeffs:
        raise ValueError("coeffs must be non-empty")

    # p_{-2}=0, p_{-1}=1; q_{-2}=1, q_{-1}=0
    p_prev2, p_prev1 = 0, 1
    q_prev2, q_prev1 = 1, 0
    convergents: List[Fraction] = []

    for a in coeffs:
        p = a * p_prev1 + p_prev2
        q = a * q_prev1 + q_prev2
        convergents.append(Fraction(p, q))
        p_prev2, p_prev1 = p_prev1, p
        q_prev2, q_prev1 = q_prev1, q

    return convergents


def evaluate_continued_fraction(coeffs: Sequence[int]) -> Fraction:
    """Evaluate a finite continued fraction exactly as Fraction."""
    return convergents_from_coeffs(coeffs)[-1]


def continued_fraction_sqrt_period(n: int) -> Tuple[int, List[int]]:
    """Return (a0, period) for sqrt(n) where n is positive non-square.

    sqrt(n) = [a0; overline(period)].
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")

    a0 = isqrt(n)
    if a0 * a0 == n:
        raise ValueError("n must be non-square to have a periodic infinite expansion")

    m = 0
    d = 1
    a = a0
    period: List[int] = []

    # For sqrt(n), the period closes when a hits 2*a0.
    while True:
        m = d * a - m
        d = (n - m * m) // d
        a = (a0 + m) // d
        period.append(a)
        if a == 2 * a0:
            break

    return a0, period


def convergent_for_quadratic_irrational(a0: int, period: Sequence[int], terms_after_a0: int) -> Fraction:
    """Build a convergent of [a0; overline(period)] with a fixed number of terms."""
    if terms_after_a0 < 0:
        raise ValueError("terms_after_a0 must be >= 0")
    if not period:
        raise ValueError("period must be non-empty")

    coeffs = [a0]
    for i in range(terms_after_a0):
        coeffs.append(period[i % len(period)])
    return evaluate_continued_fraction(coeffs)


def format_cf(coeffs: Sequence[int]) -> str:
    """Pretty-print finite continued fraction."""
    if len(coeffs) == 1:
        return f"[{coeffs[0]}]"
    tail = ", ".join(str(x) for x in coeffs[1:])
    return f"[{coeffs[0]}; {tail}]"


def main() -> None:
    print("=== 连分数算法 MVP ===")

    # 1) Rational -> finite continued fraction
    numer, denom = 415, 93
    coeffs = continued_fraction_rational(numer, denom)
    convergents = convergents_from_coeffs(coeffs)

    print("\n[1] 有理数展开")
    print(f"x = {numer}/{denom}")
    print(f"CF(x) = {format_cf(coeffs)}")
    print("收敛分数序列:")
    for i, frac in enumerate(convergents):
        print(f"  C{i} = {frac.numerator}/{frac.denominator}")

    reconstructed = evaluate_continued_fraction(coeffs)
    assert reconstructed == Fraction(numer, denom)
    print(f"重建校验: {reconstructed} == {numer}/{denom}")

    # 2) Periodic CF of quadratic irrational
    n = 23
    a0, period = continued_fraction_sqrt_period(n)
    period_text = ", ".join(str(x) for x in period)
    print("\n[2] 二次无理数周期连分数")
    print(f"sqrt({n}) = [{a0}; overline({period_text})]")

    # 3) Approximation quality for sqrt(2)
    n2 = 2
    a0_2, period_2 = continued_fraction_sqrt_period(n2)
    true_value = sqrt(n2)

    print("\n[3] sqrt(2) 的收敛分数与误差")
    print("k | p/q | |p^2-2q^2| | 误差")
    for k in range(1, 9):
        frac = convergent_for_quadratic_irrational(a0_2, period_2, k)
        p, q = frac.numerator, frac.denominator
        pell_residual = abs(p * p - n2 * q * q)
        err = abs(float(frac) - true_value)
        print(f"{k:>1} | {p}/{q:<8} | {pell_residual:^11} | {err:.3e}")

    # A lightweight bound check: |sqrt(2) - p/q| < 1/q^2 for these convergents.
    for k in range(1, 9):
        frac = convergent_for_quadratic_irrational(a0_2, period_2, k)
        q = frac.denominator
        assert abs(float(frac) - true_value) < 1.0 / (q * q)

    print("\n误差界校验: 前 8 个收敛分数均满足 |x - p/q| < 1/q^2")


if __name__ == "__main__":
    main()

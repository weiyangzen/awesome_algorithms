"""MVP: polynomial square root over formal power series (exact rational arithmetic).

Given A(x)=sum a_i x^i, this script computes B(x) such that B(x)^2=A(x)
when possible. The core routine uses coefficient recursion on formal power series:

  2*b0*b_m + sum_{i=1}^{m-1} b_i*b_{m-i} = a_m

All arithmetic is done with fractions.Fraction to avoid floating-point drift.
"""

from __future__ import annotations

from fractions import Fraction
from math import isqrt
from typing import Iterable, Sequence


def to_fraction_poly(poly: Iterable[int | Fraction]) -> list[Fraction]:
    return [c if isinstance(c, Fraction) else Fraction(c) for c in poly]


def trim(poly: Sequence[Fraction]) -> list[Fraction]:
    coeffs = list(poly)
    while len(coeffs) > 1 and coeffs[-1] == 0:
        coeffs.pop()
    if not coeffs:
        return [Fraction(0)]
    return coeffs


def poly_mul(a: Sequence[Fraction], b: Sequence[Fraction]) -> list[Fraction]:
    a_trim = trim(a)
    b_trim = trim(b)
    if a_trim == [0] or b_trim == [0]:
        return [Fraction(0)]

    out = [Fraction(0)] * (len(a_trim) + len(b_trim) - 1)
    for i, ai in enumerate(a_trim):
        if ai == 0:
            continue
        for j, bj in enumerate(b_trim):
            if bj == 0:
                continue
            out[i + j] += ai * bj
    return trim(out)


def valuation(poly: Sequence[Fraction]) -> int | None:
    for i, coeff in enumerate(poly):
        if coeff != 0:
            return i
    return None


def integer_square_root_exact(n: int) -> int | None:
    if n < 0:
        return None
    r = isqrt(n)
    return r if r * r == n else None


def sqrt_fraction_exact(value: Fraction) -> Fraction | None:
    if value < 0:
        return None
    num_root = integer_square_root_exact(value.numerator)
    den_root = integer_square_root_exact(value.denominator)
    if num_root is None or den_root is None:
        return None
    return Fraction(num_root, den_root)


def sqrt_series_nonzero_constant(
    poly: Sequence[Fraction],
    terms: int,
) -> list[Fraction]:
    """Return first `terms` coefficients of sqrt(poly) with poly[0] != 0.

    The method solves coefficients one by one:
    b_0 = sqrt(a_0),
    b_m = (a_m - sum_{i=1}^{m-1} b_i*b_{m-i}) / (2*b_0).
    """
    if terms <= 0:
        raise ValueError("terms must be positive")
    if not poly:
        raise ValueError("poly must not be empty")

    a0 = poly[0]
    b0 = sqrt_fraction_exact(a0)
    if b0 is None or b0 == 0:
        raise ValueError(
            "constant term must be a non-zero perfect square rational for this routine"
        )

    root = [b0]
    for m in range(1, terms):
        accum = Fraction(0)
        for i in range(1, m):
            accum += root[i] * root[m - i]
        am = poly[m] if m < len(poly) else Fraction(0)
        bm = (am - accum) / (2 * b0)
        root.append(bm)
    return root


def poly_sqrt_series(
    poly: Iterable[int | Fraction],
    terms: int | None = None,
) -> list[Fraction]:
    """Return first `terms` coefficients of sqrt(poly) as formal power series.

    Supports zero/leading-zero handling by factoring x^v, where
    v = valuation(poly). If v is odd, sqrt does not exist in Q[[x]].
    """
    coeffs = trim(to_fraction_poly(poly))
    if terms is None:
        terms = max(1, len(coeffs))
    if terms <= 0:
        raise ValueError("terms must be positive")

    if coeffs == [0]:
        return [Fraction(0)] * terms

    v = valuation(coeffs)
    if v is None:
        return [Fraction(0)] * terms
    if v % 2 == 1:
        raise ValueError("lowest non-zero degree is odd, sqrt series does not exist")

    shift = v // 2
    if shift >= terms:
        return [Fraction(0)] * terms

    reduced = coeffs[v:]
    reduced_terms = terms - shift
    reduced_root = sqrt_series_nonzero_constant(reduced, reduced_terms)

    result = [Fraction(0)] * shift + reduced_root
    if len(result) < terms:
        result.extend([Fraction(0)] * (terms - len(result)))
    return result[:terms]


def exact_polynomial_sqrt(
    poly: Iterable[int | Fraction],
) -> tuple[bool, list[Fraction], list[Fraction]]:
    """Try exact polynomial square root by series construction + exact check.

    Returns:
      (is_perfect_square, root_candidate_trimmed, squared_candidate_trimmed)
    """
    coeffs = trim(to_fraction_poly(poly))

    if coeffs == [0]:
        return True, [Fraction(0)], [Fraction(0)]

    root_terms = (len(coeffs) - 1) // 2 + 1
    root_candidate = trim(poly_sqrt_series(coeffs, root_terms))
    squared = trim(poly_mul(root_candidate, root_candidate))
    return squared == coeffs, root_candidate, squared


def frac_to_str(x: Fraction) -> str:
    if x.denominator == 1:
        return str(x.numerator)
    return f"{x.numerator}/{x.denominator}"


def format_poly(poly: Sequence[Fraction]) -> str:
    coeffs = trim(poly)
    if coeffs == [0]:
        return "0"

    terms: list[str] = []
    for power in range(len(coeffs) - 1, -1, -1):
        coeff = coeffs[power]
        if coeff == 0:
            continue

        abs_coeff = -coeff if coeff < 0 else coeff
        coeff_str = frac_to_str(abs_coeff)

        if power == 0:
            body = coeff_str
        elif power == 1:
            body = "x" if abs_coeff == 1 else f"{coeff_str}*x"
        else:
            body = f"x^{power}" if abs_coeff == 1 else f"{coeff_str}*x^{power}"

        if not terms:
            terms.append(body if coeff > 0 else f"-{body}")
        else:
            terms.append((" + " if coeff > 0 else " - ") + body)

    return "".join(terms)


def run_case(name: str, coeffs: Iterable[int | Fraction]) -> None:
    coeff_list = trim(to_fraction_poly(coeffs))
    print(f"[{name}]")
    print(f"A(x) = {format_poly(coeff_list)}")

    try:
        ok, root, squared = exact_polynomial_sqrt(coeff_list)
    except ValueError as exc:
        print(f"sqrt(A) 不存在于当前约束域: {exc}")
        print("-" * 72)
        return

    print(f"候选 B(x) = {format_poly(root)}")
    print(f"B(x)^2     = {format_poly(squared)}")
    print(f"是否完全平方: {ok}")
    print("-" * 72)


def main() -> None:
    # Coefficients are in ascending order: [a0, a1, a2, ...].
    cases: list[tuple[str, list[int | Fraction]]] = [
        ("完全平方(整数系数)", [9, 12, 10, 4, 1]),  # (x^2 + 2x + 3)^2
        ("非完全平方(同常数项扰动)", [9, 12, 11, 4, 1]),
        ("常数非平方(快速判无解)", [10, 12, 10, 4, 1]),
        ("含偶数阶前导零", [0, 0, 1, 2, 1]),  # x^2*(1+x)^2 = (x + x^2)^2
        ("常数为有理平方", [Fraction(1, 4), 1, 1]),  # (x + 1/2)^2
        ("奇数阶前导零(无解)", [0, 1, 0, 1]),
    ]

    print("Polynomial square root demo (exact arithmetic over Q)")
    print("=" * 72)
    for name, coeffs in cases:
        run_case(name, coeffs)


if __name__ == "__main__":
    main()

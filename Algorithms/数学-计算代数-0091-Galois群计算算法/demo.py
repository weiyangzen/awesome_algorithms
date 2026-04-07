"""Minimal runnable MVP for Galois group computation (MATH-0091).

Scope of this MVP:
- Exact classification over Q for degree-2 and degree-3 polynomials.
- Uses rational root theorem + synthetic division + discriminant tests.
- Degree >= 4 is reported as "unsupported in this MVP" (explicit boundary).
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd, isqrt
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


def normalize_int_coeffs(coeffs: Sequence[int]) -> List[int]:
    """Remove leading zeros and validate polynomial."""
    c = [int(x) for x in coeffs]
    i = 0
    while i < len(c) and c[i] == 0:
        i += 1
    if i == len(c):
        raise ValueError("zero polynomial is not supported")
    return c[i:]


def poly_degree(coeffs: Sequence[int]) -> int:
    return len(coeffs) - 1


def divisors(n: int) -> List[int]:
    """All positive divisors of a non-zero integer n."""
    n = abs(int(n))
    if n == 0:
        return [0]
    out: List[int] = []
    for d in range(1, isqrt(n) + 1):
        if n % d == 0:
            out.append(d)
            if d * d != n:
                out.append(n // d)
    out.sort()
    return out


def lcm(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // gcd(a, b)


def fractions_to_primitive_ints(coeffs: Sequence[Fraction]) -> List[int]:
    """Scale rational coefficients to a primitive integer polynomial."""
    den_lcm = 1
    for x in coeffs:
        den_lcm = lcm(den_lcm, x.denominator)
    ints = [int(x * den_lcm) for x in coeffs]

    g = 0
    for x in ints:
        g = gcd(g, abs(x))
    if g > 1:
        ints = [x // g for x in ints]
    if ints[0] < 0:
        ints = [-x for x in ints]
    return ints


def eval_poly_fraction(coeffs: Sequence[Fraction], x: Fraction) -> Fraction:
    """Horner evaluation in exact arithmetic."""
    acc = Fraction(0)
    for a in coeffs:
        acc = acc * x + a
    return acc


def synthetic_division_fraction(
    coeffs: Sequence[Fraction],
    root: Fraction,
) -> Tuple[List[Fraction], Fraction]:
    """Divide polynomial by (x-root), returning quotient and remainder."""
    n = len(coeffs)
    q: List[Fraction] = [coeffs[0]]
    for i in range(1, n - 1):
        q.append(coeffs[i] + q[-1] * root)
    rem = coeffs[-1] + q[-1] * root
    return q, rem


def rational_root_candidates(int_coeffs: Sequence[int]) -> List[Fraction]:
    """Candidates p/q from Rational Root Theorem for integer polynomial."""
    a0 = int_coeffs[-1]
    an = int_coeffs[0]
    if a0 == 0:
        # x=0 is a root.
        candidates = [Fraction(0, 1)]
    else:
        candidates = []

    p_divs = divisors(a0) if a0 != 0 else [1]
    q_divs = divisors(an) if an != 0 else [1]

    seen = set(candidates)
    for p in p_divs:
        for q in q_divs:
            for sgn in (-1, 1):
                r = Fraction(sgn * p, q)
                if r not in seen:
                    seen.add(r)
                    candidates.append(r)

    # Deterministic order by absolute value then sign.
    candidates.sort(key=lambda z: (abs(float(z)), float(z)))
    return candidates


def extract_rational_linear_factors(
    int_coeffs: Sequence[int],
) -> Tuple[List[Fraction], List[Fraction]]:
    """Extract all rational roots found by theorem + exact verification."""
    current = [Fraction(c) for c in normalize_int_coeffs(int_coeffs)]
    roots: List[Fraction] = []

    while len(current) > 1:
        primitive = fractions_to_primitive_ints(current)
        cands = rational_root_candidates(primitive)
        found = None
        for r in cands:
            if eval_poly_fraction(current, r) == 0:
                found = r
                break
        if found is None:
            break
        q, rem = synthetic_division_fraction(current, found)
        if rem != 0:
            raise RuntimeError("exact synthetic division failed unexpectedly")
        roots.append(found)
        current = q
    return roots, current


def discriminant_deg2(coeffs: Sequence[int]) -> int:
    """Discriminant of ax^2+bx+c."""
    a, b, c = coeffs
    return b * b - 4 * a * c


def discriminant_deg3(coeffs: Sequence[int]) -> int:
    """Discriminant of ax^3+bx^2+cx+d."""
    a, b, c, d = coeffs
    return (
        b * b * c * c
        - 4 * a * c * c * c
        - 4 * b * b * b * d
        - 27 * a * a * d * d
        + 18 * a * b * c * d
    )


def is_perfect_square(n: int) -> bool:
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n


def polynomial_to_string(coeffs: Sequence[int]) -> str:
    """Human-readable polynomial string."""
    c = normalize_int_coeffs(coeffs)
    n = len(c) - 1
    terms: List[str] = []
    for i, a in enumerate(c):
        p = n - i
        if a == 0:
            continue
        sign = "-" if a < 0 else "+"
        aa = abs(a)
        if p == 0:
            body = f"{aa}"
        elif p == 1:
            body = "x" if aa == 1 else f"{aa}x"
        else:
            body = f"x^{p}" if aa == 1 else f"{aa}x^{p}"
        terms.append((sign, body))

    if not terms:
        return "0"

    first_sign, first_body = terms[0]
    out = first_body if first_sign == "+" else f"-{first_body}"
    for sign, body in terms[1:]:
        out += f" {sign} {body}"
    return out


def factor_pattern(total_degree: int, linear_root_count: int, residual_degree: int) -> str:
    """Compact factor-degree signature like 1+2 or 1+1+1."""
    parts = ["1"] * linear_root_count
    if residual_degree > 0:
        parts.append(str(residual_degree))
    # If no residual and no linear roots (only possible for invalid deg), keep degree.
    if not parts:
        parts = [str(total_degree)]
    return "+".join(parts)


@dataclass
class GaloisResult:
    polynomial: str
    degree: int
    factor_pattern: str
    discriminant: int | None
    separable: bool
    group: str
    group_order: int | None
    rationale: str
    approx_roots: List[complex]


def classify_galois_group_q(int_coeffs: Sequence[int]) -> GaloisResult:
    """Classify Galois group over Q for degree 2/3; others are bounded out."""
    coeffs = normalize_int_coeffs(int_coeffs)
    deg = poly_degree(coeffs)
    poly_str = polynomial_to_string(coeffs)
    roots_q, residual = extract_rational_linear_factors(coeffs)
    residual_deg = len(residual) - 1
    pattern = factor_pattern(deg, len(roots_q), residual_deg)

    discr: int | None = None
    separable = True
    group = "UNSUPPORTED"
    order: int | None = None
    reason = ""

    if deg == 2:
        discr = discriminant_deg2(coeffs)
        if discr == 0:
            separable = False
            group = "TRIVIAL"
            order = 1
            reason = "二次重根（判别式为0），分裂域为Q。"
        elif is_perfect_square(discr):
            group = "TRIVIAL"
            order = 1
            reason = "二次可约（根在Q中），Galois群为平凡群。"
        else:
            group = "C2"
            order = 2
            reason = "二次不可约，分裂域是一次二次扩张，群同构于C2。"

    elif deg == 3:
        discr = discriminant_deg3(coeffs)
        if discr == 0:
            separable = False
            group = "NON_SEPARABLE_CASE"
            order = None
            reason = "三次判别式为0，存在重根，本MVP不再给出标准置换群阶。"
        elif len(roots_q) == 0:
            # Irreducible cubic over Q.
            if is_perfect_square(discr):
                group = "A3 (≅ C3)"
                order = 3
                reason = "三次不可约且判别式是平方，群为A3（循环3阶）。"
            else:
                group = "S3"
                order = 6
                reason = "三次不可约且判别式非平方，群为S3。"
        elif len(roots_q) == 3:
            group = "TRIVIAL"
            order = 1
            reason = "三次完全分解于Q，分裂域为Q，群平凡。"
        elif len(roots_q) == 1 and residual_deg == 2:
            residual_ints = fractions_to_primitive_ints(residual)
            d2 = discriminant_deg2(residual_ints)
            if is_perfect_square(d2):
                group = "TRIVIAL"
                order = 1
                reason = "线性因子后剩余二次也可约，整体分解于Q。"
            else:
                group = "C2"
                order = 2
                reason = "线性因子后剩余二次不可约，分裂域由该二次扩张给出。"
        else:
            group = "UNRESOLVED"
            order = None
            reason = "遇到异常分解模式，本MVP未覆盖。"

    else:
        reason = "当前MVP只支持2/3次多项式的Galois群判定。"

    roots_approx = list(np.roots(np.array(coeffs, dtype=np.float64)))
    roots_approx.sort(key=lambda z: (float(np.real(z)), float(np.imag(z))))

    return GaloisResult(
        polynomial=poly_str,
        degree=deg,
        factor_pattern=pattern,
        discriminant=discr,
        separable=separable,
        group=group,
        group_order=order,
        rationale=reason,
        approx_roots=roots_approx,
    )


def complex_list_to_text(roots: Sequence[complex], digits: int = 5) -> str:
    pieces: List[str] = []
    for z in roots:
        re = float(np.real(z))
        im = float(np.imag(z))
        if abs(im) < 10 ** (-digits):
            pieces.append(f"{re:.{digits}f}")
        else:
            sign = "+" if im >= 0 else "-"
            pieces.append(f"{re:.{digits}f}{sign}{abs(im):.{digits}f}j")
    return "[" + ", ".join(pieces) + "]"


def main() -> None:
    print("Galois Group MVP (MATH-0091)")
    print("=" * 84)

    examples: Dict[str, List[int]] = {
        # degree-2
        "quad_irreducible": [1, 0, -2],          # x^2 - 2 -> C2
        "quad_split": [1, -1, -2],               # x^2 - x - 2 -> trivial
        # degree-3
        "cubic_s3": [1, 0, 0, -2],               # x^3 - 2 -> S3
        "cubic_a3": [1, 0, -3, -1],              # x^3 - 3x - 1 -> A3
        "cubic_split": [1, -2, -1, 2],           # (x-2)(x-1)(x+1)
        "cubic_linear_times_quad": [1, -2, 1, -2],  # (x-2)(x^2+1) -> C2
        "cubic_repeated_root": [1, -1, -1, 1],   # (x-1)^2(x+1), discr=0
    }

    expected = {
        "quad_irreducible": "C2",
        "quad_split": "TRIVIAL",
        "cubic_s3": "S3",
        "cubic_a3": "A3 (≅ C3)",
        "cubic_split": "TRIVIAL",
        "cubic_linear_times_quad": "C2",
        "cubic_repeated_root": "NON_SEPARABLE_CASE",
    }

    rows = []
    details = []
    for tag, coeffs in examples.items():
        res = classify_galois_group_q(coeffs)
        if res.group != expected[tag]:
            raise RuntimeError(f"unexpected result for {tag}: got {res.group}")
        rows.append(
            {
                "case": tag,
                "poly": res.polynomial,
                "deg": res.degree,
                "factor_pattern": res.factor_pattern,
                "disc": res.discriminant,
                "group": res.group,
                "order": res.group_order,
                "separable": res.separable,
            }
        )
        details.append((tag, res.rationale, complex_list_to_text(res.approx_roots)))

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print("-" * 84)

    for tag, why, roots_text in details:
        print(f"[{tag}]")
        print(f"  rationale : {why}")
        print(f"  approx roots: {roots_text}")

    # Show explicit unsupported boundary with a quartic example.
    q4 = [1, 0, 0, 0, -2]
    q4_res = classify_galois_group_q(q4)
    print("-" * 84)
    print("Boundary check (quartic):")
    print(f"  poly  : {q4_res.polynomial}")
    print(f"  group : {q4_res.group}")
    print(f"  note  : {q4_res.rationale}")

    print("=" * 84)
    print("Run completed successfully.")


if __name__ == "__main__":
    main()

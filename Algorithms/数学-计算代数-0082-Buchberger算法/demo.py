"""Buchberger algorithm MVP (exact arithmetic, non-interactive demo).

Run:
    python3 demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from typing import Dict, Iterable, List, Sequence, Tuple

Monomial = Tuple[int, ...]
Polynomial = Dict[Monomial, Fraction]


@dataclass(frozen=True)
class Case:
    """Single deterministic demonstration case."""

    name: str
    var_names: Tuple[str, ...]
    generators: Tuple[Polynomial, ...]
    expected_basis_size: int | None = None


@dataclass
class BuchbergerStats:
    """Runtime counters for visibility."""

    processed_pairs: int = 0
    added_polynomials: int = 0
    reduction_steps: int = 0


def as_fraction(value: int | Fraction) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction(value)


def normalize_poly(poly: Polynomial) -> Polynomial:
    """Combine like terms and remove zero coefficients."""
    out: Polynomial = {}
    for monomial, coeff in poly.items():
        c = as_fraction(coeff)
        if c == 0:
            continue
        out[monomial] = out.get(monomial, Fraction(0)) + c
        if out[monomial] == 0:
            del out[monomial]
    return out


def poly_from_terms(nvars: int, terms: Dict[Monomial, int | Fraction]) -> Polynomial:
    """Build a polynomial with monomial dimension checks."""
    out: Polynomial = {}
    for monomial, coeff in terms.items():
        if len(monomial) != nvars:
            raise ValueError(f"Monomial {monomial} has dimension {len(monomial)} != {nvars}")
        out[monomial] = as_fraction(coeff)
    return normalize_poly(out)


def is_zero_poly(poly: Polynomial) -> bool:
    return len(poly) == 0


def leading_monomial(poly: Polynomial) -> Monomial:
    if is_zero_poly(poly):
        raise ValueError("Zero polynomial has no leading monomial")
    return max(poly.keys())  # lex order with x > y > z ...


def leading_coefficient(poly: Polynomial) -> Fraction:
    return poly[leading_monomial(poly)]


def leading_term(poly: Polynomial) -> Tuple[Monomial, Fraction]:
    lm = leading_monomial(poly)
    return lm, poly[lm]


def make_monic(poly: Polynomial) -> Polynomial:
    """Scale polynomial so that leading coefficient becomes 1."""
    if is_zero_poly(poly):
        return {}
    lc = leading_coefficient(poly)
    return {m: c / lc for m, c in poly.items()}


def monomial_mul(a: Monomial, b: Monomial) -> Monomial:
    return tuple(x + y for x, y in zip(a, b))


def monomial_divides(a: Monomial, b: Monomial) -> bool:
    """Return True iff monomial a divides monomial b."""
    return all(x <= y for x, y in zip(a, b))


def monomial_sub(b: Monomial, a: Monomial) -> Monomial:
    """Return b-a exponent-wise; requires a|b."""
    if not monomial_divides(a, b):
        raise ValueError(f"Monomial {a} does not divide {b}")
    return tuple(y - x for x, y in zip(a, b))


def monomial_lcm(a: Monomial, b: Monomial) -> Monomial:
    return tuple(max(x, y) for x, y in zip(a, b))


def poly_add(p: Polynomial, q: Polynomial) -> Polynomial:
    out = dict(p)
    for m, c in q.items():
        out[m] = out.get(m, Fraction(0)) + c
        if out[m] == 0:
            del out[m]
    return out


def poly_sub(p: Polynomial, q: Polynomial) -> Polynomial:
    out = dict(p)
    for m, c in q.items():
        out[m] = out.get(m, Fraction(0)) - c
        if out[m] == 0:
            del out[m]
    return out


def multiply_by_term(poly: Polynomial, coeff: Fraction, monomial: Monomial) -> Polynomial:
    if coeff == 0 or is_zero_poly(poly):
        return {}
    out: Polynomial = {}
    for m, c in poly.items():
        nm = monomial_mul(m, monomial)
        out[nm] = out.get(nm, Fraction(0)) + c * coeff
    return normalize_poly(out)


def s_polynomial(f: Polynomial, g: Polynomial) -> Polynomial:
    """Compute S-polynomial of f and g."""
    lm_f, lc_f = leading_term(f)
    lm_g, lc_g = leading_term(g)
    lcm_lm = monomial_lcm(lm_f, lm_g)

    factor_f = monomial_sub(lcm_lm, lm_f)
    factor_g = monomial_sub(lcm_lm, lm_g)

    left = multiply_by_term(f, Fraction(1, 1) / lc_f, factor_f)
    right = multiply_by_term(g, Fraction(1, 1) / lc_g, factor_g)
    return normalize_poly(poly_sub(left, right))


def remainder_by_basis(dividend: Polynomial, basis: Sequence[Polynomial], stats: BuchbergerStats | None = None) -> Polynomial:
    """Multivariate division remainder of dividend by basis."""
    p = normalize_poly(dividend)
    r: Polynomial = {}

    while not is_zero_poly(p):
        lm_p, lc_p = leading_term(p)
        reduced = False

        for g in basis:
            if is_zero_poly(g):
                continue
            lm_g, lc_g = leading_term(g)
            if monomial_divides(lm_g, lm_p):
                term_m = monomial_sub(lm_p, lm_g)
                term_c = lc_p / lc_g
                p = normalize_poly(poly_sub(p, multiply_by_term(g, term_c, term_m)))
                reduced = True
                if stats is not None:
                    stats.reduction_steps += 1
                break

        if not reduced:
            r[lm_p] = r.get(lm_p, Fraction(0)) + lc_p
            p = dict(p)
            del p[lm_p]
            p = normalize_poly(p)
            if stats is not None:
                stats.reduction_steps += 1

    return normalize_poly(r)


def buchberger(generators: Iterable[Polynomial]) -> Tuple[List[Polynomial], BuchbergerStats]:
    """Compute a Groebner basis using the classic Buchberger loop."""
    G = [make_monic(normalize_poly(f)) for f in generators if not is_zero_poly(normalize_poly(f))]
    if not G:
        raise ValueError("At least one non-zero generator is required.")

    stats = BuchbergerStats()
    pairs: List[Tuple[int, int]] = [(i, j) for i in range(len(G)) for j in range(i + 1, len(G))]

    while pairs:
        i, j = pairs.pop(0)
        stats.processed_pairs += 1

        s = s_polynomial(G[i], G[j])
        h = remainder_by_basis(s, G, stats)
        if not is_zero_poly(h):
            h = make_monic(h)
            k = len(G)
            G.append(h)
            stats.added_polynomials += 1
            for idx in range(k):
                pairs.append((idx, k))

    return G, stats


def canonical_poly_key(poly: Polynomial) -> Tuple[Tuple[Monomial, Fraction], ...]:
    return tuple(sorted(poly.items(), key=lambda item: item[0], reverse=True))


def reduce_groebner_basis(basis: Sequence[Polynomial]) -> List[Polynomial]:
    """Simple reduced-basis post-processing for cleaner output."""
    reduced: List[Polynomial] = []
    for i, f in enumerate(basis):
        others = [g for j, g in enumerate(basis) if j != i]
        r = remainder_by_basis(f, others)
        if not is_zero_poly(r):
            reduced.append(make_monic(r))

    unique: Dict[Tuple[Tuple[Monomial, Fraction], ...], Polynomial] = {}
    for poly in reduced:
        unique[canonical_poly_key(poly)] = poly

    out = list(unique.values())
    out.sort(key=leading_monomial, reverse=True)
    return out


def verify_groebner_basis(basis: Sequence[Polynomial]) -> Tuple[bool, Tuple[int, int] | None, Polynomial | None]:
    """Check Buchberger criterion: all S-polynomials reduce to zero."""
    for i, j in combinations(range(len(basis)), 2):
        s = s_polynomial(basis[i], basis[j])
        rem = remainder_by_basis(s, basis)
        if not is_zero_poly(rem):
            return False, (i, j), rem
    return True, None, None


def monomial_to_str(monomial: Monomial, var_names: Sequence[str]) -> str:
    pieces: List[str] = []
    for var, exp in zip(var_names, monomial):
        if exp == 0:
            continue
        if exp == 1:
            pieces.append(var)
        else:
            pieces.append(f"{var}^{exp}")
    return "*".join(pieces) if pieces else "1"


def fraction_to_str(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def polynomial_to_str(poly: Polynomial, var_names: Sequence[str]) -> str:
    if is_zero_poly(poly):
        return "0"

    monomials = sorted(poly.keys(), reverse=True)
    parts: List[str] = []

    for idx, monomial in enumerate(monomials):
        coeff = poly[monomial]
        sign = "-" if coeff < 0 else "+"
        abs_coeff = abs(coeff)
        mono_str = monomial_to_str(monomial, var_names)

        if mono_str == "1":
            body = fraction_to_str(abs_coeff)
        elif abs_coeff == 1:
            body = mono_str
        else:
            body = f"{fraction_to_str(abs_coeff)}*{mono_str}"

        if idx == 0:
            parts.append(body if sign == "+" else f"-{body}")
        else:
            parts.append(f" {sign} {body}")

    return "".join(parts)


def build_cases() -> List[Case]:
    """Create deterministic examples in Q[x,y] with lex order x > y."""
    xy_minus_1 = poly_from_terms(2, {(1, 1): 1, (0, 0): -1})
    y2_minus_x = poly_from_terms(2, {(0, 2): 1, (1, 0): -1})

    x2_plus_y2_minus_1 = poly_from_terms(2, {(2, 0): 1, (0, 2): 1, (0, 0): -1})
    x_minus_y = poly_from_terms(2, {(1, 0): 1, (0, 1): -1})

    return [
        Case(
            name="curve_intersection_case",
            var_names=("x", "y"),
            generators=(xy_minus_1, y2_minus_x),
            expected_basis_size=2,
        ),
        Case(
            name="line_circle_case",
            var_names=("x", "y"),
            generators=(x2_plus_y2_minus_1, x_minus_y),
            expected_basis_size=2,
        ),
    ]


def run_case(case: Case) -> None:
    print(f"=== {case.name} ===")
    print("input generators:")
    for idx, poly in enumerate(case.generators, start=1):
        print(f"  f{idx} = {polynomial_to_str(poly, case.var_names)}")

    raw_basis, stats = buchberger(case.generators)
    reduced_basis = reduce_groebner_basis(raw_basis)

    print("\nraw Buchberger basis:")
    for idx, poly in enumerate(raw_basis, start=1):
        print(f"  g{idx} = {polynomial_to_str(poly, case.var_names)}")

    print("\nreduced Groebner basis:")
    for idx, poly in enumerate(reduced_basis, start=1):
        print(f"  G{idx} = {polynomial_to_str(poly, case.var_names)}")

    ok, witness_pair, witness_rem = verify_groebner_basis(reduced_basis)
    print("\nvalidation:")
    print(f"  Buchberger criterion passed: {ok}")
    if not ok:
        assert witness_pair is not None
        assert witness_rem is not None
        print(f"  failing pair: {witness_pair}")
        print(f"  non-zero remainder: {polynomial_to_str(witness_rem, case.var_names)}")

    for idx, f in enumerate(case.generators, start=1):
        rem = remainder_by_basis(f, reduced_basis)
        rem_ok = is_zero_poly(rem)
        print(f"  generator f{idx} reduced-to-zero: {rem_ok}")
        if not rem_ok:
            print(f"    remainder: {polynomial_to_str(rem, case.var_names)}")
        assert rem_ok, f"Generator f{idx} is not in ideal generated by reduced basis"

    print("\nstats:")
    print(f"  processed pairs: {stats.processed_pairs}")
    print(f"  added polynomials: {stats.added_polynomials}")
    print(f"  reduction steps: {stats.reduction_steps}")

    if case.expected_basis_size is not None:
        assert len(reduced_basis) == case.expected_basis_size, (
            f"{case.name}: expected basis size {case.expected_basis_size}, got {len(reduced_basis)}"
        )

    assert ok, f"{case.name}: Buchberger criterion failed"
    print()


def main() -> None:
    print("Buchberger algorithm demo (exact arithmetic over Q, lex order x > y)")
    print("No external CAS black box is used for Groebner computation.\n")

    cases = build_cases()
    for case in cases:
        run_case(case)

    print("All cases passed.")


if __name__ == "__main__":
    main()

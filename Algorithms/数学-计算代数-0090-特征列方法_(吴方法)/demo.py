"""Wu's characteristic set method (minimal, non-interactive MVP).

This demo implements a transparent subset of Wu's method for polynomial systems:
- sparse multivariate polynomial representation over Q,
- pseudo-remainder (pseudo-division) by a selected leading variable,
- characteristic-set style triangularization by variable class.

Variable order convention in this script is:
    z < y < x
So class(z)=0, class(y)=1, class(x)=2.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd
from typing import Dict, Iterable, List, Sequence, Tuple

Monomial = Tuple[int, ...]
Polynomial = Dict[Monomial, Fraction]


@dataclass(frozen=True)
class Case:
    """Deterministic demonstration case."""

    name: str
    var_names: Tuple[str, ...]
    generators: Tuple[Polynomial, ...]
    expect_consistent: bool


@dataclass
class WuStats:
    """Runtime counters to make algorithm flow visible."""

    pivots_selected: int = 0
    pseudo_remainder_calls: int = 0
    pseudo_remainder_steps: int = 0


@dataclass
class CharacteristicSetResult:
    """Result container for Wu-style characteristic set computation."""

    chain: List[Polynomial]
    residuals: List[Polynomial]
    stats: WuStats


def as_fraction(value: int | Fraction) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction(value)


def lcm(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // gcd(abs(a), abs(b))


def monomial_order_key(monomial: Monomial) -> Tuple[int, ...]:
    """Lex key with highest-priority variable on the right (x > y > z)."""
    return tuple(reversed(monomial))


def normalize_poly(poly: Polynomial) -> Polynomial:
    out: Polynomial = {}
    for monomial, coeff in poly.items():
        c = as_fraction(coeff)
        if c == 0:
            continue
        out[monomial] = out.get(monomial, Fraction(0)) + c
        if out[monomial] == 0:
            del out[monomial]
    return out


def poly_content(poly: Polynomial) -> Fraction:
    if not poly:
        return Fraction(0)

    gcd_num = 0
    lcm_den = 1
    for coeff in poly.values():
        gcd_num = gcd(gcd_num, abs(coeff.numerator))
        lcm_den = lcm(lcm_den, coeff.denominator)

    if gcd_num == 0:
        return Fraction(0)
    return Fraction(gcd_num, lcm_den)


def leading_monomial(poly: Polynomial) -> Monomial:
    if not poly:
        raise ValueError("Zero polynomial has no leading monomial")
    return max(poly.keys(), key=monomial_order_key)


def canonicalize_poly(poly: Polynomial) -> Polynomial:
    """Normalize, remove scalar content, and fix sign for deterministic output."""
    p = normalize_poly(poly)
    if not p:
        return {}

    content = poly_content(p)
    if content != 0:
        p = {m: c / content for m, c in p.items()}

    lm = leading_monomial(p)
    if p[lm] < 0:
        p = {m: -c for m, c in p.items()}

    return normalize_poly(p)


def poly_key(poly: Polynomial) -> Tuple[Tuple[Monomial, Fraction], ...]:
    return tuple(sorted(poly.items(), key=lambda item: monomial_order_key(item[0]), reverse=True))


def poly_from_terms(nvars: int, terms: Dict[Monomial, int | Fraction]) -> Polynomial:
    out: Polynomial = {}
    for monomial, coeff in terms.items():
        if len(monomial) != nvars:
            raise ValueError(f"Monomial {monomial} dimension mismatch: expected {nvars}")
        out[monomial] = as_fraction(coeff)
    return canonicalize_poly(out)


def is_zero_poly(poly: Polynomial) -> bool:
    return len(poly) == 0


def poly_add(a: Polynomial, b: Polynomial) -> Polynomial:
    out = dict(a)
    for m, c in b.items():
        out[m] = out.get(m, Fraction(0)) + c
        if out[m] == 0:
            del out[m]
    return normalize_poly(out)


def poly_sub(a: Polynomial, b: Polynomial) -> Polynomial:
    out = dict(a)
    for m, c in b.items():
        out[m] = out.get(m, Fraction(0)) - c
        if out[m] == 0:
            del out[m]
    return normalize_poly(out)


def poly_mul(a: Polynomial, b: Polynomial) -> Polynomial:
    if not a or not b:
        return {}

    nvars = len(next(iter(a.keys())))
    out: Polynomial = {}

    for ma, ca in a.items():
        for mb, cb in b.items():
            mm = tuple(ma[i] + mb[i] for i in range(nvars))
            out[mm] = out.get(mm, Fraction(0)) + ca * cb
            if out[mm] == 0:
                del out[mm]

    return normalize_poly(out)


def poly_mul_var_power(poly: Polynomial, var_idx: int, power: int) -> Polynomial:
    if power < 0:
        raise ValueError("power must be non-negative")
    if power == 0 or not poly:
        return dict(poly)

    out: Polynomial = {}
    for monomial, coeff in poly.items():
        mm = list(monomial)
        mm[var_idx] += power
        out[tuple(mm)] = coeff
    return normalize_poly(out)


def poly_class(poly: Polynomial) -> int:
    """Return highest variable index that appears; constants return -1."""
    if not poly:
        return -1

    nvars = len(next(iter(poly.keys())))
    cls = -1
    for monomial in poly:
        for idx in range(nvars):
            if monomial[idx] > 0:
                cls = max(cls, idx)
    return cls


def degree_in_var(poly: Polynomial, var_idx: int) -> int:
    if not poly:
        return -1
    return max(m[var_idx] for m in poly.keys())


def coeff_poly_at_degree(poly: Polynomial, var_idx: int, degree: int) -> Polynomial:
    if not poly:
        return {}

    out: Polynomial = {}
    for monomial, coeff in poly.items():
        if monomial[var_idx] != degree:
            continue
        mm = list(monomial)
        mm[var_idx] = 0
        key = tuple(mm)
        out[key] = out.get(key, Fraction(0)) + coeff
        if out[key] == 0:
            del out[key]

    return normalize_poly(out)


def leading_coeff_poly(poly: Polynomial, var_idx: int) -> Polynomial:
    d = degree_in_var(poly, var_idx)
    if d < 0:
        return {}
    return coeff_poly_at_degree(poly, var_idx, d)


def pseudo_remainder(
    dividend: Polynomial,
    divisor: Polynomial,
    var_idx: int,
    stats: WuStats | None = None,
    max_steps: int = 200,
) -> Polynomial:
    """Pseudo-remainder of dividend by divisor with respect to x_var_idx.

    Implements:
        R <- lc(divisor)*R - lc(R)*x^(deg(R)-deg(divisor))*divisor
    until deg_var(R) < deg_var(divisor).
    """
    if is_zero_poly(divisor):
        raise ValueError("Divisor cannot be zero")

    d_div = degree_in_var(divisor, var_idx)
    if d_div <= 0:
        return canonicalize_poly(dividend)

    if stats is not None:
        stats.pseudo_remainder_calls += 1

    R = canonicalize_poly(dividend)
    B = canonicalize_poly(divisor)

    steps = 0
    while not is_zero_poly(R) and degree_in_var(R, var_idx) >= d_div:
        d_R = degree_in_var(R, var_idx)
        lc_B = leading_coeff_poly(B, var_idx)
        lc_R = leading_coeff_poly(R, var_idx)
        shift = d_R - d_div

        left = poly_mul(lc_B, R)
        right = poly_mul(lc_R, poly_mul_var_power(B, var_idx, shift))
        R = canonicalize_poly(poly_sub(left, right))

        steps += 1
        if stats is not None:
            stats.pseudo_remainder_steps += 1
        if steps > max_steps:
            raise RuntimeError("pseudo_remainder exceeded max_steps; input likely too complex for MVP")

    return canonicalize_poly(R)


def pseudo_reduce_by_chain(poly: Polynomial, chain: Sequence[Polynomial], stats: WuStats | None = None) -> Polynomial:
    r = canonicalize_poly(poly)
    for base in chain:
        if is_zero_poly(r):
            break

        cls = poly_class(base)
        if cls < 0:
            continue

        deg_base = degree_in_var(base, cls)
        if deg_base <= 0:
            continue

        if degree_in_var(r, cls) >= deg_base:
            r = pseudo_remainder(r, base, cls, stats=stats)
    return canonicalize_poly(r)


def unique_polys(polys: Iterable[Polynomial]) -> List[Polynomial]:
    bucket: Dict[Tuple[Tuple[Monomial, Fraction], ...], Polynomial] = {}
    for poly in polys:
        cp = canonicalize_poly(poly)
        if is_zero_poly(cp):
            continue
        bucket[poly_key(cp)] = cp
    return [bucket[k] for k in sorted(bucket.keys())]


def select_pivot(candidates: Sequence[Polynomial], cls: int) -> Polynomial:
    """Pick low-rank polynomial within one class: min degree, then sparse size."""
    return min(
        candidates,
        key=lambda p: (degree_in_var(p, cls), len(p), poly_key(p)),
    )


def wu_characteristic_set(generators: Sequence[Polynomial], nvars: int) -> CharacteristicSetResult:
    """Build a Wu-style triangular characteristic set (MVP variant)."""
    stats = WuStats()
    pool = unique_polys(generators)
    chain: List[Polynomial] = []

    for cls in range(nvars):
        # First reduce all current equations by already selected lower-class chain.
        reduced_pool = [pseudo_reduce_by_chain(poly, chain, stats=stats) for poly in pool]
        pool = unique_polys(reduced_pool)

        candidates = [p for p in pool if poly_class(p) == cls and degree_in_var(p, cls) > 0]
        if not candidates:
            continue

        pivot = select_pivot(candidates, cls)
        chain.append(pivot)
        stats.pivots_selected += 1

        pivot_key = poly_key(pivot)
        removed = False
        rest: List[Polynomial] = []
        for poly in pool:
            if not removed and poly_key(poly) == pivot_key:
                removed = True
                continue
            rest.append(poly)

        deg_pivot = degree_in_var(pivot, cls)
        next_pool: List[Polynomial] = []
        for poly in rest:
            if degree_in_var(poly, cls) >= deg_pivot:
                remainder = pseudo_remainder(poly, pivot, cls, stats=stats)
                if not is_zero_poly(remainder):
                    next_pool.append(remainder)
            else:
                next_pool.append(poly)

        pool = unique_polys(next_pool)

    residuals = [pseudo_reduce_by_chain(poly, chain, stats=stats) for poly in pool]
    residuals = unique_polys(residuals)

    return CharacteristicSetResult(chain=chain, residuals=residuals, stats=stats)


def monomial_to_str(monomial: Monomial, var_names: Sequence[str]) -> str:
    chunks: List[str] = []
    for var, exp in zip(var_names, monomial):
        if exp == 0:
            continue
        if exp == 1:
            chunks.append(var)
        else:
            chunks.append(f"{var}^{exp}")
    return "*".join(chunks) if chunks else "1"


def fraction_to_str(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def polynomial_to_str(poly: Polynomial, var_names: Sequence[str]) -> str:
    if is_zero_poly(poly):
        return "0"

    terms = sorted(poly.items(), key=lambda item: monomial_order_key(item[0]), reverse=True)
    out: List[str] = []

    for idx, (monomial, coeff) in enumerate(terms):
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
            out.append(body if sign == "+" else f"-{body}")
        else:
            out.append(f" {sign} {body}")

    return "".join(out)


def is_triangular_chain(chain: Sequence[Polynomial]) -> bool:
    classes = [poly_class(poly) for poly in chain]
    return classes == sorted(classes) and len(set(classes)) == len(classes)


def has_nonzero_constant(poly: Polynomial) -> bool:
    if not poly:
        return False
    cls = poly_class(poly)
    if cls != -1:
        return False
    return any(coeff != 0 for coeff in poly.values())


def build_cases() -> List[Case]:
    """Build two fixed cases: one consistent and one inconsistent."""
    # Variable order: z < y < x
    nvars = 3

    z_minus_1 = poly_from_terms(nvars, {(1, 0, 0): 1, (0, 0, 0): -1})
    y2_minus_z = poly_from_terms(nvars, {(0, 2, 0): 1, (1, 0, 0): -1})
    x2_minus_y = poly_from_terms(nvars, {(0, 0, 2): 1, (0, 1, 0): -1})
    x3_minus_xy = poly_from_terms(nvars, {(0, 0, 3): 1, (0, 1, 1): -1})

    z_poly = poly_from_terms(nvars, {(1, 0, 0): 1})
    xz_poly = poly_from_terms(nvars, {(1, 0, 1): 1})

    return [
        Case(
            name="consistent_triangularizable_system",
            var_names=("z", "y", "x"),
            generators=(x3_minus_xy, x2_minus_y, y2_minus_z, z_minus_1),
            expect_consistent=True,
        ),
        Case(
            name="inconsistent_system",
            var_names=("z", "y", "x"),
            generators=(xz_poly, z_poly, z_minus_1),
            expect_consistent=False,
        ),
    ]


def run_case(case: Case) -> None:
    nvars = len(case.var_names)
    result = wu_characteristic_set(case.generators, nvars=nvars)

    print(f"\n=== {case.name} ===")
    print(f"variable order: {' < '.join(case.var_names)}")
    print("input generators:")
    for idx, poly in enumerate(case.generators, start=1):
        print(f"  f{idx} = {polynomial_to_str(poly, case.var_names)}")

    print("characteristic chain:")
    for idx, poly in enumerate(result.chain, start=1):
        cls = poly_class(poly)
        leading_var = case.var_names[cls] if cls >= 0 else "const"
        print(f"  C{idx} (class={cls}, lv={leading_var}) = {polynomial_to_str(poly, case.var_names)}")

    if result.residuals:
        print("residual equations after chain-reduction:")
        for idx, poly in enumerate(result.residuals, start=1):
            print(f"  r{idx} = {polynomial_to_str(poly, case.var_names)}")
    else:
        print("residual equations after chain-reduction: (none)")

    reductions = [pseudo_reduce_by_chain(poly, result.chain) for poly in case.generators]
    all_zero = all(is_zero_poly(r) for r in reductions)
    inconsistent = any(has_nonzero_constant(r) for r in reductions + result.residuals)

    print(f"triangular_chain_ok: {is_triangular_chain(result.chain)}")
    print(f"all_generators_reduce_to_zero: {all_zero}")
    print(f"detected_inconsistency: {inconsistent}")
    print(
        "stats: "
        f"pivots={result.stats.pivots_selected}, "
        f"pseudo_remainder_calls={result.stats.pseudo_remainder_calls}, "
        f"pseudo_remainder_steps={result.stats.pseudo_remainder_steps}"
    )

    if case.expect_consistent:
        assert is_triangular_chain(result.chain), "Expected a triangular chain"
        assert all_zero, "Expected all generators to reduce to zero"
        assert not inconsistent, "Consistent case should not contain non-zero constant remainder"
    else:
        assert inconsistent, "Inconsistent case should expose a non-zero constant remainder"


def main() -> None:
    print("MATH-0090 特征列方法 (吴方法) - Minimal MVP")
    for case in build_cases():
        run_case(case)


if __name__ == "__main__":
    main()

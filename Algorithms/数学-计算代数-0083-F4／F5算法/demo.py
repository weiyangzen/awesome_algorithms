"""F4/F5 algorithm MVP (transparent, exact arithmetic, non-interactive).

This is a practical teaching MVP:
- F4-style: process S-polynomials in minimal-degree batches and run matrix row-reduction.
- F5-style (conservative): signature-key cache avoids exact duplicate pair signatures.

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
class F4F5Stats:
    """Runtime counters for observability."""

    processed_pairs: int = 0
    signature_skips: int = 0
    batches: int = 0
    row_reduction_calls: int = 0
    matrix_rows_total: int = 0
    matrix_cols_peak: int = 0
    added_polynomials: int = 0
    remainder_steps: int = 0


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
    """Build a sparse polynomial with dimension checks."""
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
    return max(poly.keys())  # lex order: x > y > z > ...


def leading_coefficient(poly: Polynomial) -> Fraction:
    return poly[leading_monomial(poly)]


def leading_term(poly: Polynomial) -> Tuple[Monomial, Fraction]:
    lm = leading_monomial(poly)
    return lm, poly[lm]


def monomial_degree(monomial: Monomial) -> int:
    return sum(monomial)


def make_monic(poly: Polynomial) -> Polynomial:
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
    """Return exponent-wise difference b-a (requires a|b)."""
    if not monomial_divides(a, b):
        raise ValueError(f"Monomial {a} does not divide {b}")
    return tuple(y - x for x, y in zip(a, b))


def monomial_lcm(a: Monomial, b: Monomial) -> Monomial:
    return tuple(max(x, y) for x, y in zip(a, b))


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


def remainder_by_basis(
    dividend: Polynomial,
    basis: Sequence[Polynomial],
    stats: F4F5Stats | None = None,
) -> Polynomial:
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
                    stats.remainder_steps += 1
                break

        if not reduced:
            r[lm_p] = r.get(lm_p, Fraction(0)) + lc_p
            p = dict(p)
            del p[lm_p]
            p = normalize_poly(p)
            if stats is not None:
                stats.remainder_steps += 1

    return normalize_poly(r)


def canonical_poly_key(poly: Polynomial) -> Tuple[Tuple[Monomial, Fraction], ...]:
    return tuple(sorted(poly.items(), key=lambda item: item[0], reverse=True))


def build_matrix(polys: Sequence[Polynomial]) -> Tuple[List[Monomial], List[List[Fraction]]]:
    """Build coefficient matrix with a shared monomial column space."""
    monomial_set = {m for poly in polys for m in poly}
    columns = sorted(monomial_set, reverse=True)

    rows: List[List[Fraction]] = []
    for poly in polys:
        rows.append([poly.get(m, Fraction(0)) for m in columns])
    return columns, rows


def row_reduce_rref(rows: Sequence[Sequence[Fraction]]) -> List[List[Fraction]]:
    """Exact Gaussian elimination over Q (RREF)."""
    if not rows:
        return []
    ncols = len(rows[0])
    mat = [list(row) for row in rows]

    pivot_row = 0
    for col in range(ncols):
        if pivot_row >= len(mat):
            break

        pivot = None
        for r in range(pivot_row, len(mat)):
            if mat[r][col] != 0:
                pivot = r
                break
        if pivot is None:
            continue

        mat[pivot_row], mat[pivot] = mat[pivot], mat[pivot_row]

        pivot_value = mat[pivot_row][col]
        mat[pivot_row] = [v / pivot_value for v in mat[pivot_row]]

        for r in range(len(mat)):
            if r == pivot_row:
                continue
            factor = mat[r][col]
            if factor == 0:
                continue
            mat[r] = [v - factor * pv for v, pv in zip(mat[r], mat[pivot_row])]

        pivot_row += 1

    return mat


def row_to_poly(row: Sequence[Fraction], columns: Sequence[Monomial]) -> Polynomial:
    poly: Polynomial = {}
    for coeff, monomial in zip(row, columns):
        if coeff != 0:
            poly[monomial] = coeff
    return normalize_poly(poly)


def f4_batch_row_reduce(spolys: Sequence[Polynomial], stats: F4F5Stats) -> List[Polynomial]:
    """F4-style batch linear algebra pass on S-polynomials."""
    active = [normalize_poly(p) for p in spolys if not is_zero_poly(p)]
    if not active:
        return []

    columns, rows = build_matrix(active)
    stats.row_reduction_calls += 1
    stats.matrix_rows_total += len(rows)
    stats.matrix_cols_peak = max(stats.matrix_cols_peak, len(columns))

    reduced_rows = row_reduce_rref(rows)
    out: Dict[Tuple[Tuple[Monomial, Fraction], ...], Polynomial] = {}
    for row in reduced_rows:
        poly = row_to_poly(row, columns)
        if not is_zero_poly(poly):
            out[canonical_poly_key(poly)] = poly
    return list(out.values())


def pair_signature_key(f: Polynomial, g: Polynomial) -> Tuple[Monomial, Monomial, Monomial]:
    """Conservative signature key for duplicate pair suppression."""
    lm_f = leading_monomial(f)
    lm_g = leading_monomial(g)
    if lm_f <= lm_g:
        a, b = lm_f, lm_g
    else:
        a, b = lm_g, lm_f
    return a, b, monomial_lcm(a, b)


def split_min_degree_pairs(
    pairs: Sequence[Tuple[int, int]],
    basis: Sequence[Polynomial],
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Select all pairs with minimal lcm-degree for one batch."""
    if not pairs:
        return [], []

    degree_table: List[Tuple[Tuple[int, int], int]] = []
    for i, j in pairs:
        lm_i = leading_monomial(basis[i])
        lm_j = leading_monomial(basis[j])
        deg = monomial_degree(monomial_lcm(lm_i, lm_j))
        degree_table.append(((i, j), deg))

    min_deg = min(deg for _, deg in degree_table)
    batch = [pair for pair, deg in degree_table if deg == min_deg]
    rest = [pair for pair, deg in degree_table if deg != min_deg]
    return batch, rest


def f4_f5_mvp(generators: Iterable[Polynomial]) -> Tuple[List[Polynomial], F4F5Stats]:
    """Compute a Groebner basis via a small F4/F5-inspired workflow."""
    basis = [make_monic(normalize_poly(f)) for f in generators if not is_zero_poly(normalize_poly(f))]
    if not basis:
        raise ValueError("At least one non-zero generator is required.")

    stats = F4F5Stats()
    seen_pair_signatures: set[Tuple[Monomial, Monomial, Monomial]] = set()

    pairs: List[Tuple[int, int]] = [(i, j) for i in range(len(basis)) for j in range(i + 1, len(basis))]
    poly_key_set = {canonical_poly_key(poly) for poly in basis}

    while pairs:
        batch, pairs = split_min_degree_pairs(pairs, basis)
        stats.batches += 1

        s_batch: List[Polynomial] = []
        for i, j in batch:
            stats.processed_pairs += 1
            sig_key = pair_signature_key(basis[i], basis[j])
            if sig_key in seen_pair_signatures:
                stats.signature_skips += 1
                continue
            seen_pair_signatures.add(sig_key)

            s = s_polynomial(basis[i], basis[j])
            if not is_zero_poly(s):
                s_batch.append(s)

        if not s_batch:
            continue

        # F4-style linear algebra pre-pass.
        matrix_reduced = f4_batch_row_reduce(s_batch, stats)

        # Safety: also reduce original S-polynomials to keep the MVP robust.
        candidate_pool = matrix_reduced + s_batch

        candidate_remainders: Dict[Tuple[Tuple[Monomial, Fraction], ...], Polynomial] = {}
        for poly in candidate_pool:
            rem = remainder_by_basis(poly, basis, stats)
            if is_zero_poly(rem):
                continue
            rem = make_monic(rem)
            candidate_remainders[canonical_poly_key(rem)] = rem

        for rem in candidate_remainders.values():
            rem2 = remainder_by_basis(rem, basis, stats)
            if is_zero_poly(rem2):
                continue
            rem2 = make_monic(rem2)
            key = canonical_poly_key(rem2)
            if key in poly_key_set:
                continue

            idx = len(basis)
            basis.append(rem2)
            poly_key_set.add(key)
            stats.added_polynomials += 1
            for old_idx in range(idx):
                pairs.append((old_idx, idx))

    return basis, stats


def reduce_groebner_basis(basis: Sequence[Polynomial]) -> List[Polynomial]:
    """Simple reduced-basis post-processing."""
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

    x2_minus_y = poly_from_terms(2, {(2, 0): 1, (0, 1): -1})

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
        Case(
            name="three_generators_case",
            var_names=("x", "y"),
            generators=(xy_minus_1, y2_minus_x, x2_minus_y),
            expected_basis_size=None,
        ),
    ]


def run_case(case: Case) -> None:
    print(f"=== {case.name} ===")
    print("input generators:")
    for idx, poly in enumerate(case.generators, start=1):
        print(f"  f{idx} = {polynomial_to_str(poly, case.var_names)}")

    raw_basis, stats = f4_f5_mvp(case.generators)
    reduced_basis = reduce_groebner_basis(raw_basis)

    print("\nraw F4/F5-like basis:")
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
    print(f"  signature skips: {stats.signature_skips}")
    print(f"  batches: {stats.batches}")
    print(f"  row-reduction calls: {stats.row_reduction_calls}")
    print(f"  matrix rows total: {stats.matrix_rows_total}")
    print(f"  matrix cols peak: {stats.matrix_cols_peak}")
    print(f"  added polynomials: {stats.added_polynomials}")
    print(f"  remainder steps: {stats.remainder_steps}")

    if case.expected_basis_size is not None:
        assert len(reduced_basis) == case.expected_basis_size, (
            f"{case.name}: expected basis size {case.expected_basis_size}, got {len(reduced_basis)}"
        )

    assert ok, f"{case.name}: Buchberger criterion failed"
    print()


def main() -> None:
    print("F4/F5 algorithm MVP demo (exact arithmetic over Q, lex order x > y)")
    print("No external CAS black box is used for Groebner computation.\n")

    for case in build_cases():
        run_case(case)

    print("All cases passed.")


if __name__ == "__main__":
    main()

"""Minimal runnable MVP for Rational Canonical Form (有理标准形).

Core idea of this MVP:
- Work over exact rational numbers via `fractions.Fraction`.
- Build polynomial matrix M(x) = xI - A.
- Compute determinantal divisors Δ_k(x) by gcd of all kxk minors of M(x).
- Recover invariant factors d_k(x) = Δ_k / Δ_{k-1}.
- Build Rational Canonical Form as block diagonal companion matrices of
  non-unit invariant factors.

No interactive input is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, permutations
from typing import Iterable, List, Sequence, Tuple

import numpy as np

Poly = Tuple[Fraction, ...]  # low -> high coefficients
MatrixQ = List[List[Fraction]]
PolyMatrix = List[List[Poly]]


# -----------------------------
# Polynomial helpers over Q[x]
# -----------------------------

def frac(x: int | Fraction) -> Fraction:
    return x if isinstance(x, Fraction) else Fraction(x)


def poly_trim(p: Sequence[Fraction]) -> Poly:
    q = list(p)
    while len(q) > 1 and q[-1] == 0:
        q.pop()
    if not q:
        return (Fraction(0),)
    return tuple(q)


def poly_const(c: int | Fraction) -> Poly:
    return (frac(c),)


def poly_zero() -> Poly:
    return (Fraction(0),)


def poly_one() -> Poly:
    return (Fraction(1),)


def poly_x() -> Poly:
    return (Fraction(0), Fraction(1))


def poly_is_zero(p: Poly) -> bool:
    return len(p) == 1 and p[0] == 0


def poly_degree(p: Poly) -> int:
    return -1 if poly_is_zero(p) else len(p) - 1


def poly_lc(p: Poly) -> Fraction:
    return p[-1]


def poly_add(a: Poly, b: Poly) -> Poly:
    n = max(len(a), len(b))
    out = [Fraction(0)] * n
    for i in range(n):
        ai = a[i] if i < len(a) else Fraction(0)
        bi = b[i] if i < len(b) else Fraction(0)
        out[i] = ai + bi
    return poly_trim(out)


def poly_neg(a: Poly) -> Poly:
    return tuple(-c for c in a)


def poly_sub(a: Poly, b: Poly) -> Poly:
    return poly_add(a, poly_neg(b))


def poly_scale(a: Poly, s: Fraction) -> Poly:
    return poly_trim([c * s for c in a])


def poly_mul(a: Poly, b: Poly) -> Poly:
    if poly_is_zero(a) or poly_is_zero(b):
        return poly_zero()
    out = [Fraction(0)] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            out[i + j] += ai * bj
    return poly_trim(out)


def poly_divmod(a: Poly, b: Poly) -> Tuple[Poly, Poly]:
    if poly_is_zero(b):
        raise ZeroDivisionError("Polynomial division by zero.")
    if poly_is_zero(a):
        return poly_zero(), poly_zero()

    r = list(a)
    db = poly_degree(b)
    lb = poly_lc(b)
    q = [Fraction(0)] * max(0, (len(a) - len(b) + 1))

    while len(r) >= len(b) and not (len(r) == 1 and r[0] == 0):
        dr = len(r) - 1
        t_deg = dr - db
        t_coeff = r[-1] / lb
        q[t_deg] += t_coeff
        for j in range(len(b)):
            r[t_deg + j] -= t_coeff * b[j]
        while len(r) > 1 and r[-1] == 0:
            r.pop()
    return poly_trim(q), poly_trim(r)


def poly_exact_div(a: Poly, b: Poly) -> Poly:
    q, r = poly_divmod(a, b)
    if not poly_is_zero(r):
        raise ValueError("Expected exact polynomial division but got non-zero remainder.")
    return q


def poly_monic(a: Poly) -> Poly:
    if poly_is_zero(a):
        return a
    lc = poly_lc(a)
    return poly_scale(a, Fraction(1, 1) / lc)


def poly_gcd(a: Poly, b: Poly) -> Poly:
    x = poly_trim(a)
    y = poly_trim(b)
    if poly_is_zero(x):
        return poly_monic(y)
    if poly_is_zero(y):
        return poly_monic(x)
    while not poly_is_zero(y):
        _, r = poly_divmod(x, y)
        x, y = y, r
    return poly_monic(x)


def poly_divides(a: Poly, b: Poly) -> bool:
    if poly_is_zero(a):
        return poly_is_zero(b)
    _, r = poly_divmod(b, a)
    return poly_is_zero(r)


def poly_equal(a: Poly, b: Poly) -> bool:
    return poly_trim(a) == poly_trim(b)


def poly_from_high(coeffs_high: Sequence[int | Fraction]) -> Poly:
    return poly_trim([frac(c) for c in reversed(coeffs_high)])


def poly_to_str(p: Poly) -> str:
    p = poly_trim(p)
    if poly_is_zero(p):
        return "0"

    terms: List[str] = []
    for deg in range(len(p) - 1, -1, -1):
        c = p[deg]
        if c == 0:
            continue

        abs_c = abs(c)
        sign = "-" if c < 0 else "+"

        if deg == 0:
            core = f"{abs_c}"
        elif deg == 1:
            if abs_c == 1:
                core = "x"
            else:
                core = f"{abs_c}*x"
        else:
            if abs_c == 1:
                core = f"x^{deg}"
            else:
                core = f"{abs_c}*x^{deg}"

        if not terms:
            terms.append(core if c > 0 else f"-{core}")
        else:
            terms.append(f" {sign} {core}")

    return "".join(terms)


# -----------------------------
# Rational matrix helpers
# -----------------------------

def eye_q(n: int) -> MatrixQ:
    return [[Fraction(1 if i == j else 0) for j in range(n)] for i in range(n)]


def mat_mul_q(a: MatrixQ, b: MatrixQ) -> MatrixQ:
    n = len(a)
    m = len(a[0])
    if m != len(b):
        raise ValueError("Matrix dimensions mismatch for multiplication.")
    p = len(b[0])
    out = [[Fraction(0) for _ in range(p)] for _ in range(n)]
    for i in range(n):
        for k in range(m):
            aik = a[i][k]
            if aik == 0:
                continue
            for j in range(p):
                out[i][j] += aik * b[k][j]
    return out


def mat_inv_q(a: MatrixQ) -> MatrixQ:
    n = len(a)
    if any(len(row) != n for row in a):
        raise ValueError("Matrix inverse requires square matrix.")

    aug: List[List[Fraction]] = [row[:] + eye_row[:] for row, eye_row in zip(a, eye_q(n))]

    for col in range(n):
        pivot = None
        for r in range(col, n):
            if aug[r][col] != 0:
                pivot = r
                break
        if pivot is None:
            raise ValueError("Matrix is singular over Q.")

        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]

        pv = aug[col][col]
        for j in range(2 * n):
            aug[col][j] /= pv

        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if factor == 0:
                continue
            for j in range(2 * n):
                aug[r][j] -= factor * aug[col][j]

    inv = [row[n:] for row in aug]
    return inv


def to_float_np(a: MatrixQ) -> np.ndarray:
    return np.array([[float(v) for v in row] for row in a], dtype=float)


# -----------------------------
# Invariant factors via minors
# -----------------------------

def xi_minus_a(a: MatrixQ) -> PolyMatrix:
    n = len(a)
    out: PolyMatrix = [[poly_zero() for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                out[i][j] = poly_trim([-a[i][j], Fraction(1)])  # x - a_ii
            else:
                out[i][j] = poly_const(-a[i][j])
    return out


def permutation_sign(perm: Sequence[int]) -> int:
    inv_count = 0
    n = len(perm)
    for i in range(n):
        for j in range(i + 1, n):
            if perm[i] > perm[j]:
                inv_count += 1
    return -1 if (inv_count % 2) else 1


def poly_det(mat: PolyMatrix) -> Poly:
    n = len(mat)
    if n == 0:
        return poly_one()
    if any(len(row) != n for row in mat):
        raise ValueError("Determinant requires a square polynomial matrix.")
    if n == 1:
        return mat[0][0]

    total = poly_zero()
    for perm in permutations(range(n)):
        term = poly_one()
        for i in range(n):
            term = poly_mul(term, mat[i][perm[i]])
        if permutation_sign(perm) < 0:
            term = poly_neg(term)
        total = poly_add(total, term)
    return poly_trim(total)


def all_k_minors_det(poly_mat: PolyMatrix, k: int) -> List[Poly]:
    n = len(poly_mat)
    dets: List[Poly] = []
    for rows in combinations(range(n), k):
        for cols in combinations(range(n), k):
            minor = [[poly_mat[r][c] for c in cols] for r in rows]
            dets.append(poly_det(minor))
    return dets


def invariant_factors_from_matrix(a: MatrixQ) -> List[Poly]:
    n = len(a)
    if any(len(row) != n for row in a):
        raise ValueError("Input matrix A must be square.")

    pmat = xi_minus_a(a)
    deltas: List[Poly] = [poly_one()]  # Delta_0

    for k in range(1, n + 1):
        dets = all_k_minors_det(pmat, k)
        g = poly_zero()
        for d in dets:
            if poly_is_zero(d):
                continue
            g = poly_monic(d) if poly_is_zero(g) else poly_gcd(g, d)
        deltas.append(poly_monic(g) if not poly_is_zero(g) else poly_zero())

    inv: List[Poly] = []
    for k in range(1, n + 1):
        dk = deltas[k]
        dkm1 = deltas[k - 1]
        if poly_is_zero(dk):
            inv.append(poly_zero())
            continue
        inv_k = poly_monic(poly_exact_div(dk, dkm1))
        inv.append(inv_k)

    return inv


# -----------------------------
# Rational canonical form builder
# -----------------------------

def companion_matrix(poly: Poly) -> MatrixQ:
    p = poly_monic(poly)
    deg = poly_degree(p)
    if deg <= 0:
        raise ValueError("Companion matrix needs a non-constant monic polynomial.")

    # p(x) = x^deg + a_{deg-1}x^{deg-1} + ... + a_0
    coeff_low = list(p)  # a_0 ... a_{deg-1}, 1
    if coeff_low[-1] != 1:
        raise ValueError("Polynomial must be monic for companion matrix.")

    c = [[Fraction(0) for _ in range(deg)] for _ in range(deg)]
    for i in range(1, deg):
        c[i][i - 1] = Fraction(1)
    for i in range(deg):
        c[i][deg - 1] = -coeff_low[i]
    return c


def block_diag_q(blocks: Sequence[MatrixQ]) -> MatrixQ:
    total = sum(len(b) for b in blocks)
    out = [[Fraction(0) for _ in range(total)] for _ in range(total)]
    offset = 0
    for b in blocks:
        m = len(b)
        for i in range(m):
            for j in range(m):
                out[offset + i][offset + j] = b[i][j]
        offset += m
    return out


def rational_canonical_form_from_invariant_factors(factors: Sequence[Poly]) -> MatrixQ:
    blocks = [companion_matrix(f) for f in factors if poly_degree(f) >= 1]
    if not blocks:
        raise ValueError("No non-unit invariant factors found.")
    return block_diag_q(blocks)


# -----------------------------
# Demo assembly + checks
# -----------------------------
@dataclass
class RCFReport:
    dimension: int
    recovered_invariant_factors: List[str]
    canonical_invariant_factors: List[str]
    divisibility_chain_ok: bool
    same_invariants_after_canonicalization: bool
    same_characteristic_polynomial: bool
    nontrivial_similarity_example: bool



def build_demo_input() -> Tuple[MatrixQ, List[Poly], MatrixQ]:
    # Choose a known invariant-factor chain f1 | f2.
    f1 = poly_from_high([1, 0, 1])  # x^2 + 1
    f2 = poly_from_high([1, 2, 1, 2])  # x^3 + 2x^2 + x + 2 = (x^2+1)(x+2)

    f_blocks = [companion_matrix(f1), companion_matrix(f2)]
    f_canonical = block_diag_q(f_blocks)

    # A simple unimodular upper-triangular P (det = 1), invertible over Z and Q.
    p = [
        [Fraction(1), Fraction(1), Fraction(0), Fraction(0), Fraction(0)],
        [Fraction(0), Fraction(1), Fraction(1), Fraction(0), Fraction(0)],
        [Fraction(0), Fraction(0), Fraction(1), Fraction(1), Fraction(0)],
        [Fraction(0), Fraction(0), Fraction(0), Fraction(1), Fraction(1)],
        [Fraction(0), Fraction(0), Fraction(0), Fraction(0), Fraction(1)],
    ]
    p_inv = mat_inv_q(p)

    # A = P * F * P^{-1}, so A is similar to canonical F but typically not equal to F.
    a = mat_mul_q(mat_mul_q(p, f_canonical), p_inv)
    return a, [f1, f2], f_canonical


def run_checks(
    a: MatrixQ,
    expected_chain: Sequence[Poly],
    recovered_chain: Sequence[Poly],
    f_canonical: MatrixQ,
) -> RCFReport:
    recovered_non_unit = [poly_monic(f) for f in recovered_chain if poly_degree(f) >= 1]
    canonical_recovered = [
        poly_monic(f) for f in invariant_factors_from_matrix(f_canonical) if poly_degree(f) >= 1
    ]

    # Divisibility chain f1 | f2 | ...
    chain_ok = True
    for i in range(len(recovered_non_unit) - 1):
        if not poly_divides(recovered_non_unit[i], recovered_non_unit[i + 1]):
            chain_ok = False
            break

    # Characteristic polynomial = product of non-unit invariant factors.
    def product(polys: Iterable[Poly]) -> Poly:
        out = poly_one()
        for p in polys:
            out = poly_mul(out, p)
        return poly_monic(out)

    char_a = poly_monic(poly_det(xi_minus_a(a)))
    char_f = poly_monic(poly_det(xi_minus_a(f_canonical)))

    expected_norm = [poly_monic(p) for p in expected_chain]
    recovered_eq_expected = len(recovered_non_unit) == len(expected_norm) and all(
        poly_equal(x, y) for x, y in zip(recovered_non_unit, expected_norm)
    )

    same_invariants_after_canonical = len(recovered_non_unit) == len(canonical_recovered) and all(
        poly_equal(x, y) for x, y in zip(recovered_non_unit, canonical_recovered)
    )

    same_char_poly = poly_equal(char_a, char_f) and poly_equal(char_a, product(recovered_non_unit))

    if not recovered_eq_expected:
        raise AssertionError("Recovered invariant factors do not match the planted ground truth.")
    if not chain_ok:
        raise AssertionError("Invariant factors do not satisfy divisibility chain.")
    if not same_invariants_after_canonical:
        raise AssertionError("Canonical form does not preserve invariant factors.")
    if not same_char_poly:
        raise AssertionError("Characteristic polynomial consistency check failed.")

    a_np = to_float_np(a)
    f_np = to_float_np(f_canonical)
    nontrivial = not np.allclose(a_np, f_np)
    if not nontrivial:
        raise AssertionError("Demo matrix unexpectedly equals canonical matrix; choose a nontrivial similarity.")

    return RCFReport(
        dimension=len(a),
        recovered_invariant_factors=[poly_to_str(p) for p in recovered_non_unit],
        canonical_invariant_factors=[poly_to_str(p) for p in canonical_recovered],
        divisibility_chain_ok=chain_ok,
        same_invariants_after_canonicalization=same_invariants_after_canonical,
        same_characteristic_polynomial=same_char_poly,
        nontrivial_similarity_example=nontrivial,
    )



def main() -> None:
    a, expected_chain, _ = build_demo_input()

    recovered_all = invariant_factors_from_matrix(a)
    recovered_non_unit = [f for f in recovered_all if poly_degree(f) >= 1]

    f_rcf = rational_canonical_form_from_invariant_factors(recovered_non_unit)
    report = run_checks(a, expected_chain, recovered_all, f_rcf)

    print("Rational Canonical Form demo (exact arithmetic over Q)")
    print(f"dimension={report.dimension}")
    print(f"recovered_invariant_factors={report.recovered_invariant_factors}")
    print(f"canonical_invariant_factors={report.canonical_invariant_factors}")
    print(f"divisibility_chain_ok={report.divisibility_chain_ok}")
    print(f"same_invariants_after_canonicalization={report.same_invariants_after_canonicalization}")
    print(f"same_characteristic_polynomial={report.same_characteristic_polynomial}")
    print(f"nontrivial_similarity_example={report.nontrivial_similarity_example}")
    print("canonical_form_matrix=")
    for row in f_rcf:
        print("  ", [str(v) for v in row])
    print("All checks passed.")


if __name__ == "__main__":
    main()

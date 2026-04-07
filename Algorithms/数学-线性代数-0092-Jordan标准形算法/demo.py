"""Minimal runnable MVP for Jordan canonical form (Jordan标准形).

Algorithm path in this MVP:
1) Use exact rational arithmetic (`fractions.Fraction`) to avoid floating drift.
2) Compute exact characteristic polynomial via Faddeev-LeVerrier.
3) Factor polynomial with rational-root theorem (demo matrix is chosen to split over Q).
4) For each eigenvalue λ, use nullity growth of (A-λI)^k to recover Jordan block sizes.
5) Build Jordan matrix J from recovered blocks and run algebraic consistency checks.

No interactive input is required.
"""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
from math import gcd
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

MatrixQ = List[List[Fraction]]


def frac(x: int | Fraction) -> Fraction:
    return x if isinstance(x, Fraction) else Fraction(x)


def lcm(a: int, b: int) -> int:
    return abs(a * b) // gcd(a, b)


def eye_q(n: int) -> MatrixQ:
    return [[Fraction(1 if i == j else 0) for j in range(n)] for i in range(n)]


def zeros_q(m: int, n: int) -> MatrixQ:
    return [[Fraction(0) for _ in range(n)] for _ in range(m)]


def mat_copy(a: MatrixQ) -> MatrixQ:
    return [row[:] for row in a]


def mat_add_q(a: MatrixQ, b: MatrixQ) -> MatrixQ:
    m = len(a)
    n = len(a[0])
    out = zeros_q(m, n)
    for i in range(m):
        for j in range(n):
            out[i][j] = a[i][j] + b[i][j]
    return out


def mat_sub_q(a: MatrixQ, b: MatrixQ) -> MatrixQ:
    m = len(a)
    n = len(a[0])
    out = zeros_q(m, n)
    for i in range(m):
        for j in range(n):
            out[i][j] = a[i][j] - b[i][j]
    return out


def mat_scale_q(a: MatrixQ, s: Fraction) -> MatrixQ:
    m = len(a)
    n = len(a[0])
    out = zeros_q(m, n)
    for i in range(m):
        for j in range(n):
            out[i][j] = a[i][j] * s
    return out


def mat_mul_q(a: MatrixQ, b: MatrixQ) -> MatrixQ:
    m = len(a)
    n = len(a[0])
    if n != len(b):
        raise ValueError("Matrix dimensions mismatch for multiplication.")
    p = len(b[0])
    out = zeros_q(m, p)
    for i in range(m):
        for k in range(n):
            aik = a[i][k]
            if aik == 0:
                continue
            for j in range(p):
                out[i][j] += aik * b[k][j]
    return out


def mat_pow_q(a: MatrixQ, k: int) -> MatrixQ:
    if k < 0:
        raise ValueError("Power must be non-negative.")
    n = len(a)
    if any(len(row) != n for row in a):
        raise ValueError("Matrix power requires square matrix.")

    out = eye_q(n)
    base = mat_copy(a)
    exp = k
    while exp > 0:
        if exp & 1:
            out = mat_mul_q(out, base)
        base = mat_mul_q(base, base)
        exp >>= 1
    return out


def mat_inv_q(a: MatrixQ) -> MatrixQ:
    n = len(a)
    if any(len(row) != n for row in a):
        raise ValueError("Matrix inverse requires square matrix.")

    aug = [row[:] + eye_row[:] for row, eye_row in zip(mat_copy(a), eye_q(n))]

    pivot_row = 0
    for col in range(n):
        pivot = None
        for r in range(pivot_row, n):
            if aug[r][col] != 0:
                pivot = r
                break
        if pivot is None:
            continue

        if pivot != pivot_row:
            aug[pivot_row], aug[pivot] = aug[pivot], aug[pivot_row]

        pv = aug[pivot_row][col]
        for j in range(2 * n):
            aug[pivot_row][j] /= pv

        for r in range(n):
            if r == pivot_row:
                continue
            factor = aug[r][col]
            if factor == 0:
                continue
            for j in range(2 * n):
                aug[r][j] -= factor * aug[pivot_row][j]

        pivot_row += 1
        if pivot_row == n:
            break

    if pivot_row != n:
        raise ValueError("Matrix is singular over Q.")

    return [row[n:] for row in aug]


def mat_rank_q(a: MatrixQ) -> int:
    m = len(a)
    n = len(a[0])
    rref = mat_copy(a)

    pivot_row = 0
    for col in range(n):
        pivot = None
        for r in range(pivot_row, m):
            if rref[r][col] != 0:
                pivot = r
                break
        if pivot is None:
            continue

        if pivot != pivot_row:
            rref[pivot_row], rref[pivot] = rref[pivot], rref[pivot_row]

        pv = rref[pivot_row][col]
        for j in range(col, n):
            rref[pivot_row][j] /= pv

        for r in range(m):
            if r == pivot_row:
                continue
            factor = rref[r][col]
            if factor == 0:
                continue
            for j in range(col, n):
                rref[r][j] -= factor * rref[pivot_row][j]

        pivot_row += 1
        if pivot_row == m:
            break

    return pivot_row


def nullity_q(a: MatrixQ) -> int:
    n = len(a[0])
    return n - mat_rank_q(a)


def trace_q(a: MatrixQ) -> Fraction:
    return sum(a[i][i] for i in range(len(a)))


def characteristic_polynomial_coeffs_q(a: MatrixQ) -> List[Fraction]:
    """Return [1, c1, ..., cn] for x^n + c1*x^(n-1) + ... + cn."""
    n = len(a)
    if any(len(row) != n for row in a):
        raise ValueError("Characteristic polynomial requires square matrix.")

    b_prev = eye_q(n)
    coeffs = [Fraction(1)]

    for k in range(1, n + 1):
        ab = mat_mul_q(a, b_prev)
        c_k = -trace_q(ab) / Fraction(k)
        coeffs.append(c_k)
        b_prev = mat_add_q(ab, mat_scale_q(eye_q(n), c_k))

    return coeffs


def poly_eval_high(coeffs_high: Sequence[Fraction], x: Fraction) -> Fraction:
    val = Fraction(0)
    for c in coeffs_high:
        val = val * x + c
    return val


def poly_div_linear_high(coeffs_high: Sequence[Fraction], root: Fraction) -> Tuple[List[Fraction], Fraction]:
    if len(coeffs_high) < 2:
        raise ValueError("Polynomial degree must be at least 1 for linear division.")

    acc = [coeffs_high[0]]
    for c in coeffs_high[1:]:
        acc.append(c + acc[-1] * root)
    remainder = acc[-1]
    quotient = acc[:-1]
    return quotient, remainder


def divisors_positive(n: int) -> List[int]:
    n = abs(n)
    if n == 0:
        return [1]
    out: List[int] = []
    d = 1
    while d * d <= n:
        if n % d == 0:
            out.append(d)
            if d * d != n:
                out.append(n // d)
        d += 1
    return sorted(out)


def normalize_integer_coeffs(coeffs_high: Sequence[Fraction]) -> List[int]:
    den_lcm = 1
    for c in coeffs_high:
        den_lcm = lcm(den_lcm, c.denominator)

    ints = [int(c * den_lcm) for c in coeffs_high]
    g = 0
    for v in ints:
        g = gcd(g, abs(v))
    g = max(g, 1)
    ints = [v // g for v in ints]

    if ints[0] < 0:
        ints = [-v for v in ints]

    return ints


def factor_over_rationals_from_charpoly(coeffs_high: Sequence[Fraction]) -> List[Fraction]:
    """Factor by rational-root theorem; returns roots with multiplicity.

    MVP scope: expects complete split over Q for demo matrix.
    """
    remaining = list(coeffs_high)
    roots: List[Fraction] = []

    while len(remaining) > 1:
        ints = normalize_integer_coeffs(remaining)
        lead = abs(ints[0])
        const = ints[-1]

        p_candidates = divisors_positive(const) if const != 0 else [0]
        q_candidates = divisors_positive(lead)

        candidates: set[Fraction] = set()
        if const == 0:
            candidates.add(Fraction(0))
        for p in p_candidates:
            for q in q_candidates:
                candidates.add(Fraction(p, q))
                candidates.add(Fraction(-p, q))

        found_root = None
        for r in sorted(candidates, key=float):
            if poly_eval_high(remaining, r) == 0:
                found_root = r
                break

        if found_root is None:
            raise ValueError(
                "Characteristic polynomial does not fully split over Q in this MVP setup."
            )

        quotient, rem = poly_div_linear_high(remaining, found_root)
        if rem != 0:
            raise ValueError("Internal linear division failure during factoring.")

        roots.append(found_root)
        remaining = quotient

    return roots


def jordan_block(lam: Fraction, size: int) -> MatrixQ:
    out = zeros_q(size, size)
    for i in range(size):
        out[i][i] = lam
        if i + 1 < size:
            out[i][i + 1] = Fraction(1)
    return out


def block_diag_q(blocks: Sequence[MatrixQ]) -> MatrixQ:
    n = sum(len(b) for b in blocks)
    out = zeros_q(n, n)

    offset = 0
    for b in blocks:
        m = len(b)
        for i in range(m):
            for j in range(m):
                out[offset + i][offset + j] = b[i][j]
        offset += m

    return out


def format_fraction(x: Fraction) -> str:
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


def format_poly_high(coeffs_high: Sequence[Fraction]) -> str:
    n = len(coeffs_high) - 1
    terms: List[str] = []

    for i, c in enumerate(coeffs_high):
        if c == 0:
            continue
        deg = n - i
        sign = "-" if c < 0 else "+"
        abs_c = -c if c < 0 else c

        if deg == 0:
            core = format_fraction(abs_c)
        elif deg == 1:
            core = "x" if abs_c == 1 else f"{format_fraction(abs_c)}*x"
        else:
            core = f"x^{deg}" if abs_c == 1 else f"{format_fraction(abs_c)}*x^{deg}"

        if not terms:
            terms.append(core if c > 0 else f"-{core}")
        else:
            terms.append(f" {sign} {core}")

    return "".join(terms) if terms else "0"


def matrix_to_pretty_lines(a: MatrixQ) -> List[str]:
    rows = []
    for row in a:
        rows.append("[" + ", ".join(f"{format_fraction(v):>4}" for v in row) + "]")
    return rows


def jordan_block_sizes_from_nullity_profile(a: MatrixQ, lam: Fraction, algebraic_mult: int) -> Tuple[List[int], List[int]]:
    n = len(a)
    shifted = mat_sub_q(a, mat_scale_q(eye_q(n), lam))

    nullities = [0]
    for k in range(1, algebraic_mult + 1):
        nk = mat_pow_q(shifted, k)
        nullities.append(nullity_q(nk))

    at_least_k = [nullities[k] - nullities[k - 1] for k in range(1, algebraic_mult + 1)]

    exact_sizes: List[int] = []
    for k in range(1, algebraic_mult + 1):
        next_count = at_least_k[k] if k < algebraic_mult else 0
        exact_count = at_least_k[k - 1] - next_count
        exact_sizes.extend([k] * exact_count)

    exact_sizes.sort(reverse=True)
    return exact_sizes, nullities[1:]


def recover_jordan_form(a: MatrixQ) -> Tuple[MatrixQ, Dict[Fraction, List[int]], List[Fraction], Dict[Fraction, int]]:
    char_poly = characteristic_polynomial_coeffs_q(a)
    roots = factor_over_rationals_from_charpoly(char_poly)
    multiplicities = dict(sorted(Counter(roots).items(), key=lambda item: float(item[0])))

    blocks: Dict[Fraction, List[int]] = {}
    block_mats: List[MatrixQ] = []

    for lam, mult in multiplicities.items():
        sizes, _ = jordan_block_sizes_from_nullity_profile(a, lam, mult)
        if sum(sizes) != mult:
            raise ValueError(f"Recovered block sizes mismatch for eigenvalue {lam}.")
        blocks[lam] = sizes
        for s in sizes:
            block_mats.append(jordan_block(lam, s))

    j = block_diag_q(block_mats)
    return j, blocks, char_poly, multiplicities


def build_demo_input() -> Tuple[MatrixQ, MatrixQ, Dict[Fraction, List[int]]]:
    """Construct a deterministic nontrivial similarity example A = P J P^{-1}."""
    expected_blocks: Dict[Fraction, List[int]] = {
        Fraction(-1): [2],
        Fraction(2): [3, 1],
    }

    j_true = block_diag_q(
        [
            jordan_block(Fraction(-1), 2),
            jordan_block(Fraction(2), 3),
            jordan_block(Fraction(2), 1),
        ]
    )

    p = eye_q(6)
    p[0][1] = Fraction(1)
    p[0][3] = Fraction(-2)
    p[1][2] = Fraction(1)
    p[1][5] = Fraction(2)
    p[2][4] = Fraction(2)
    p[3][4] = Fraction(-1)
    p[3][5] = Fraction(1)
    p[4][5] = Fraction(-1)

    p_inv = mat_inv_q(p)
    a = mat_mul_q(mat_mul_q(p, j_true), p_inv)
    return a, j_true, expected_blocks


def run_checks(
    a: MatrixQ,
    j_est: MatrixQ,
    j_true: MatrixQ,
    recovered_blocks: Dict[Fraction, List[int]],
    expected_blocks: Dict[Fraction, List[int]],
    multiplicities: Dict[Fraction, int],
) -> None:
    if recovered_blocks != expected_blocks:
        raise AssertionError(f"Block structure mismatch. expected={expected_blocks}, got={recovered_blocks}")

    char_a = characteristic_polynomial_coeffs_q(a)
    char_j = characteristic_polynomial_coeffs_q(j_est)
    if char_a != char_j:
        raise AssertionError("Characteristic polynomials of A and recovered J do not match.")

    for lam, mult in multiplicities.items():
        _, profile_a = jordan_block_sizes_from_nullity_profile(a, lam, mult)
        _, profile_j = jordan_block_sizes_from_nullity_profile(j_est, lam, mult)
        if profile_a != profile_j:
            raise AssertionError(
                f"Nullity profile mismatch for eigenvalue {lam}: A={profile_a}, J={profile_j}"
            )

    norm_gap = np.linalg.norm(to_numpy_float(a) - to_numpy_float(j_true), ord="fro")
    if norm_gap < 1e-9:
        raise AssertionError("Demo input is trivial (A == J_true); expected nontrivial similarity transform.")


def to_numpy_float(a: MatrixQ) -> np.ndarray:
    return np.array([[float(v) for v in row] for row in a], dtype=float)


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)

    a, j_true, expected_blocks = build_demo_input()
    j_est, recovered_blocks, char_poly, multiplicities = recover_jordan_form(a)

    run_checks(a, j_est, j_true, recovered_blocks, expected_blocks, multiplicities)

    print("=== Jordan Canonical Form MVP ===")
    print("Input matrix A (exact rational display):")
    for line in matrix_to_pretty_lines(a):
        print(line)

    print("\nCharacteristic polynomial p_A(x):")
    print(format_poly_high(char_poly))

    print("\nRecovered eigenvalue multiplicities:")
    print({format_fraction(k): v for k, v in multiplicities.items()})

    print("\nRecovered Jordan block sizes per eigenvalue:")
    print({format_fraction(k): v for k, v in recovered_blocks.items()})

    print("\nRecovered Jordan matrix J:")
    for line in matrix_to_pretty_lines(j_est):
        print(line)

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()

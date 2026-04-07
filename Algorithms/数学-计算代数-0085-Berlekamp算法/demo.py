"""Berlekamp algorithm MVP (finite-field polynomial factorization).

This script is self-contained and runnable with:
    python3 demo.py

Scope of this MVP:
- Field: GF(p), where p is a prime.
- Input polynomial must be square-free (checked in code).
- Uses only Python standard library.
"""

from __future__ import annotations

import random
from typing import Iterable, List, Sequence, Tuple

Poly = List[int]  # coefficients in ascending order: c0 + c1*x + ...


def is_prime_number(n: int) -> bool:
    """Return True if n is a prime integer (small deterministic test)."""
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def normalize(poly: Sequence[int], p: int) -> Poly:
    """Normalize polynomial into GF(p) with trimmed trailing zeros."""
    out = [c % p for c in poly]
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    return out


def poly_degree(poly: Sequence[int]) -> int:
    return len(poly) - 1


def poly_is_zero(poly: Sequence[int]) -> bool:
    return len(poly) == 1 and poly[0] == 0


def poly_add(a: Sequence[int], b: Sequence[int], p: int) -> Poly:
    n = max(len(a), len(b))
    out = [0] * n
    for i in range(n):
        av = a[i] if i < len(a) else 0
        bv = b[i] if i < len(b) else 0
        out[i] = (av + bv) % p
    return normalize(out, p)


def poly_sub(a: Sequence[int], b: Sequence[int], p: int) -> Poly:
    n = max(len(a), len(b))
    out = [0] * n
    for i in range(n):
        av = a[i] if i < len(a) else 0
        bv = b[i] if i < len(b) else 0
        out[i] = (av - bv) % p
    return normalize(out, p)


def poly_scalar_mul(a: Sequence[int], k: int, p: int) -> Poly:
    return normalize([(k * c) % p for c in a], p)


def poly_mul(a: Sequence[int], b: Sequence[int], p: int) -> Poly:
    if poly_is_zero(a) or poly_is_zero(b):
        return [0]
    out = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        if ai == 0:
            continue
        for j, bj in enumerate(b):
            if bj == 0:
                continue
            out[i + j] = (out[i + j] + ai * bj) % p
    return normalize(out, p)


def poly_divmod(a: Sequence[int], b: Sequence[int], p: int) -> Tuple[Poly, Poly]:
    """Return quotient and remainder of a(x)/b(x) in GF(p)[x]."""
    dividend = normalize(a, p)
    divisor = normalize(b, p)
    if poly_is_zero(divisor):
        raise ZeroDivisionError("polynomial division by zero")

    if poly_degree(dividend) < poly_degree(divisor):
        return [0], dividend

    q = [0] * (poly_degree(dividend) - poly_degree(divisor) + 1)
    r = dividend[:]
    inv_lead = pow(divisor[-1], p - 2, p)

    while poly_degree(r) >= poly_degree(divisor) and not poly_is_zero(r):
        shift = poly_degree(r) - poly_degree(divisor)
        coef = (r[-1] * inv_lead) % p
        q[shift] = coef

        subtractor = [0] * shift + poly_scalar_mul(divisor, coef, p)
        r = poly_sub(r, subtractor, p)

    return normalize(q, p), normalize(r, p)


def poly_mod(a: Sequence[int], modulus: Sequence[int], p: int) -> Poly:
    _, r = poly_divmod(a, modulus, p)
    return r


def poly_gcd(a: Sequence[int], b: Sequence[int], p: int) -> Poly:
    x = normalize(a, p)
    y = normalize(b, p)
    while not poly_is_zero(y):
        _, r = poly_divmod(x, y, p)
        x, y = y, r
    if poly_is_zero(x):
        return [0]
    lead_inv = pow(x[-1], p - 2, p)
    return poly_scalar_mul(x, lead_inv, p)


def poly_derivative(a: Sequence[int], p: int) -> Poly:
    if len(a) <= 1:
        return [0]
    out = []
    for i in range(1, len(a)):
        out.append((i * a[i]) % p)
    return normalize(out, p)


def poly_powmod(base: Sequence[int], exp: int, modulus: Sequence[int], p: int) -> Poly:
    result = [1]
    x = normalize(base, p)
    m = normalize(modulus, p)
    e = exp
    while e > 0:
        if e & 1:
            result = poly_mod(poly_mul(result, x, p), m, p)
        x = poly_mod(poly_mul(x, x, p), m, p)
        e >>= 1
    return normalize(result, p)


def poly_monic(a: Sequence[int], p: int) -> Poly:
    x = normalize(a, p)
    if poly_is_zero(x):
        return [0]
    inv = pow(x[-1], p - 2, p)
    return poly_scalar_mul(x, inv, p)


def poly_to_str(a: Sequence[int], var: str = "x") -> str:
    poly = a[:]
    terms: List[str] = []
    for i in range(len(poly) - 1, -1, -1):
        c = poly[i]
        if c == 0:
            continue
        if i == 0:
            terms.append(str(c))
        elif i == 1:
            terms.append(var if c == 1 else f"{c}{var}")
        else:
            terms.append(f"{var}^{i}" if c == 1 else f"{c}{var}^{i}")
    return " + ".join(terms) if terms else "0"


def berlekamp_matrix(f: Sequence[int], p: int) -> List[List[int]]:
    """Build Q - I, where Q_{row,col} are coefficients of x^(p*col) mod f."""
    n = poly_degree(f)
    q = [[0] * n for _ in range(n)]
    x = [0, 1]
    for col in range(n):
        xp = poly_powmod(x, p * col, f, p)
        for row in range(min(n, len(xp))):
            q[row][col] = xp[row] % p

    for i in range(n):
        q[i][i] = (q[i][i] - 1) % p
    return q


def nullspace_mod_p(matrix: Sequence[Sequence[int]], p: int) -> List[List[int]]:
    """Return a basis of null space for matrix over GF(p)."""
    if not matrix:
        return [[1]]

    rows = len(matrix)
    cols = len(matrix[0])
    m = [list(row) for row in matrix]

    pivot_cols: List[int] = []
    pivot_row = 0

    for col in range(cols):
        sel = None
        for r in range(pivot_row, rows):
            if m[r][col] % p != 0:
                sel = r
                break
        if sel is None:
            continue

        m[pivot_row], m[sel] = m[sel], m[pivot_row]

        inv = pow(m[pivot_row][col] % p, p - 2, p)
        for j in range(col, cols):
            m[pivot_row][j] = (m[pivot_row][j] * inv) % p

        for r in range(rows):
            if r == pivot_row:
                continue
            factor = m[r][col] % p
            if factor == 0:
                continue
            for j in range(col, cols):
                m[r][j] = (m[r][j] - factor * m[pivot_row][j]) % p

        pivot_cols.append(col)
        pivot_row += 1
        if pivot_row == rows:
            break

    free_cols = [c for c in range(cols) if c not in set(pivot_cols)]
    basis: List[List[int]] = []

    for free_col in free_cols:
        vec = [0] * cols
        vec[free_col] = 1
        for r in range(len(pivot_cols) - 1, -1, -1):
            c = pivot_cols[r]
            acc = 0
            for j in range(c + 1, cols):
                acc = (acc + m[r][j] * vec[j]) % p
            vec[c] = (-acc) % p
        basis.append(vec)

    # In Berlekamp contexts for square-free f, nullity is at least 1.
    # If numerical edge-case occurs, return the zero polynomial basis fallback.
    if not basis:
        basis.append([0] * cols)

    return basis


def vector_to_poly(v: Sequence[int], p: int) -> Poly:
    return normalize(v, p)


def berlekamp_basis(f: Sequence[int], p: int) -> List[Poly]:
    mat = berlekamp_matrix(f, p)
    vectors = nullspace_mod_p(mat, p)
    return [vector_to_poly(v, p) for v in vectors]


def split_factor(
    h: Sequence[int],
    basis: Sequence[Sequence[int]],
    p: int,
    rng: random.Random,
) -> Tuple[Poly, Poly] | None:
    """Try to split h using Berlekamp subspace vectors."""
    n = poly_degree(h)

    candidates: List[Poly] = []
    non_constant_basis = [normalize(v, p) for v in basis if any(v[1:])]
    candidates.extend(non_constant_basis)

    # Add simple pairwise sums to enrich deterministic candidates.
    for i in range(len(non_constant_basis)):
        for j in range(i + 1, len(non_constant_basis)):
            candidates.append(poly_add(non_constant_basis[i], non_constant_basis[j], p))

    # Add a few random linear combinations for robustness.
    for _ in range(20):
        coeffs = [rng.randrange(p) for _ in non_constant_basis]
        if all(c == 0 for c in coeffs):
            continue
        g = [0]
        for c, v in zip(coeffs, non_constant_basis):
            if c == 0:
                continue
            g = poly_add(g, poly_scalar_mul(v, c, p), p)
        candidates.append(g)

    for g in candidates:
        g = normalize(g[:n], p)
        for a in range(p):
            d = poly_gcd(h, poly_sub(g, [a], p), p)
            dd = poly_degree(d)
            if 0 < dd < poly_degree(h):
                q, r = poly_divmod(h, d, p)
                if not poly_is_zero(r):
                    continue
                return poly_monic(d, p), poly_monic(q, p)

    return None


def berlekamp_factor_square_free(f: Sequence[int], p: int) -> List[Poly]:
    """Factor a monic square-free polynomial over GF(p)."""
    todo = [poly_monic(f, p)]
    done: List[Poly] = []
    rng = random.Random(0)

    while todo:
        h = poly_monic(todo.pop(), p)
        if poly_degree(h) <= 1:
            done.append(h)
            continue

        basis = berlekamp_basis(h, p)
        if len(basis) <= 1:
            done.append(h)
            continue

        split = split_factor(h, basis, p, rng)
        if split is None:
            # Conservative fallback: keep as not-split in this MVP.
            done.append(h)
            continue

        a, b = split
        todo.append(a)
        todo.append(b)

    done.sort(key=lambda poly: (poly_degree(poly), poly))
    return done


def berlekamp_factor(f: Sequence[int], p: int) -> List[Poly]:
    """Public entry: factor square-free polynomial over GF(p)."""
    if not is_prime_number(p):
        raise ValueError(f"p must be prime for GF(p), got {p}")

    poly = normalize(f, p)
    if poly_is_zero(poly):
        raise ValueError("zero polynomial has undefined factorization")

    poly = poly_monic(poly, p)
    g = poly_gcd(poly, poly_derivative(poly, p), p)
    if poly_degree(g) > 0:
        raise ValueError(
            "This MVP expects a square-free polynomial. "
            "Please run square-free decomposition first."
        )

    return berlekamp_factor_square_free(poly, p)


def multiply_all(polys: Iterable[Sequence[int]], p: int) -> Poly:
    acc = [1]
    for poly in polys:
        acc = poly_mul(acc, poly, p)
    return normalize(acc, p)


def run_case(title: str, p: int, f: Sequence[int]) -> None:
    print(f"\n=== {title} ===")
    print(f"Field: GF({p})")
    ff = poly_monic(normalize(f, p), p)
    print(f"Input f(x): {poly_to_str(ff)}")

    factors = berlekamp_factor(ff, p)
    print("Factors:")
    for i, fac in enumerate(factors, start=1):
        print(f"  {i}. {poly_to_str(fac)}")

    rebuilt = poly_monic(multiply_all(factors, p), p)
    assert rebuilt == ff, "factor multiplication check failed"
    print("Check: product of factors equals f(x) [OK]")


def main() -> None:
    # Case 1: GF(2), reducible degree-6 polynomial.
    # f(x) = x^6 + x^5 + x^4 + x^3 + 1
    run_case(
        title="Case A (GF(2), reducible)",
        p=2,
        f=[1, 0, 0, 1, 1, 1, 1],
    )

    # Case 2: GF(5), square-free reducible quartic.
    # f(x) = (x+1)(x+2)(x+3)(x+4) = x^4 + 4  (mod 5)
    run_case(
        title="Case B (GF(5), reducible)",
        p=5,
        f=[4, 0, 0, 0, 1],
    )

    # Case 3: GF(2), irreducible sample x^3 + x + 1.
    run_case(
        title="Case C (GF(2), irreducible)",
        p=2,
        f=[1, 1, 0, 1],
    )

    print("\nAll demo cases finished successfully.")


if __name__ == "__main__":
    main()

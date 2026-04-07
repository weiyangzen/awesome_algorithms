"""Coppersmith method MVP (univariate, monic) with an in-file LLL implementation.

This demo solves a small-root problem:
    f(x) == 0 (mod N), |x| < X
for a monic polynomial f over Z.

Notes:
- This is an educational MVP, not a production cryptanalysis toolkit.
- We intentionally avoid black-box lattice/crypto packages and implement LLL directly.
"""

from __future__ import annotations

from fractions import Fraction
from typing import List, Optional, Sequence, Set, Tuple


Poly = List[int]
Vector = List[int]


def poly_trim(poly: Poly) -> Poly:
    """Remove trailing zeros while keeping at least one coefficient."""
    if not poly:
        return [0]
    out = poly[:]
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    return out


def poly_add(a: Poly, b: Poly) -> Poly:
    n = max(len(a), len(b))
    out = [0] * n
    for i in range(n):
        av = a[i] if i < len(a) else 0
        bv = b[i] if i < len(b) else 0
        out[i] = av + bv
    return poly_trim(out)


def poly_mul(a: Poly, b: Poly) -> Poly:
    out = [0] * (len(a) + len(b) - 1)
    for i, av in enumerate(a):
        if av == 0:
            continue
        for j, bv in enumerate(b):
            if bv == 0:
                continue
            out[i + j] += av * bv
    return poly_trim(out)


def poly_pow(base: Poly, exp: int) -> Poly:
    if exp < 0:
        raise ValueError("Exponent must be non-negative.")
    result: Poly = [1]
    cur = base[:]
    e = exp
    while e > 0:
        if e & 1:
            result = poly_mul(result, cur)
        cur = poly_mul(cur, cur)
        e >>= 1
    return result


def poly_scale(poly: Poly, scalar: int) -> Poly:
    return poly_trim([c * scalar for c in poly])


def poly_shift(poly: Poly, k: int) -> Poly:
    if k < 0:
        raise ValueError("Shift must be non-negative.")
    if poly == [0]:
        return [0]
    return [0] * k + poly


def poly_eval(poly: Sequence[int], x: int) -> int:
    value = 0
    for coeff in reversed(poly):
        value = value * x + coeff
    return value


def poly_eval_mod(poly: Sequence[int], x: int, mod: int) -> int:
    value = 0
    x_mod = x % mod
    for coeff in reversed(poly):
        value = (value * x_mod + coeff) % mod
    return value


def poly_to_string(poly: Sequence[int]) -> str:
    terms: List[str] = []
    for power in range(len(poly) - 1, -1, -1):
        coeff = poly[power]
        if coeff == 0:
            continue

        sign = "-" if coeff < 0 else "+"
        abs_coeff = abs(coeff)

        if power == 0:
            core = f"{abs_coeff}"
        elif power == 1:
            core = "x" if abs_coeff == 1 else f"{abs_coeff}*x"
        else:
            core = f"x^{power}" if abs_coeff == 1 else f"{abs_coeff}*x^{power}"

        if not terms:
            terms.append(core if coeff > 0 else f"-{core}")
        else:
            terms.append(f" {sign} {core}")

    return "".join(terms) if terms else "0"


def nearest_integer(frac_value: Fraction) -> int:
    """Round a Fraction to the nearest integer (half up)."""
    n = frac_value.numerator
    d = frac_value.denominator
    if n >= 0:
        return (n + d // 2) // d
    return -((-n + d // 2) // d)


def gram_schmidt(basis: List[Vector]) -> Tuple[List[List[Fraction]], List[Fraction], List[List[Fraction]]]:
    n = len(basis)
    dim = len(basis[0]) if n else 0

    mu = [[Fraction(0) for _ in range(n)] for _ in range(n)]
    b_star = [[Fraction(0) for _ in range(dim)] for _ in range(n)]
    norms = [Fraction(0) for _ in range(n)]

    for i in range(n):
        bi = [Fraction(v) for v in basis[i]]
        vi = bi[:]

        for j in range(i):
            if norms[j] == 0:
                continue
            numer = sum(bi[k] * b_star[j][k] for k in range(dim))
            mu_ij = numer / norms[j]
            mu[i][j] = mu_ij
            for k in range(dim):
                vi[k] -= mu_ij * b_star[j][k]

        b_star[i] = vi
        norms[i] = sum(v * v for v in vi)

    return mu, norms, b_star


def lll_reduction(basis: List[Vector], delta: Fraction = Fraction(3, 4)) -> List[Vector]:
    """Classic LLL for small integer lattices (educational implementation)."""
    if not basis:
        return []

    b = [row[:] for row in basis]
    n = len(b)
    dim = len(b[0])

    if any(len(row) != dim for row in b):
        raise ValueError("Basis vectors must have equal dimension.")

    mu, norms, _ = gram_schmidt(b)
    k = 1

    while k < n:
        # Size reduction.
        for j in range(k - 1, -1, -1):
            q = nearest_integer(mu[k][j])
            if q != 0:
                for col in range(dim):
                    b[k][col] -= q * b[j][col]

        mu, norms, _ = gram_schmidt(b)

        # Lovasz condition.
        left = norms[k]
        right = (delta - mu[k][k - 1] * mu[k][k - 1]) * norms[k - 1]

        if left >= right:
            k += 1
        else:
            b[k], b[k - 1] = b[k - 1], b[k]
            mu, norms, _ = gram_schmidt(b)
            k = max(1, k - 1)

    return b


def build_coppersmith_lattice(f: Poly, n_mod: int, x_bound: int, m: int, t: int) -> List[Vector]:
    """Build the univariate Coppersmith lattice basis for a monic polynomial."""
    f = poly_trim(f)
    degree = len(f) - 1

    if degree <= 0:
        raise ValueError("Polynomial degree must be >= 1.")
    if f[-1] != 1:
        raise ValueError("This MVP requires a monic polynomial.")
    if m <= 0 or t <= 0:
        raise ValueError("Parameters m and t must be positive.")

    lattice_dim = degree * m + t

    f_pows: List[Poly] = [[1]]
    for i in range(1, m + 1):
        f_pows.append(poly_mul(f_pows[-1], f))

    polys: List[Poly] = []

    # x^j * N^(m-i) * f(x)^i, 0<=i<m, 0<=j<degree
    for i in range(m):
        base = poly_scale(f_pows[i], n_mod ** (m - i))
        for j in range(degree):
            polys.append(poly_shift(base, j))

    # x^j * f(x)^m, 0<=j<t
    fm = f_pows[m]
    for j in range(t):
        polys.append(poly_shift(fm, j))

    if len(polys) != lattice_dim:
        raise RuntimeError("Unexpected lattice polynomial count.")

    basis: List[Vector] = []
    for g in polys:
        # Coefficients of g(x * X): each degree-k coefficient is multiplied by X^k.
        scaled = [coeff * (x_bound ** k) for k, coeff in enumerate(g)]
        if len(scaled) > lattice_dim:
            raise RuntimeError("Scaled polynomial degree exceeds lattice dimension.")
        row = scaled + [0] * (lattice_dim - len(scaled))
        basis.append(row)

    return basis


def vector_to_unscaled_poly(vec: Vector, x_bound: int) -> Optional[Poly]:
    """Convert coefficients of Q(x*X) back to Q(x) by dividing X^k."""
    out: Poly = []
    for k, value in enumerate(vec):
        divisor = x_bound ** k
        if divisor == 0:
            return None
        if value % divisor != 0:
            return None
        out.append(value // divisor)
    return poly_trim(out)


def coppersmith_univariate_monic(
    f: Poly,
    n_mod: int,
    x_bound: int,
    m: int = 2,
    t: int = 1,
) -> Tuple[List[int], List[Vector], List[Vector]]:
    """Find small roots of f(x) == 0 (mod N), |x| < X.

    Returns (roots, original_basis, reduced_basis).
    """
    if x_bound <= 1:
        raise ValueError("x_bound must be > 1.")

    basis = build_coppersmith_lattice(f, n_mod, x_bound, m, t)
    reduced = lll_reduction(basis)

    roots: Set[int] = set()

    for vec in reduced:
        q = vector_to_unscaled_poly(vec, x_bound)
        if q is None or len(q) <= 1:
            continue

        # MVP root extraction by bounded scan. For didactic scale this is enough.
        for cand in range(-x_bound + 1, x_bound):
            if poly_eval(q, cand) == 0 and poly_eval_mod(f, cand, n_mod) == 0:
                roots.add(cand)

    return sorted(roots), basis, reduced


def build_demo_instance() -> Tuple[Poly, int, int, int, int, int]:
    """Construct a deterministic toy instance with one known small root."""
    p = 1_000_003
    q = 1_009_837
    n_mod = p * q

    root = 123
    helper = 4_567

    # f(x) = x^2 + (helper-root)*x + (N - helper*root)
    # Then f(root) = N, so root is a modular root mod N but not an integer root of f(x)=0.
    f: Poly = [n_mod - helper * root, helper - root, 1]

    x_bound = 512  # root=123 is within the bound, and 512 < N^(1/4) for this toy N.
    m = 2
    t = 1

    return f, n_mod, root, x_bound, m, t


def main() -> None:
    f, n_mod, expected_root, x_bound, m, t = build_demo_instance()

    print("=== Coppersmith Method MVP (Univariate, Monic) ===")
    print(f"N = {n_mod}")
    print(f"f(x) = {poly_to_string(f)}")
    print(f"Target: find x such that f(x) ≡ 0 (mod N), with |x| < {x_bound}")
    print(f"Known planted root for validation: x0 = {expected_root}")
    print(f"Parameters: m={m}, t={t}")

    roots, basis, reduced = coppersmith_univariate_monic(
        f=f,
        n_mod=n_mod,
        x_bound=x_bound,
        m=m,
        t=t,
    )

    print(f"Lattice dimension: {len(basis)} x {len(basis[0])}")
    print(f"Recovered roots: {roots}")

    if expected_root not in roots:
        raise RuntimeError(
            "MVP failed to recover the planted root. "
            "Try adjusting parameters (m, t, X) or choosing an easier toy instance."
        )

    # Show modular validation for all recovered roots.
    print("Validation (f(root) mod N):")
    for r in roots:
        print(f"  x={r:>4d} -> f(x) mod N = {poly_eval_mod(f, r, n_mod)}")

    print("Done: planted small root recovered successfully.")


if __name__ == "__main__":
    main()

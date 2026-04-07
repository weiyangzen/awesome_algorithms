"""Minimal runnable MVP for polynomial inversion (MATH-0563)."""

from __future__ import annotations

import random
from typing import List, Sequence, Tuple

try:
    import numpy as np
except ImportError:  # fallback keeps the script runnable on minimal Python envs
    np = None


MOD = 998244353


def normalize_poly(coeffs: Sequence[int], mod: int) -> List[int]:
    """Map coefficients into [0, mod)."""
    return [int(c) % mod for c in coeffs]


def poly_mul_trunc_mod(a: Sequence[int], b: Sequence[int], n: int, mod: int) -> List[int]:
    """Return (A * B) mod x^n over Z_mod.

    Uses numpy.convolve when available, otherwise a pure-Python nested-loop fallback.
    """
    if n <= 0:
        return []

    a_list = [int(x) % mod for x in a]
    b_list = [int(x) % mod for x in b]

    if not a_list or not b_list:
        return [0] * n

    if np is not None:
        conv = np.convolve(np.array(a_list, dtype=object), np.array(b_list, dtype=object))
        upto = min(n, int(conv.size))
        out = [int(conv[i]) % mod for i in range(upto)]
        if upto < n:
            out.extend([0] * (n - upto))
        return out

    out = [0] * n
    for i, ai in enumerate(a_list):
        if ai == 0 or i >= n:
            continue
        upper = min(len(b_list), n - i)
        for j in range(upper):
            out[i + j] = (out[i + j] + ai * b_list[j]) % mod
    return out


def poly_inv_newton_mod(a: Sequence[int], n: int, mod: int) -> Tuple[List[int], List[int]]:
    """Compute inverse polynomial B so that A*B ≡ 1 (mod x^n), using Newton iteration.

    Returns:
        (b, precisions), where precisions records solved truncation orders after each step.
    """
    if n <= 0:
        return [], []
    if not a:
        raise ValueError("input polynomial must be non-empty")

    a_norm = normalize_poly(a, mod)
    if a_norm[0] == 0:
        raise ValueError("constant term is not invertible modulo mod")

    b = [pow(a_norm[0], -1, mod)]
    m = 1
    precisions = [1]

    while m < n:
        m2 = min(2 * m, n)
        a_cut = a_norm[:m2]

        ab = poly_mul_trunc_mod(a_cut, b, m2, mod)
        correction = [0] * m2
        correction[0] = 2 % mod
        for i, val in enumerate(ab):
            correction[i] = (correction[i] - val) % mod

        b = poly_mul_trunc_mod(b, correction, m2, mod)
        m = m2
        precisions.append(m)

    return b[:n], precisions


def poly_inv_naive_mod(a: Sequence[int], n: int, mod: int) -> List[int]:
    """Reference O(n^2) inverse via coefficient recurrence."""
    if n <= 0:
        return []
    if not a:
        raise ValueError("input polynomial must be non-empty")

    a_norm = normalize_poly(a, mod)
    if a_norm[0] == 0:
        raise ValueError("constant term is not invertible modulo mod")

    inv_a0 = pow(a_norm[0], -1, mod)
    b = [0] * n
    b[0] = inv_a0

    for k in range(1, n):
        accum = 0
        upper = min(k, len(a_norm) - 1)
        for i in range(1, upper + 1):
            accum = (accum + a_norm[i] * b[k - i]) % mod
        b[k] = (-accum * inv_a0) % mod

    return b


def verify_inverse(a: Sequence[int], b: Sequence[int], n: int, mod: int) -> bool:
    """Check A*B == 1 (mod x^n, mod)."""
    prod = poly_mul_trunc_mod(a, b, n, mod)
    target = [1] + [0] * (n - 1)
    return prod == target


def main() -> None:
    print("Polynomial Inversion MVP (MATH-0563)")
    print(f"MOD = {MOD}")
    print("=" * 72)

    cases: List[Tuple[List[int], int, str]] = [
        ([3, 5, 2, 7], 8, "fixed-1"),
        ([1, -2, 3, -4, 5], 10, "fixed-2"),
        ([7], 6, "fixed-3"),
    ]

    rng = random.Random(20260407)
    rand_poly = [rng.randrange(0, MOD) for _ in range(6)]
    if rand_poly[0] == 0:
        rand_poly[0] = 1
    cases.append((rand_poly, 12, "random-seeded"))

    for a_raw, n, tag in cases:
        a = normalize_poly(a_raw, MOD)
        b_newton, precisions = poly_inv_newton_mod(a, n, MOD)
        b_naive = poly_inv_naive_mod(a, n, MOD)

        assert b_newton == b_naive, f"Newton vs naive mismatch on {tag}"
        assert verify_inverse(a, b_newton, n, MOD), f"inverse identity check failed on {tag}"

        print(f"case={tag:>13} | deg(A)<={len(a)-1:>2} | n={n:>2} | iterations={len(precisions)-1}")
        print(f"  precision path: {precisions}")
        print(f"  A[:min(6,n)] = {a[: min(6, n)]}")
        print(f"  B[:min(6,n)] = {b_newton[: min(6, n)]}")

    print("=" * 72)
    print("All polynomial inversion checks passed.")


if __name__ == "__main__":
    main()

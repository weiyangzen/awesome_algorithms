"""Toy Number Field Sieve (NFS) style MVP.

This script is intentionally small and runnable in a standalone way.
It demonstrates the NFS workflow skeleton:
1) polynomial-like setup,
2) smooth relation collection,
3) GF(2) linear algebra,
4) square-congruence extraction,
5) gcd split.

For reliability on a tiny example, relation collection uses a QS-like surrogate
on Q(a)=a^2-N (instead of full GNFS rational+algebraic ideal machinery).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None


@dataclass
class Relation:
    a: int
    b: int
    rational_norm: int
    algebraic_norm: int
    exponents: List[int]


def prime_sieve(limit: int) -> List[int]:
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit**0.5) + 1):
        if sieve[p]:
            step = p
            start = p * p
            sieve[start : limit + 1 : step] = [False] * (((limit - start) // step) + 1)
    return [i for i, is_prime in enumerate(sieve) if is_prime]


def build_factor_base(n: int, bound: int) -> List[int]:
    """QS-style factor base used here as a toy stand-in for NFS relation base."""
    base = [-1]
    for p in prime_sieve(bound):
        if p == 2:
            if n % 2 == 1:
                base.append(p)
            continue
        if pow(n % p, (p - 1) // 2, p) == 1:
            base.append(p)
    return base


def factor_over_base(value: int, base: Sequence[int]) -> Optional[List[int]]:
    exponents = [0] * len(base)
    cur = value
    if cur < 0:
        exponents[0] = 1
        cur = -cur
    for i, p in enumerate(base[1:], start=1):
        while cur % p == 0:
            exponents[i] += 1
            cur //= p
    if cur == 1:
        return exponents
    return None


def collect_relations(
    n: int,
    base: Sequence[int],
    required: int,
    max_scan: int = 200_000,
) -> List[Relation]:
    """Collect smooth relations from Q(a)=a^2-n with b fixed to 1."""
    relations: List[Relation] = []
    a0 = math.isqrt(n) + 1
    for offset in range(max_scan):
        a = a0 + offset
        q = a * a - n
        r = a - a0
        if r == 0:
            continue
        sq = math.isqrt(abs(q))
        if sq * sq == abs(q):
            # Skip single-relation perfect-square degeneracy; we want matrix combination.
            continue
        exp = factor_over_base(q, base)
        if exp is None:
            continue
        relations.append(
            Relation(
                a=a,
                b=1,
                rational_norm=r,
                algebraic_norm=q,
                exponents=exp,
            )
        )
        if len(relations) >= required:
            break
    return relations


def gf2_nullspace(matrix: np.ndarray) -> List[np.ndarray]:
    """Return a basis of nullspace vectors over GF(2) for matrix shape (m, n)."""
    a = (matrix.copy() & 1).astype(np.uint8)
    m, n = a.shape

    pivots: List[int] = []
    row = 0
    for col in range(n):
        pivot = None
        for r in range(row, m):
            if a[r, col]:
                pivot = r
                break
        if pivot is None:
            continue
        if pivot != row:
            a[[row, pivot]] = a[[pivot, row]]

        for r in range(m):
            if r != row and a[r, col]:
                a[r] ^= a[row]

        pivots.append(col)
        row += 1
        if row == m:
            break

    pivot_set = set(pivots)
    free_cols = [c for c in range(n) if c not in pivot_set]
    if not free_cols:
        return []

    basis: List[np.ndarray] = []
    for free_col in free_cols:
        v = np.zeros(n, dtype=np.uint8)
        v[free_col] = 1

        for r, pcol in enumerate(pivots):
            acc = 0
            for fc in free_cols:
                if a[r, fc] and v[fc]:
                    acc ^= 1
            v[pcol] = acc
        basis.append(v)
    return basis


def combine_dependency(
    n: int,
    base: Sequence[int],
    relations: Sequence[Relation],
    dep: np.ndarray,
) -> Optional[Tuple[int, int, List[int]]]:
    chosen = [i for i, bit in enumerate(dep.tolist()) if bit == 1]
    if not chosen:
        return None

    exp_sum = [0] * len(base)
    x_mod_n = 1
    for idx in chosen:
        rel = relations[idx]
        x_mod_n = (x_mod_n * rel.a) % n
        for j, e in enumerate(rel.exponents):
            exp_sum[j] += e

    if any(e % 2 != 0 for e in exp_sum):
        return None

    y = 1
    for j, p in enumerate(base[1:], start=1):
        y *= pow(p, exp_sum[j] // 2)

    g1 = math.gcd((x_mod_n - y) % n, n)
    if 1 < g1 < n:
        return g1, n // g1, chosen

    g2 = math.gcd((x_mod_n + y) % n, n)
    if 1 < g2 < n:
        return g2, n // g2, chosen

    return None


def is_probable_prime(n: int) -> bool:
    if n < 2:
        return False
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    for p in small_primes:
        if n % p == 0:
            return n == p

    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1

    # Deterministic bases for 64-bit range are enough for this MVP.
    for a in [2, 325, 9375, 28178, 450775, 9780504, 1795265022]:
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        skip = False
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                skip = True
                break
        if skip:
            continue
        return False
    return True


def pollard_rho(n: int) -> int:
    if n % 2 == 0:
        return 2
    if is_probable_prime(n):
        return n

    while True:
        c = random.randrange(1, n - 1)
        x = random.randrange(2, n - 1)
        y = x
        d = 1
        while d == 1:
            x = (x * x + c) % n
            y = (y * y + c) % n
            y = (y * y + c) % n
            d = math.gcd(abs(x - y), n)
        if d != n:
            return d


def demo_nfs_style_factor(n: int = 10403, base_bound: int = 50) -> Tuple[int, int]:
    random.seed(20260407)

    if n <= 1:
        raise ValueError("n must be > 1")
    if n % 2 == 0:
        return 2, n // 2

    base = build_factor_base(n, base_bound)
    required = len(base) + 8
    relations = collect_relations(n, base, required=required)

    print(f"[setup] N={n}")
    print(f"[setup] factor base size={len(base)} (bound={base_bound})")
    print(f"[collect] relations={len(relations)} / required={required}")

    if len(relations) < len(base) + 1:
        print("[warn] not enough smooth relations, using Pollard Rho fallback")
        p = pollard_rho(n)
        return min(p, n // p), max(p, n // p)

    if pd is not None:
        preview = pd.DataFrame(
            {
                "a": [r.a for r in relations[:8]],
                "R(a,b)": [r.rational_norm for r in relations[:8]],
                "A(a,b)": [r.algebraic_norm for r in relations[:8]],
            }
        )
        print("[collect] first relations preview:")
        print(preview.to_string(index=False))

    parity = np.array([[e & 1 for e in r.exponents] for r in relations], dtype=np.uint8)
    deps = gf2_nullspace(parity.T)
    print(f"[linalg] nullspace basis count={len(deps)}")

    for dep in deps:
        hit = combine_dependency(n, base, relations, dep)
        if hit is not None:
            p, q, chosen = hit
            print(f"[sqrt] dependency size={len(chosen)}")
            return min(p, q), max(p, q)

    print("[warn] congruence extraction failed, using Pollard Rho fallback")
    p = pollard_rho(n)
    return min(p, n // p), max(p, n // p)


def main() -> None:
    n = 10403  # 101 * 103
    p, q = demo_nfs_style_factor(n=n, base_bound=50)
    print(f"[result] factors of {n}: {p} * {q}")


if __name__ == "__main__":
    main()

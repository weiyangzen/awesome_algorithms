"""Quadratic Sieve MVP (educational, self-contained).

This script factors a composite integer n by:
1) Building a factor base,
2) Collecting B-smooth values of Q(x) = x^2 - n,
3) Solving a GF(2) linear system to find dependencies,
4) Extracting non-trivial factors via gcd.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Optional


@dataclass
class Relation:
    """One smooth relation Q(x)=x^2-n over the factor base."""

    x: int
    qx: int
    exponents: List[int]
    parity: List[int]


@dataclass
class QSResult:
    """Result and useful run statistics for the MVP."""

    n: int
    factor1: int
    factor2: int
    factor_base_bound: int
    factor_base_size: int
    relations_collected: int
    dependencies_tested: int


def sieve_primes(limit: int) -> List[int]:
    """Simple Eratosthenes sieve."""
    if limit < 2:
        return []
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    p = 2
    while p * p <= limit:
        if is_prime[p]:
            step = p
            start = p * p
            for k in range(start, limit + 1, step):
                is_prime[k] = False
        p += 1
    return [i for i, ok in enumerate(is_prime) if ok]


def small_trial_factor(n: int, limit: int = 2000) -> Optional[int]:
    """Find a small factor quickly, if present."""
    for p in sieve_primes(limit):
        if n % p == 0 and p < n:
            return p
    return None


def build_factor_base(n: int, bound: int) -> List[int]:
    """Build factor base of primes p where n is a quadratic residue mod p."""
    fb: List[int] = []
    for p in sieve_primes(bound):
        if p == 2:
            fb.append(2)
            continue
        if n % p == 0:
            # Caller should have removed trivial small factors first.
            continue
        # Legendre symbol (n|p): residue iff value is 1.
        if pow(n % p, (p - 1) // 2, p) == 1:
            fb.append(p)
    return fb


def find_smooth_relations(
    n: int,
    factor_base: List[int],
    required: int,
    max_candidates: int,
) -> List[Relation]:
    """Collect relations x where Q(x)=x^2-n is fully factorable over the factor base."""
    x0 = math.isqrt(n)
    if x0 * x0 < n:
        x0 += 1

    relations: List[Relation] = []
    fb_size = len(factor_base)

    for offset in range(max_candidates):
        x = x0 + offset
        qx = x * x - n
        if qx <= 0:
            continue

        rem = qx
        exponents = [0] * fb_size

        for i, p in enumerate(factor_base):
            while rem % p == 0:
                rem //= p
                exponents[i] += 1

        if rem == 1:
            parity = [e & 1 for e in exponents]
            relations.append(Relation(x=x, qx=qx, exponents=exponents, parity=parity))
            if len(relations) >= required:
                break

    return relations


def find_dependencies(parity_rows: List[List[int]]) -> List[List[int]]:
    """Find linear dependencies among rows over GF(2) via elimination.

    Each dependency is returned as a list of row indices whose XOR is zero.
    """
    if not parity_rows:
        return []

    row_count = len(parity_rows)
    col_count = len(parity_rows[0])

    mat = [row[:] for row in parity_rows]
    combos = [1 << i for i in range(row_count)]

    pivot_row = 0
    for col in range(col_count):
        pivot = None
        for r in range(pivot_row, row_count):
            if mat[r][col] == 1:
                pivot = r
                break

        if pivot is None:
            continue

        if pivot != pivot_row:
            mat[pivot_row], mat[pivot] = mat[pivot], mat[pivot_row]
            combos[pivot_row], combos[pivot] = combos[pivot], combos[pivot_row]

        for r in range(row_count):
            if r != pivot_row and mat[r][col] == 1:
                for c in range(col, col_count):
                    mat[r][c] ^= mat[pivot_row][c]
                combos[r] ^= combos[pivot_row]

        pivot_row += 1
        if pivot_row == row_count:
            break

    deps: List[List[int]] = []
    seen = set()
    for r in range(row_count):
        if any(mat[r]):
            continue

        mask = combos[r]
        if mask == 0:
            continue

        idxs = tuple(i for i in range(row_count) if (mask >> i) & 1)
        if len(idxs) < 2:
            continue
        if idxs in seen:
            continue
        seen.add(idxs)
        deps.append(list(idxs))

    return deps


def try_dependency(
    n: int,
    factor_base: List[int],
    relations: List[Relation],
    dependency: List[int],
) -> Optional[int]:
    """Attempt to extract a non-trivial factor from one dependency."""
    a = 1
    exp_sums = [0] * len(factor_base)

    for idx in dependency:
        rel = relations[idx]
        a = (a * rel.x) % n
        for j, exp in enumerate(rel.exponents):
            exp_sums[j] += exp

    b = 1
    for p, total_exp in zip(factor_base, exp_sums):
        half = total_exp // 2
        if half:
            b = (b * pow(p, half, n)) % n

    g = math.gcd((a - b) % n, n)
    if 1 < g < n:
        return g

    g = math.gcd((a + b) % n, n)
    if 1 < g < n:
        return g

    return None


def quadratic_sieve(n: int, initial_bound: int = 120) -> QSResult:
    """Factor n with an educational Quadratic Sieve MVP."""
    if n <= 1:
        raise ValueError("n must be > 1")

    if n % 2 == 0:
        return QSResult(n, 2, n // 2, 2, 1, 0, 0)

    root = math.isqrt(n)
    if root * root == n:
        return QSResult(n, root, root, 0, 0, 0, 0)

    sf = small_trial_factor(n)
    if sf is not None and 1 < sf < n:
        return QSResult(n, sf, n // sf, sf, 0, 0, 0)

    bound = initial_bound
    for _ in range(6):
        factor_base = build_factor_base(n, bound)
        if len(factor_base) < 8:
            bound = int(bound * 1.6) + 10
            continue

        needed = len(factor_base) + 8
        max_candidates = needed * 80

        relations = find_smooth_relations(
            n=n,
            factor_base=factor_base,
            required=needed,
            max_candidates=max_candidates,
        )

        if len(relations) < needed:
            bound = int(bound * 1.6) + 10
            continue

        deps = find_dependencies([rel.parity for rel in relations])

        tested = 0
        for dep in deps:
            tested += 1
            factor = try_dependency(n, factor_base, relations, dep)
            if factor is not None:
                f1 = factor
                f2 = n // factor
                if f1 > f2:
                    f1, f2 = f2, f1
                return QSResult(
                    n=n,
                    factor1=f1,
                    factor2=f2,
                    factor_base_bound=bound,
                    factor_base_size=len(factor_base),
                    relations_collected=len(relations),
                    dependencies_tested=tested,
                )

        bound = int(bound * 1.6) + 10

    raise RuntimeError("Quadratic sieve MVP failed; try increasing bounds/attempts.")


def main() -> None:
    # Demo target: a medium semiprime suitable for this educational MVP.
    n = 10007 * 10009

    result = quadratic_sieve(n)

    print("Quadratic Sieve MVP demo")
    print(f"n = {result.n}")
    print(f"factors = {result.factor1} * {result.factor2}")
    print(f"check = {result.factor1 * result.factor2 == result.n}")
    print(f"factor_base_bound = {result.factor_base_bound}")
    print(f"factor_base_size = {result.factor_base_size}")
    print(f"relations_collected = {result.relations_collected}")
    print(f"dependencies_tested = {result.dependencies_tested}")


if __name__ == "__main__":
    main()

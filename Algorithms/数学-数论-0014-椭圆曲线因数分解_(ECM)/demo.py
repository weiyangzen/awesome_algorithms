"""Elliptic Curve Method (ECM) factorization demo (Stage-1 only).

This is a small, fully self-contained MVP:
- no interactive input
- deterministic random seed for reproducibility
- explicit source-level implementation of point arithmetic and ECM flow
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

Point = Optional[Tuple[int, int]]  # None means point at infinity.


def is_probable_prime(n: int) -> bool:
    """Deterministic Miller-Rabin for 64-bit integers."""
    if n < 2:
        return False
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    for p in small_primes:
        if n == p:
            return True
        if n % p == 0:
            return False

    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1

    # Deterministic bases for testing 64-bit integers.
    for a in [2, 3, 5, 7, 11, 13, 17]:
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        skip_to_next_base = False
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                skip_to_next_base = True
                break
        if not skip_to_next_base:
            return False
    return True


def primes_up_to(limit: int) -> List[int]:
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            step_start = i * i
            sieve[step_start : limit + 1 : i] = [False] * (((limit - step_start) // i) + 1)
    return [i for i, is_prime in enumerate(sieve) if is_prime]


def inverse_or_factor(value: int, n: int) -> Tuple[Optional[int], Optional[int]]:
    """Return (inverse_mod_n, factor). One of them is usually None.

    - If gcd(value, n) == 1: return (inverse, None)
    - If 1 < gcd(value, n) < n: return (None, gcd)
    - If gcd(value, n) == n: return (None, None) as a non-informative failure
    """
    value %= n
    g = math.gcd(value, n)
    if 1 < g < n:
        return None, g
    if g == n:
        return None, None
    return pow(value, -1, n), None


def ec_add(P: Point, Q: Point, a: int, n: int) -> Tuple[Point, Optional[int]]:
    """Add points on y^2 = x^3 + a*x + b (mod n), returning (point, factor)."""
    if P is None:
        return Q, None
    if Q is None:
        return P, None

    x1, y1 = P
    x2, y2 = Q

    # P + (-P) = O
    if (x1 - x2) % n == 0 and (y1 + y2) % n == 0:
        return None, None

    if x1 == x2 and y1 == y2:
        numerator = (3 * x1 * x1 + a) % n
        denominator = (2 * y1) % n
    else:
        numerator = (y2 - y1) % n
        denominator = (x2 - x1) % n

    inv, factor = inverse_or_factor(denominator, n)
    if factor is not None:
        return None, factor
    if inv is None:
        return None, None

    lam = (numerator * inv) % n
    x3 = (lam * lam - x1 - x2) % n
    y3 = (lam * (x1 - x3) - y1) % n
    return (x3, y3), None


def ec_mul(k: int, P: Point, a: int, n: int) -> Tuple[Point, Optional[int]]:
    """Double-and-add scalar multiplication, returning (kP, factor)."""
    result: Point = None
    addend: Point = P
    kk = k
    while kk > 0:
        if kk & 1:
            result, factor = ec_add(result, addend, a, n)
            if factor is not None:
                return None, factor
        kk >>= 1
        if kk == 0:
            break
        addend, factor = ec_add(addend, addend, a, n)
        if factor is not None:
            return None, factor
    return result, None


def random_curve_and_point(n: int, rng: random.Random) -> Tuple[int, int, Point, Optional[int]]:
    """Build random curve and point; may immediately leak a factor via discriminant gcd."""
    while True:
        x = rng.randrange(2, n - 1)
        y = rng.randrange(2, n - 1)
        a = rng.randrange(1, n - 1)
        b = (y * y - x * x * x - a * x) % n
        disc = (4 * a * a * a + 27 * b * b) % n
        g = math.gcd(disc, n)
        if 1 < g < n:
            return a, b, None, g
        if g == 1:
            return a, b, (x, y), None


def ecm_stage1(n: int, B1: int, a: int, P: Point) -> Optional[int]:
    """Try one curve with Stage-1 bound B1."""
    current = P
    for p in primes_up_to(B1):
        pe = p
        while pe * p <= B1:
            pe *= p
        current, factor = ec_mul(pe, current, a, n)
        if factor is not None and 1 < factor < n:
            return factor
        if current is None:
            # Point at infinity on this modulus path; this curve is unlikely to help further.
            break
    return None


def ecm_find_factor(n: int, max_curves: int, B1: int, seed: int) -> Optional[int]:
    """Return one non-trivial factor or None."""
    if n % 2 == 0:
        return 2

    rng = random.Random(seed)
    for _ in range(max_curves):
        a, _b, P, immediate_factor = random_curve_and_point(n, rng)
        if immediate_factor is not None:
            return immediate_factor
        factor = ecm_stage1(n, B1, a, P)
        if factor is not None and factor not in (1, n):
            return factor
    return None


def trial_division_fallback(n: int, limit: int = 200_000) -> Optional[int]:
    """Small fallback to keep demo self-contained if ECM is unlucky."""
    if n % 2 == 0:
        return 2
    d = 3
    while d * d <= n and d <= limit:
        if n % d == 0:
            return d
        d += 2
    return None


def factorize_with_ecm(n: int) -> List[int]:
    """Recursive factorization using ECM-first strategy."""

    def _factor(x: int, out: List[int]) -> None:
        if x == 1:
            return
        if is_probable_prime(x):
            out.append(x)
            return
        if x % 2 == 0:
            out.append(2)
            _factor(x // 2, out)
            return

        schedules = [
            (24, 200),
            (36, 800),
            (64, 3000),
        ]
        factor: Optional[int] = None
        for max_curves, b1 in schedules:
            factor = ecm_find_factor(
                x,
                max_curves=max_curves,
                B1=b1,
                seed=(x ^ (b1 << 1) ^ max_curves) & 0xFFFFFFFF,
            )
            if factor is not None and factor not in (1, x):
                break

        if factor is None or factor in (1, x):
            factor = trial_division_fallback(x)
            if factor is None:
                # Keep the unresolved composite as-is rather than crashing.
                out.append(x)
                return

        _factor(factor, out)
        _factor(x // factor, out)

    result: List[int] = []
    _factor(n, result)
    result.sort()
    return result


def summarize_factors(factors: List[int]) -> Dict[int, int]:
    summary: Dict[int, int] = {}
    for f in factors:
        summary[f] = summary.get(f, 0) + 1
    return summary


def main() -> None:
    # Deterministic demo target: three medium-size primes.
    n = 10007 * 10009 * 10037
    factors = factorize_with_ecm(n)

    product = 1
    for f in factors:
        product *= f

    print("ECM demo (Stage-1 MVP)")
    print(f"N = {n}")
    print(f"factors = {factors}")
    print(f"factor_count = {summarize_factors(factors)}")
    print(f"product_check = {product == n}")


if __name__ == "__main__":
    main()

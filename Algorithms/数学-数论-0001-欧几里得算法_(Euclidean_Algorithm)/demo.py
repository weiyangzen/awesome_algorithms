"""Minimal runnable MVP for Euclidean Algorithm (MATH-0001)."""

from __future__ import annotations

import math
from typing import List, Tuple


def gcd_euclidean(a: int, b: int) -> int:
    """Return gcd(a, b) using iterative Euclidean algorithm.

    Convention used here: gcd(0, 0) = 0.
    """
    a, b = abs(a), abs(b)
    while b != 0:
        a, b = b, a % b
    return a


def gcd_trace(a: int, b: int) -> List[Tuple[int, int, int, int]]:
    """Return the division trace as (a, b, q, r) for each iteration."""
    trace: List[Tuple[int, int, int, int]] = []
    a, b = abs(a), abs(b)

    while b != 0:
        q, r = divmod(a, b)
        trace.append((a, b, q, r))
        a, b = b, r

    trace.append((a, 0, 0, 0))
    return trace


def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """Return (g, x, y) such that ax + by = g = gcd(a, b)."""
    if a == 0 and b == 0:
        return 0, 0, 0

    old_r, r = abs(a), abs(b)
    old_s, s = 1, 0
    old_t, t = 0, 1

    while r != 0:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t

    if a < 0:
        old_s = -old_s
    if b < 0:
        old_t = -old_t

    return old_r, old_s, old_t


def lcm_from_gcd(a: int, b: int) -> int:
    """Return lcm(a, b) based on gcd. Convention: if a==0 or b==0, return 0."""
    g = gcd_euclidean(a, b)
    if g == 0:
        return 0
    return abs((a // g) * b)


def main() -> None:
    cases = [
        (252, 198),
        (270, 192),
        (13, 17),
        (0, 5),
        (-24, 60),
        (0, 0),
        (123456789101112, 1314151617181920),
    ]

    print("Euclidean Algorithm MVP (MATH-0001)")
    print("=" * 64)

    for a, b in cases:
        g = gcd_euclidean(a, b)
        g_std = math.gcd(a, b)
        l = lcm_from_gcd(a, b)
        g_ext, x, y = extended_gcd(a, b)

        assert g == g_std, f"gcd mismatch for ({a}, {b}): {g} != {g_std}"
        assert g_ext == g, f"extended_gcd mismatch for ({a}, {b}): {g_ext} != {g}"
        if not (a == 0 and b == 0):
            assert a * x + b * y == g, "Bezout identity check failed"

        print(f"a={a:>16}, b={b:>16} | gcd={g:>8}, lcm={l:>16}")

    print("=" * 64)
    print("Trace for gcd(252, 198):")
    for step, (a, b, q, r) in enumerate(gcd_trace(252, 198), start=1):
        if b == 0:
            print(f"Step {step}: stop, gcd = {a}")
        else:
            print(f"Step {step}: {a} = {b} * {q} + {r}")

    print("=" * 64)
    print("All checks passed.")


if __name__ == "__main__":
    main()

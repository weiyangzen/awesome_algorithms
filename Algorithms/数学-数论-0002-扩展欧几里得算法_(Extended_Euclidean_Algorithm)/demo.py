"""Minimal runnable MVP for Extended Euclidean Algorithm."""

from __future__ import annotations

import math
from typing import Optional, Tuple


def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """Return (g, x, y) such that a*x + b*y = g = gcd(a, b), with g >= 0."""
    if a == 0 and b == 0:
        raise ValueError("extended_gcd(0, 0) is undefined in this MVP.")

    old_r, r = abs(a), abs(b)
    old_s, s = 1, 0
    old_t, t = 0, 1

    while r != 0:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t

    x = old_s if a >= 0 else -old_s
    y = old_t if b >= 0 else -old_t
    g = old_r
    return g, x, y


def mod_inverse(a: int, m: int) -> int:
    """Return modular inverse of a modulo m, if it exists."""
    if m <= 1:
        raise ValueError("modulus m must be > 1")
    g, x, _ = extended_gcd(a, m)
    if g != 1:
        raise ValueError(f"inverse does not exist because gcd({a}, {m}) = {g} != 1")
    return x % m


def solve_linear_diophantine(a: int, b: int, c: int) -> Optional[Tuple[int, int, int]]:
    """Solve a*x + b*y = c. Return one solution (x, y, gcd(a,b)) or None."""
    if a == 0 and b == 0:
        return None
    g, x0, y0 = extended_gcd(a, b)
    if c % g != 0:
        return None
    k = c // g
    return x0 * k, y0 * k, g


def run_self_checks() -> None:
    cases = [(240, 46), (99, 78), (-25, 18), (0, 7), (7, 0)]
    for a, b in cases:
        g, x, y = extended_gcd(a, b)
        assert g == math.gcd(a, b)
        assert a * x + b * y == g


def main() -> None:
    run_self_checks()

    print("=== Extended GCD Demo ===")
    pairs = [(240, 46), (99, 78), (-25, 18), (0, 7), (7, 0)]
    for a, b in pairs:
        g, x, y = extended_gcd(a, b)
        print(f"a={a:>4}, b={b:>4} -> gcd={g:>2}, x={x:>4}, y={y:>4}, check={a*x + b*y}")

    print("\n=== Modular Inverse Demo ===")
    inverse_cases = [(3, 11), (10, 17), (14, 15), (6, 15)]
    for a, m in inverse_cases:
        try:
            inv = mod_inverse(a, m)
            print(f"a={a:>3}, m={m:>3} -> inv={inv:>3}, check={(a * inv) % m}")
        except ValueError as exc:
            print(f"a={a:>3}, m={m:>3} -> {exc}")

    print("\n=== Linear Diophantine Demo ===")
    a, b, c = 15, 21, 84
    solution = solve_linear_diophantine(a, b, c)
    if solution is None:
        print(f"{a}x + {b}y = {c} has no integer solution.")
    else:
        x, y, g = solution
        print(f"{a}x + {b}y = {c}, gcd={g}, one solution: x={x}, y={y}, check={a*x + b*y}")


if __name__ == "__main__":
    main()

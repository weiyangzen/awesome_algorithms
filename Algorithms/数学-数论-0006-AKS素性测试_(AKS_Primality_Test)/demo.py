"""AKS Primality Test - Minimal runnable MVP.

This script implements a compact, readable version of the deterministic
AKS primality test and runs a built-in demo set without interactive input.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List


def is_perfect_power(n: int) -> bool:
    """Return True if n = a^b for integers a > 1, b > 1."""
    if n < 4:
        return False
    max_b = n.bit_length()
    for b in range(2, max_b + 1):
        low, high = 2, int(round(n ** (1.0 / b))) + 2
        if high < 2:
            continue
        while low <= high:
            mid = (low + high) // 2
            value = pow(mid, b)
            if value == n:
                return True
            if value < n:
                low = mid + 1
            else:
                high = mid - 1
    return False


def euler_phi(x: int) -> int:
    """Euler's totient function φ(x)."""
    result = x
    n = x
    p = 2
    while p * p <= n:
        if n % p == 0:
            while n % p == 0:
                n //= p
            result -= result // p
        p += 1 if p == 2 else 2
    if n > 1:
        result -= result // n
    return result


def find_smallest_r(n: int) -> int:
    """Find the smallest r such that ord_r(n) > (log2 n)^2."""
    max_k = math.ceil((math.log2(n)) ** 2)
    r = 2
    while True:
        if math.gcd(n, r) == 1:
            residue = n % r
            cur = 1
            has_small_order = False
            for _k in range(1, max_k + 1):
                cur = (cur * residue) % r
                if cur == 1:
                    has_small_order = True
                    break
            if not has_small_order:
                return r
        r += 1


def poly_mul_mod(p: List[int], q: List[int], r: int, n: int) -> List[int]:
    """Multiply two polynomials modulo (x^r - 1, n)."""
    out = [0] * r
    for i, pi in enumerate(p):
        if pi == 0:
            continue
        for j, qj in enumerate(q):
            if qj == 0:
                continue
            out[(i + j) % r] = (out[(i + j) % r] + pi * qj) % n
    return out


def poly_pow_mod(base: List[int], exp: int, r: int, n: int) -> List[int]:
    """Fast exponentiation of polynomial modulo (x^r - 1, n)."""
    result = [0] * r
    result[0] = 1
    cur = base[:]
    e = exp
    while e > 0:
        if e & 1:
            result = poly_mul_mod(result, cur, r, n)
        cur = poly_mul_mod(cur, cur, r, n)
        e >>= 1
    return result


@dataclass
class AKSResult:
    n: int
    is_prime: bool
    reason: str


def aks_primality_test(n: int) -> AKSResult:
    """Deterministic AKS primality test (compact educational implementation)."""
    if n < 2:
        return AKSResult(n, False, "n < 2")
    if n in (2, 3):
        return AKSResult(n, True, "small prime")
    if n % 2 == 0:
        return AKSResult(n, False, "even composite")
    if is_perfect_power(n):
        return AKSResult(n, False, "perfect power")

    r = find_smallest_r(n)

    for a in range(2, r + 1):
        g = math.gcd(a, n)
        if 1 < g < n:
            return AKSResult(n, False, f"non-trivial gcd found: gcd({a}, {n})={g}")

    if n <= r:
        return AKSResult(n, True, "n <= r after AKS checks")

    limit = int(math.floor(math.sqrt(euler_phi(r)) * math.log2(n)))
    for a in range(1, limit + 1):
        base = [0] * r
        base[0] = a % n
        base[1] = 1  # x + a
        lhs = poly_pow_mod(base, n, r, n)

        rhs = [0] * r
        rhs[0] = a % n
        rhs[n % r] = (rhs[n % r] + 1) % n

        if lhs != rhs:
            return AKSResult(n, False, f"polynomial congruence fails at a={a}")

    return AKSResult(n, True, "all AKS congruence checks passed")


def main() -> None:
    demo_numbers = [
        2,
        3,
        4,
        5,
        9,
        17,
        31,
        91,
        97,
        121,
        561,  # Carmichael number, should still be composite.
    ]

    print("AKS Primality Test Demo")
    print("=" * 72)
    for n in demo_numbers:
        res = aks_primality_test(n)
        label = "PRIME" if res.is_prime else "COMPOSITE"
        print(f"n={n:>4} -> {label:<9} | {res.reason}")


if __name__ == "__main__":
    main()

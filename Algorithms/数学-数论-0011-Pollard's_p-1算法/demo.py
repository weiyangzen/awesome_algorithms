"""Pollard's p-1 (stage-1) minimal runnable MVP.

Run:
    python3 demo.py
"""

from __future__ import annotations

import math
import time
from typing import Dict, List


def sieve_primes(limit: int) -> List[int]:
    """Return all primes <= limit using a basic sieve."""
    if limit < 2:
        return []

    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False

    p = 2
    while p * p <= limit:
        if is_prime[p]:
            step_start = p * p
            is_prime[step_start : limit + 1 : p] = [False] * (((limit - step_start) // p) + 1)
        p += 1

    return [i for i, flag in enumerate(is_prime) if flag]


def largest_power_leq(prime: int, limit: int) -> int:
    """Return largest power prime^e such that prime^e <= limit."""
    value = prime
    while value * prime <= limit:
        value *= prime
    return value


def pollards_p_minus_1_stage1(n: int, bound: int, base: int = 2) -> Dict[str, int]:
    """Single stage-1 attempt. Returns details including gcd result."""
    if n <= 1:
        return {"factor": n, "g": n, "bound": bound, "base": base, "prime_count": 0}
    if n % 2 == 0:
        return {"factor": 2, "g": 2, "bound": bound, "base": base, "prime_count": 0}

    g0 = math.gcd(base, n)
    if 1 < g0 < n:
        return {"factor": g0, "g": g0, "bound": bound, "base": base, "prime_count": 0}

    x = base % n
    primes = sieve_primes(bound)
    for q in primes:
        q_power = largest_power_leq(q, bound)
        x = pow(x, q_power, n)

    g = math.gcd(x - 1, n)
    factor = g if 1 < g < n else 0
    return {
        "factor": factor,
        "g": g,
        "bound": bound,
        "base": base,
        "prime_count": len(primes),
    }


def factor_with_schedule(n: int, bounds: List[int], bases: List[int]) -> Dict[str, object]:
    """Try multiple (bound, base) combinations until a factor is found."""
    attempts: List[Dict[str, int]] = []

    for bound in bounds:
        for base in bases:
            result = pollards_p_minus_1_stage1(n=n, bound=bound, base=base)
            attempts.append(result)
            if result["factor"]:
                factor = int(result["factor"])
                return {
                    "n": n,
                    "factor": factor,
                    "cofactor": n // factor,
                    "attempts": attempts,
                }

    return {
        "n": n,
        "factor": None,
        "cofactor": None,
        "attempts": attempts,
    }


def print_case(title: str, n: int, bounds: List[int], bases: List[int]) -> None:
    print(f"\n=== {title} ===")
    print(f"n = {n}")
    print(f"bounds = {bounds}, bases = {bases}")

    t0 = time.perf_counter()
    result = factor_with_schedule(n=n, bounds=bounds, bases=bases)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    for item in result["attempts"]:
        print(
            "  attempt:",
            f"B={item['bound']}",
            f"a={item['base']}",
            f"gcd={item['g']}",
            f"primes={item['prime_count']}",
        )

    if result["factor"] is not None:
        factor = int(result["factor"])
        cofactor = int(result["cofactor"])
        print(f"  success: {n} = {factor} * {cofactor}")
    else:
        print("  fail: no non-trivial factor found under given schedule")

    print(f"  elapsed: {elapsed_ms:.3f} ms")


def main() -> None:
    # Case A: success, because 113-1 = 112 is 7-smooth.
    print_case(
        title="Case A (expected success)",
        n=11413,  # 101 * 113
        bounds=[7, 11, 13, 17],
        bases=[2],
    )

    # Case B: expected failure under this bounded schedule.
    print_case(
        title="Case B (expected failure under small B)",
        n=1019 * 1231,
        bounds=[13, 17, 23, 31],
        bases=[2, 3, 5],
    )


if __name__ == "__main__":
    main()

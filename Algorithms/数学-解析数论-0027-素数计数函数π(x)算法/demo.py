"""Prime counting function pi(x) MVP.

Implements Lehmer's prime-counting algorithm with:
- Pre-sieved prime table for fast small queries.
- Recursive phi(x, a) with memoization.
- Optional NumPy-accelerated sieve, with pure Python fallback.

Run:
    python3 demo.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isqrt
from typing import Dict, List, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore


def icbrt(n: int) -> int:
    """Integer cube root: floor(cuberoot(n))."""
    if n < 0:
        raise ValueError("n must be non-negative")
    lo, hi = 0, 1
    while hi * hi * hi <= n:
        hi <<= 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if mid * mid * mid <= n:
            lo = mid
        else:
            hi = mid
    return lo


def sieve_with_numpy(limit: int) -> Tuple[List[int], List[int]]:
    """Return (primes, pi_table) using NumPy vectorized sieve."""
    is_prime = np.ones(limit + 1, dtype=np.bool_)  # type: ignore[union-attr]
    is_prime[:2] = False
    root = isqrt(limit)
    for p in range(2, root + 1):
        if is_prime[p]:
            is_prime[p * p : limit + 1 : p] = False
    primes = np.flatnonzero(is_prime).tolist()  # type: ignore[union-attr]
    pi_table = np.cumsum(is_prime, dtype=np.int64).tolist()  # type: ignore[union-attr]
    return primes, pi_table


def sieve_pure_python(limit: int) -> Tuple[List[int], List[int]]:
    """Return (primes, pi_table) using standard Eratosthenes sieve."""
    is_prime = [True] * (limit + 1)
    if limit >= 0:
        is_prime[0] = False
    if limit >= 1:
        is_prime[1] = False
    root = isqrt(limit)
    for p in range(2, root + 1):
        if is_prime[p]:
            start = p * p
            is_prime[start : limit + 1 : p] = [False] * (((limit - start) // p) + 1)

    primes: List[int] = []
    pi_table = [0] * (limit + 1)
    count = 0
    for i, flag in enumerate(is_prime):
        if flag:
            primes.append(i)
            count += 1
        pi_table[i] = count
    return primes, pi_table


@dataclass
class LehmerPrimeCounter:
    """Prime counting pi(x) via Lehmer algorithm."""

    sieve_limit: int = 2_000_000
    primes: List[int] = field(init=False)
    pi_table: List[int] = field(init=False)
    phi_cache: Dict[Tuple[int, int], int] = field(default_factory=dict, init=False)
    pi_cache: Dict[int, int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.sieve_limit < 2:
            raise ValueError("sieve_limit must be >= 2")

        if np is not None:
            self.primes, self.pi_table = sieve_with_numpy(self.sieve_limit)
        else:
            self.primes, self.pi_table = sieve_pure_python(self.sieve_limit)

    def phi(self, x: int, a: int) -> int:
        """Count numbers <= x not divisible by first a primes."""
        key = (x, a)
        if key in self.phi_cache:
            return self.phi_cache[key]

        if a == 0:
            value = x
        elif a == 1:
            value = x - x // 2
        else:
            value = self.phi(x, a - 1) - self.phi(x // self.primes[a - 1], a - 1)

        self.phi_cache[key] = value
        return value

    def pi(self, x: int) -> int:
        """Return prime-counting function pi(x)."""
        if x < 2:
            return 0
        if x < len(self.pi_table):
            return self.pi_table[x]
        if x in self.pi_cache:
            return self.pi_cache[x]

        a = self.pi(icbrt(x))
        b = self.pi(isqrt(x))

        result = self.phi(x, a) + (a - 1)
        for i in range(a, b):
            p = self.primes[i]
            result -= self.pi(x // p) - i

        self.pi_cache[x] = result
        return result


def main() -> None:
    counter = LehmerPrimeCounter(sieve_limit=2_000_000)

    known_values = {
        10: 4,
        100: 25,
        1_000: 168,
        10_000: 1229,
        100_000: 9592,
        1_000_000: 78498,
        10_000_000: 664579,
        100_000_000: 5761455,
    }

    print("验证已知值:")
    all_ok = True
    for x, expected in known_values.items():
        got = counter.pi(x)
        ok = "OK" if got == expected else "FAIL"
        if got != expected:
            all_ok = False
        print(f"pi({x}) = {got} (expected {expected}) [{ok}]")

    big_x = 1_000_000_000
    big_val = counter.pi(big_x)
    print(f"\n扩展示例: pi({big_x}) = {big_val}")

    if all_ok and big_val == 50_847_534:
        print("\n结果校验: 全部通过")
    else:
        print("\n结果校验: 存在偏差，请检查实现")


if __name__ == "__main__":
    main()

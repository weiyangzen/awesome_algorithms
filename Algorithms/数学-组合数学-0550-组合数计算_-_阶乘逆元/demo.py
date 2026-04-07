"""MVP: compute binomial coefficients with factorial inverse under a prime modulus.

Method:
1) Precompute factorial array fac[i] = i! mod p.
2) Precompute inverse-factorial array ifac[i] = (i!)^{-1} mod p.
3) Answer each query C(n, k) mod p in O(1):
   fac[n] * ifac[k] * ifac[n-k] mod p.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence, Tuple


def _is_prime_trial_division(x: int) -> bool:
    """Return True iff x is prime (deterministic trial division)."""
    if x < 2:
        return False
    if x in (2, 3):
        return True
    if x % 2 == 0:
        return False
    d = 3
    while d * d <= x:
        if x % d == 0:
            return False
        d += 2
    return True


def _validate_mod_and_max_n(mod: int, max_n: int) -> None:
    """Validate constructor inputs for factorial-inverse combinatorics."""
    if not isinstance(mod, int) or not isinstance(max_n, int):
        raise TypeError("mod and max_n must both be integers")
    if max_n < 0:
        raise ValueError("max_n must be non-negative")
    if mod <= 2:
        raise ValueError("mod must be an odd prime greater than 2")
    if not _is_prime_trial_division(mod):
        raise ValueError("mod must be prime for Fermat inverse")
    # For factorial inverse method without Lucas theorem, keep n < mod.
    if max_n >= mod:
        raise ValueError("max_n must be strictly smaller than mod")


@dataclass
class FactorialInverseComb:
    """Precompute fac/ifac and answer C(n, k) mod prime in O(1)."""

    mod: int
    max_n: int
    fac: List[int] = field(init=False)
    ifac: List[int] = field(init=False)

    def __post_init__(self) -> None:
        _validate_mod_and_max_n(self.mod, self.max_n)
        self.fac = [1] * (self.max_n + 1)
        self.ifac = [1] * (self.max_n + 1)
        self._precompute()

    def _precompute(self) -> None:
        """Build fac and ifac arrays.

        fac[i]  = i! mod p
        ifac[i] = (i!)^{-1} mod p
        """
        for i in range(1, self.max_n + 1):
            self.fac[i] = (self.fac[i - 1] * i) % self.mod

        # Fermat: a^(p-2) mod p == a^{-1} mod p when p is prime and a != 0.
        self.ifac[self.max_n] = pow(self.fac[self.max_n], self.mod - 2, self.mod)
        for i in range(self.max_n, 0, -1):
            self.ifac[i - 1] = (self.ifac[i] * i) % self.mod

    def comb(self, n: int, k: int) -> int:
        """Return C(n, k) mod mod.

        Returns 0 for k outside [0, n].
        """
        if not isinstance(n, int) or not isinstance(k, int):
            raise TypeError("n and k must both be integers")
        if n < 0:
            raise ValueError("n must be non-negative")
        if n > self.max_n:
            raise ValueError(f"n={n} exceeds precomputed max_n={self.max_n}")
        if k < 0 or k > n:
            return 0
        if k > n - k:
            k = n - k

        return (self.fac[n] * self.ifac[k] % self.mod) * self.ifac[n - k] % self.mod


def _self_check_small(engine: FactorialInverseComb, limit_n: int) -> None:
    """Cross-check small values against math.comb and key identities."""
    if limit_n < 0:
        raise ValueError("limit_n must be non-negative")
    if limit_n > engine.max_n:
        raise ValueError("limit_n must not exceed max_n")

    for n in range(limit_n + 1):
        row_sum = 0
        for k in range(n + 1):
            value = engine.comb(n, k)
            reference = math.comb(n, k) % engine.mod
            if value != reference:
                raise AssertionError(f"value mismatch at n={n}, k={k}")
            if value != engine.comb(n, n - k):
                raise AssertionError(f"symmetry failed at n={n}, k={k}")
            row_sum = (row_sum + value) % engine.mod
        if row_sum != pow(2, n, engine.mod):
            raise AssertionError(f"row-sum identity failed at n={n}")


def _print_case_table(engine: FactorialInverseComb, cases: Sequence[Tuple[int, int]]) -> None:
    """Print deterministic demo cases."""
    print("=== Binomial Coefficient via Factorial Inverse (mod prime) ===")
    print(f"mod = {engine.mod}, precompute max_n = {engine.max_n}")
    print(f"{'n':>8} {'k':>8} {'C(n,k) mod p':>16} {'reference':>16} {'check':>8}")

    for n, k in cases:
        value = engine.comb(n, k)
        if 0 <= k <= n <= 300:
            reference = math.comb(n, k) % engine.mod
            check = "PASS" if value == reference else "FAIL"
            ref_text = str(reference)
        else:
            # For very large n, avoid building huge exact integers with math.comb.
            check = "N/A"
            ref_text = "-"

        print(f"{n:8d} {k:8d} {value:16d} {ref_text:>16} {check:>8}")


def main() -> None:
    mod = 1_000_000_007
    max_n = 200_000

    engine = FactorialInverseComb(mod=mod, max_n=max_n)
    _self_check_small(engine, limit_n=160)

    cases = [
        (5, 2),
        (10, 3),
        (100, 50),
        (1000, 123),
        (50_000, 321),
        (200_000, 100_000),
        (12, -1),
        (12, 20),
    ]
    _print_case_table(engine, cases)

    print("\nSmall-range checks (n<=160): PASS")
    print("Demo finished successfully.")


if __name__ == "__main__":
    main()

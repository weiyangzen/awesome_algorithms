"""MVP: compute binomial coefficient C(n, k) using recurrence (dynamic programming)."""

from __future__ import annotations

import math
from typing import Iterable


def _validate_n_k(n: int, k: int) -> None:
    """Validate that n and k are integers and n is non-negative."""
    if not isinstance(n, int) or not isinstance(k, int):
        raise TypeError("n and k must both be integers")
    if n < 0:
        raise ValueError("n must be non-negative")


def comb_recurrence(n: int, k: int) -> int:
    """Return C(n, k) via recurrence C(n,k)=C(n-1,k-1)+C(n-1,k).

    For k outside [0, n], this function returns 0, which is the standard
    extension used in Pascal identities.
    """
    _validate_n_k(n, k)

    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1

    # Symmetry: C(n, k) == C(n, n-k), reduces work when k > n/2.
    k = min(k, n - k)

    # dp[j] stores current-row C(i, j); update from right to left.
    dp = [0] * (k + 1)
    dp[0] = 1

    for i in range(1, n + 1):
        upper = min(i, k)
        for j in range(upper, 0, -1):
            dp[j] = dp[j] + dp[j - 1]

    return dp[k]


def _check_core_identities(max_n: int) -> None:
    """Sanity-check recurrence identities on a small range."""
    if max_n < 0:
        raise ValueError("max_n must be non-negative")

    for n in range(max_n + 1):
        row_sum = 0
        for k in range(n + 1):
            c_nk = comb_recurrence(n, k)

            # Symmetry identity.
            if c_nk != comb_recurrence(n, n - k):
                raise AssertionError(f"symmetry failed at n={n}, k={k}")

            # Reference check against stdlib exact implementation.
            if c_nk != math.comb(n, k):
                raise AssertionError(f"value mismatch at n={n}, k={k}")

            row_sum += c_nk

        # Sum of row in Pascal triangle is 2^n.
        if row_sum != (1 << n):
            raise AssertionError(f"row-sum identity failed at n={n}")


def _print_case_table(cases: Iterable[tuple[int, int]]) -> None:
    """Print deterministic demo cases and exact checks."""
    print("=== Binomial Coefficient via Recurrence ===")
    print(f"{'n':>6} {'k':>6} {'C(n,k)':>30} {'digits':>8} {'check':>8}")

    for n, k in cases:
        value = comb_recurrence(n, k)
        reference = math.comb(n, k)
        ok = "PASS" if value == reference else "FAIL"
        print(f"{n:6d} {k:6d} {value:>30d} {len(str(value)):8d} {ok:>8}")


def main() -> None:
    # 1) Deterministic correctness checks on identities.
    _check_core_identities(max_n=30)

    # 2) Fixed showcase cases (small, medium, and larger values).
    cases = [
        (5, 2),
        (10, 3),
        (20, 10),
        (52, 5),
        (100, 50),
        (200, 100),
    ]
    _print_case_table(cases)

    print("\nIdentity checks on n<=30: PASS")
    print("Demo finished successfully.")


if __name__ == "__main__":
    main()

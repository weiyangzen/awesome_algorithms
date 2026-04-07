"""Stirling numbers MVP (first kind and second kind).

Run:
    uv run python Algorithms/数学-组合数学-0552-Stirling数/demo.py
"""

from __future__ import annotations

from math import comb, factorial
from typing import List

Table = List[List[int]]


def _validate_n_max(n_max: int) -> None:
    if not isinstance(n_max, int):
        raise TypeError(f"n_max must be int, got {type(n_max).__name__}")
    if n_max < 0:
        raise ValueError("n_max must be non-negative")


def stirling_first_unsigned_table(n_max: int) -> Table:
    """Return c(n, k) for 0 <= n, k <= n_max.

    c(n, k) is the unsigned Stirling number of the first kind.
    Recurrence:
        c(0,0)=1
        c(n,k)=c(n-1,k-1)+(n-1)c(n-1,k)
    """
    _validate_n_max(n_max)
    c = [[0] * (n_max + 1) for _ in range(n_max + 1)]
    c[0][0] = 1

    for n in range(1, n_max + 1):
        for k in range(1, n + 1):
            c[n][k] = c[n - 1][k - 1] + (n - 1) * c[n - 1][k]
    return c


def stirling_first_signed_table(n_max: int) -> Table:
    """Return s(n, k) for 0 <= n, k <= n_max.

    s(n, k) is the signed Stirling number of the first kind.
    Recurrence:
        s(0,0)=1
        s(n,k)=s(n-1,k-1)-(n-1)s(n-1,k)
    """
    _validate_n_max(n_max)
    s = [[0] * (n_max + 1) for _ in range(n_max + 1)]
    s[0][0] = 1

    for n in range(1, n_max + 1):
        for k in range(1, n + 1):
            s[n][k] = s[n - 1][k - 1] - (n - 1) * s[n - 1][k]
    return s


def stirling_second_table(n_max: int) -> Table:
    """Return S(n, k) for 0 <= n, k <= n_max.

    S(n, k) is the Stirling number of the second kind.
    Recurrence:
        S(0,0)=1
        S(n,k)=S(n-1,k-1)+k*S(n-1,k)
    """
    _validate_n_max(n_max)
    s2 = [[0] * (n_max + 1) for _ in range(n_max + 1)]
    s2[0][0] = 1

    for n in range(1, n_max + 1):
        for k in range(1, n + 1):
            s2[n][k] = s2[n - 1][k - 1] + k * s2[n - 1][k]
    return s2


def stirling_second_closed_form(n: int, k: int) -> int:
    """Closed form (inclusion-exclusion):

    S(n,k) = 1/k! * sum_{j=0..k} (-1)^(k-j) * C(k,j) * j^n
    """
    if n == 0 and k == 0:
        return 1
    if n < 0 or k < 0:
        raise ValueError("n and k must be non-negative")
    if k == 0 or k > n:
        return 0

    total = 0
    for j in range(0, k + 1):
        total += ((-1) ** (k - j)) * comb(k, j) * (j**n)
    return total // factorial(k)


def falling_factorial(x: int, n: int) -> int:
    result = 1
    for t in range(n):
        result *= x - t
    return result


def bell_numbers_from_second_kind(s2: Table) -> List[int]:
    n_max = len(s2) - 1
    bells = []
    for n in range(n_max + 1):
        bells.append(sum(s2[n][k] for k in range(n + 1)))
    return bells


def _format_row(table: Table, n: int) -> str:
    vals = [table[n][k] for k in range(n + 1)]
    return f"n={n}: {vals}"


def _assert_core_properties(c1: Table, s1: Table, s2: Table) -> None:
    n_max = len(s2) - 1

    # 1) Known values for n=5.
    assert s2[5][2] == 15
    assert s2[5][3] == 25
    assert c1[5][2] == 50
    assert s1[5][2] == -50

    # 2) |s(n,k)| == c(n,k)
    for n in range(n_max + 1):
        for k in range(n + 1):
            assert abs(s1[n][k]) == c1[n][k]

    # 3) Closed-form check for second-kind values.
    for n in range(n_max + 1):
        for k in range(n + 1):
            assert s2[n][k] == stirling_second_closed_form(n, k)

    # 4) Bell numbers from row sums.
    expected_bell = [1, 1, 2, 5, 15, 52, 203, 877, 4140]
    bells = bell_numbers_from_second_kind(s2)
    assert bells[: len(expected_bell)] == expected_bell

    # 5) Basis transform checks for x=7.
    x = 7
    for n in range(n_max + 1):
        lhs_power = x**n
        rhs_power = sum(s2[n][k] * falling_factorial(x, k) for k in range(n + 1))
        assert lhs_power == rhs_power

        lhs_fall = falling_factorial(x, n)
        rhs_fall = sum(s1[n][k] * (x**k) for k in range(n + 1))
        assert lhs_fall == rhs_fall


def main() -> None:
    n_max = 8

    c1 = stirling_first_unsigned_table(n_max)
    s1 = stirling_first_signed_table(n_max)
    s2 = stirling_second_table(n_max)

    _assert_core_properties(c1, s1, s2)

    print("Stirling numbers MVP")
    print("=" * 60)
    print("Second kind S(n,k) rows (n=0..6):")
    for n in range(0, 7):
        print(_format_row(s2, n))

    print("\nFirst kind unsigned c(n,k) rows (n=0..6):")
    for n in range(0, 7):
        print(_format_row(c1, n))

    print("\nFirst kind signed s(n,k) rows (n=0..6):")
    for n in range(0, 7):
        print(_format_row(s1, n))

    bells = bell_numbers_from_second_kind(s2)
    print("\nBell numbers B_n (n=0..8):")
    print(bells)

    print("\nRepresentative values:")
    print(f"S(5,2) = {s2[5][2]}")
    print(f"c(5,2) = {c1[5][2]}")
    print(f"s(5,2) = {s1[5][2]}")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()

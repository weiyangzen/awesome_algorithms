"""Bell numbers MVP.

Run:
    uv run python Algorithms/数学-组合数学-0553-Bell数/demo.py
"""

from __future__ import annotations

from math import comb
from typing import List

Triangle = List[List[int]]


def _validate_n_max(n_max: int) -> None:
    if not isinstance(n_max, int):
        raise TypeError(f"n_max must be int, got {type(n_max).__name__}")
    if n_max < 0:
        raise ValueError("n_max must be non-negative")


def bell_triangle(n_max: int) -> Triangle:
    """Construct Bell triangle up to row n_max.

    Recurrence:
        T[0][0] = 1
        T[n][0] = T[n-1][n-1]
        T[n][k] = T[n][k-1] + T[n-1][k-1]   (1 <= k <= n)

    Bell number is the first element of each row: B_n = T[n][0].
    """
    _validate_n_max(n_max)

    triangle: Triangle = [[0] * (n + 1) for n in range(n_max + 1)]
    triangle[0][0] = 1

    for n in range(1, n_max + 1):
        triangle[n][0] = triangle[n - 1][n - 1]
        for k in range(1, n + 1):
            triangle[n][k] = triangle[n][k - 1] + triangle[n - 1][k - 1]

    return triangle


def bell_numbers_from_triangle(triangle: Triangle) -> List[int]:
    return [row[0] for row in triangle]


def stirling_second_table(n_max: int) -> List[List[int]]:
    """Return S(n,k) for 0 <= n,k <= n_max."""
    _validate_n_max(n_max)

    s2 = [[0] * (n_max + 1) for _ in range(n_max + 1)]
    s2[0][0] = 1

    for n in range(1, n_max + 1):
        for k in range(1, n + 1):
            s2[n][k] = s2[n - 1][k - 1] + k * s2[n - 1][k]

    return s2


def bell_numbers_from_stirling_second(s2: List[List[int]]) -> List[int]:
    n_max = len(s2) - 1
    bells = []
    for n in range(n_max + 1):
        bells.append(sum(s2[n][k] for k in range(n + 1)))
    return bells


def bell_numbers_via_binomial_recurrence(n_max: int) -> List[int]:
    """Use Bell recurrence B_{n+1} = sum_{k=0}^n C(n,k) B_k."""
    _validate_n_max(n_max)

    bells = [0] * (n_max + 1)
    bells[0] = 1

    for n in range(0, n_max):
        total = 0
        for k in range(0, n + 1):
            total += comb(n, k) * bells[k]
        bells[n + 1] = total

    return bells


def _assert_core_properties(triangle: Triangle, bells: List[int]) -> None:
    n_max = len(bells) - 1

    # Known Bell number prefix.
    expected_prefix = [1, 1, 2, 5, 15, 52, 203, 877, 4140, 21147, 115975]
    assert bells[: len(expected_prefix)] == expected_prefix

    # Cross-check with Stirling second kind row sums.
    s2 = stirling_second_table(n_max)
    bells_from_s2 = bell_numbers_from_stirling_second(s2)
    assert bells == bells_from_s2

    # Cross-check with binomial recurrence.
    bells_from_recurrence = bell_numbers_via_binomial_recurrence(n_max)
    assert bells == bells_from_recurrence

    # Representative values.
    assert bells[5] == 52
    assert bells[8] == 4140
    assert triangle[4][0] == 15
    assert triangle[4][4] == 52


def _format_triangle_row(triangle: Triangle, n: int) -> str:
    return f"n={n}: {triangle[n]}"


def main() -> None:
    n_max = 12

    triangle = bell_triangle(n_max)
    bells = bell_numbers_from_triangle(triangle)

    _assert_core_properties(triangle, bells)

    print("Bell Numbers MVP")
    print("=" * 60)
    print("Bell triangle rows (n=0..6):")
    for n in range(0, 7):
        print(_format_triangle_row(triangle, n))

    print("\nBell numbers B_n (n=0..12):")
    print(bells)

    print("\nRepresentative values:")
    print(f"B_5  = {bells[5]}")
    print(f"B_8  = {bells[8]}")
    print(f"B_12 = {bells[12]}")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()

"""MVP: Pell equation solver for x^2 - D*y^2 = 1."""

from __future__ import annotations

from math import isqrt
from typing import List, Tuple


Pair = Tuple[int, int]


def is_perfect_square(n: int) -> bool:
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n


def continued_fraction_sqrt_period(D: int) -> tuple[int, list[int]]:
    """Return (a0, period) for sqrt(D) where D is positive non-square."""
    if D <= 0:
        raise ValueError(f"D must be positive, got {D}")
    if is_perfect_square(D):
        raise ValueError(f"D must be non-square, got square D={D}")

    a0 = isqrt(D)
    m = 0
    d = 1
    a = a0
    period: list[int] = []

    while True:
        m = d * a - m
        d = (D - m * m) // d
        a = (a0 + m) // d
        period.append(a)
        if a == 2 * a0:
            break

    return a0, period


def convergent_from_coeffs(coeffs: list[int]) -> Pair:
    """Return numerator/denominator of finite continued fraction coefficients."""
    if not coeffs:
        raise ValueError("coeffs must not be empty")

    p_prev2, p_prev1 = 0, 1
    q_prev2, q_prev1 = 1, 0

    for a in coeffs:
        p = a * p_prev1 + p_prev2
        q = a * q_prev1 + q_prev2
        p_prev2, p_prev1 = p_prev1, p
        q_prev2, q_prev1 = q_prev1, q

    return p_prev1, q_prev1


def fundamental_solution_pell(D: int) -> Pair:
    """Compute minimal positive (x, y) solving x^2 - D*y^2 = 1."""
    a0, period = continued_fraction_sqrt_period(D)
    L = len(period)

    target_n = (L - 1) if (L % 2 == 0) else (2 * L - 1)

    coeffs = [a0]
    for i in range(1, target_n + 1):
        coeffs.append(period[(i - 1) % L])

    x, y = convergent_from_coeffs(coeffs)
    if not verify_pell(D, x, y):
        raise RuntimeError(f"Computed convergent does not satisfy Pell for D={D}")
    return x, y


def verify_pell(D: int, x: int, y: int) -> bool:
    return x * x - D * y * y == 1


def generate_solutions(D: int, x1: int, y1: int, count: int) -> list[Pair]:
    """Generate first 'count' positive solutions from the fundamental unit."""
    if count < 0:
        raise ValueError("count must be non-negative")

    solutions: list[Pair] = []
    x, y = x1, y1

    for _ in range(count):
        solutions.append((x, y))
        x, y = x1 * x + D * y1 * y, x1 * y + y1 * x

    return solutions


def brute_force_min_solution(D: int, y_limit: int = 200_000) -> Pair:
    """Slow validator: find minimal solution by scanning y upward."""
    if D <= 0 or is_perfect_square(D):
        raise ValueError("brute_force_min_solution expects positive non-square D")

    for y in range(1, y_limit + 1):
        x2 = 1 + D * y * y
        x = isqrt(x2)
        if x * x == x2:
            return x, y

    raise RuntimeError(f"No solution found up to y_limit={y_limit} for D={D}")


def verify_known_fundamentals() -> None:
    known = {
        2: (3, 2),
        3: (2, 1),
        5: (9, 4),
        6: (5, 2),
        7: (8, 3),
        13: (649, 180),
        61: (1766319049, 226153980),
    }
    for D, expected in known.items():
        got = fundamental_solution_pell(D)
        assert got == expected, f"D={D}: got={got}, expected={expected}"


def verify_by_bruteforce_small_D(max_D: int = 30) -> None:
    for D in range(2, max_D + 1):
        if is_perfect_square(D):
            continue
        got = fundamental_solution_pell(D)
        brute = brute_force_min_solution(D, y_limit=50_000)
        assert got == brute, f"D={D}: got={got}, brute={brute}"


def run_case(D: int, count: int = 5) -> None:
    a0, period = continued_fraction_sqrt_period(D)
    x1, y1 = fundamental_solution_pell(D)
    solutions = generate_solutions(D, x1, y1, count=count)

    print(f"D={D} a0={a0} period_len={len(period)} period={period}")
    print(f"  fundamental=({x1}, {y1})")

    for i, (x, y) in enumerate(solutions, start=1):
        val = x * x - D * y * y
        print(f"  n={i:<2d} x={x} y={y} eq={val}")


def main() -> None:
    verify_known_fundamentals()
    verify_by_bruteforce_small_D(30)
    print("All validations passed.")

    for D in (2, 3, 5, 13, 61):
        run_case(D, count=4)
        print()


if __name__ == "__main__":
    main()

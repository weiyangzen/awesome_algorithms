"""Minimal runnable MVP for Romberg integration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List


@dataclass
class RombergResult:
    """Container for Romberg integration outputs."""

    estimate: float
    table: List[List[float]]
    levels_used: int
    eval_count: int


def safe_eval(func: Callable[[float], float], x: float) -> float:
    """Evaluate function and ensure the return value is finite."""
    y = float(func(x))
    if not math.isfinite(y):
        raise RuntimeError(f"non-finite function value f({x})={y}")
    return y


def romberg_integral(
    func: Callable[[float], float],
    a: float,
    b: float,
    max_level: int = 7,
    tol: float = 1e-12,
) -> RombergResult:
    """Compute integral by Romberg extrapolation on top of trapezoidal refinement."""
    if not math.isfinite(a) or not math.isfinite(b):
        raise ValueError(f"integration bounds must be finite, got a={a}, b={b}")
    if max_level < 1:
        raise ValueError("max_level must be >= 1")
    if tol <= 0.0:
        raise ValueError("tol must be positive")
    if a == b:
        return RombergResult(estimate=0.0, table=[[0.0]], levels_used=0, eval_count=0)

    sign = 1.0
    left, right = a, b
    if b < a:
        sign = -1.0
        left, right = b, a

    fa = safe_eval(func, left)
    fb = safe_eval(func, right)
    eval_count = 2

    table: List[List[float]] = [[0.5 * (right - left) * (fa + fb)]]
    levels_used = 0
    prev_diag = table[0][0]

    for k in range(1, max_level + 1):
        h = (right - left) / (2**k)
        odd_sum = 0.0
        odd_count = 1 << (k - 1)

        for i in range(1, odd_count + 1):
            x_i = left + (2 * i - 1) * h
            odd_sum += safe_eval(func, x_i)
        eval_count += odd_count

        row0 = 0.5 * table[k - 1][0] + h * odd_sum
        row = [row0]

        for j in range(1, k + 1):
            correction = (row[j - 1] - table[k - 1][j - 1]) / (4**j - 1)
            row.append(row[j - 1] + correction)

        table.append(row)
        levels_used = k

        curr_diag = row[k]
        if abs(curr_diag - prev_diag) <= tol * max(1.0, abs(curr_diag)):
            break
        prev_diag = curr_diag

    if sign < 0.0:
        table = [[sign * value for value in row] for row in table]

    return RombergResult(
        estimate=table[levels_used][levels_used],
        table=table,
        levels_used=levels_used,
        eval_count=eval_count,
    )


def relative_error(estimate: float, exact: float) -> float:
    """Relative error with safe zero-reference handling."""
    if exact == 0.0:
        return abs(estimate)
    return abs(estimate - exact) / abs(exact)


def print_romberg_table(table: List[List[float]], max_rows: int = 8) -> None:
    """Pretty-print the Romberg triangular table."""
    rows = min(len(table), max_rows)
    for k in range(rows):
        row_text = "  ".join(f"{table[k][j]: .16e}" for j in range(k + 1))
        print(f"  k={k:2d}: {row_text}")
    if len(table) > max_rows:
        print(f"  ... ({len(table) - max_rows} more rows)")


def run_case(
    name: str,
    func: Callable[[float], float],
    a: float,
    b: float,
    exact: float,
    max_level: int,
    tol: float,
) -> None:
    """Execute one deterministic test case and print diagnostics."""
    result = romberg_integral(func, a, b, max_level=max_level, tol=tol)
    abs_err = abs(result.estimate - exact)
    rel_err = relative_error(result.estimate, exact)
    diagonal = [result.table[k][k] for k in range(result.levels_used + 1)]

    print("=" * 88)
    print(name)
    print(f"interval       = [{a:.12g}, {b:.12g}]")
    print(f"estimate       = {result.estimate:.16e}")
    print(f"exact          = {exact:.16e}")
    print(f"abs_error      = {abs_err:.3e}")
    print(f"rel_error      = {rel_err:.3e}")
    print(f"levels_used    = {result.levels_used}")
    print(f"eval_count     = {result.eval_count}")
    print("diagonal terms =")
    for idx, value in enumerate(diagonal):
        print(f"  R[{idx}][{idx}] = {value:.16e}")
    print("romberg table =")
    print_romberg_table(result.table)


def main() -> None:
    max_level = 7
    tol = 1e-12

    cases = [
        ("Case 1: integral of sin(x)", math.sin, 0.0, math.pi, 2.0),
        ("Case 2: integral of exp(x)", math.exp, 0.0, 1.0, math.e - 1.0),
        (
            "Case 3: integral of 1/(1+x^2)",
            lambda x: 1.0 / (1.0 + x * x),
            0.0,
            1.0,
            math.pi / 4.0,
        ),
        ("Case 4: integral of sqrt(x)", math.sqrt, 0.0, 1.0, 2.0 / 3.0),
        ("Case 5: reversed bounds for sin(x)", math.sin, math.pi, 0.0, -2.0),
    ]

    for case in cases:
        run_case(*case, max_level=max_level, tol=tol)


if __name__ == "__main__":
    main()

"""Richardson extrapolation MVP (non-interactive, runnable)."""

from __future__ import annotations

import math
from typing import Callable, List, Sequence, Tuple

import numpy as np


def richardson_table(
    raw_estimates: Sequence[float],
    order: int,
    ratio: float = 2.0,
    error_stride: int = 2,
) -> np.ndarray:
    """Build a Richardson triangular table.

    Parameters
    ----------
    raw_estimates:
        Sequence [A(h0), A(h0/r), A(h0/r^2), ...].
    order:
        Leading truncation error power p.
    ratio:
        Step-size reduction ratio r (>1).
    error_stride:
        Gap between error powers in the asymptotic expansion.
        For centered difference / trapezoidal sequence it is commonly 2.
    """
    levels = len(raw_estimates)
    table = np.full((levels, levels), np.nan, dtype=float)
    table[:, 0] = np.asarray(raw_estimates, dtype=float)

    for i in range(1, levels):
        for j in range(1, i + 1):
            exponent = order + error_stride * (j - 1)
            factor = ratio**exponent
            table[i, j] = table[i, j - 1] + (
                (table[i, j - 1] - table[i - 1, j - 1]) / (factor - 1.0)
            )
    return table


def centered_difference(f: Callable[[float], float], x: float, h: float) -> float:
    """Second-order centered finite difference for first derivative."""
    return (f(x + h) - f(x - h)) / (2.0 * h)


def trapezoidal_rule(f: Callable[[np.ndarray], np.ndarray], a: float, b: float, n: int) -> float:
    """Composite trapezoidal rule with n uniform subintervals."""
    xs = np.linspace(a, b, n + 1, dtype=float)
    ys = f(xs)
    h = (b - a) / n
    return h * (0.5 * ys[0] + ys[1:-1].sum() + 0.5 * ys[-1])


def build_raw_sequence_derivative(
    f: Callable[[float], float],
    x: float,
    h0: float,
    levels: int,
    ratio: float,
) -> List[float]:
    values: List[float] = []
    for i in range(levels):
        h = h0 / (ratio**i)
        values.append(centered_difference(f, x, h))
    return values


def build_raw_sequence_integral(
    f: Callable[[np.ndarray], np.ndarray],
    a: float,
    b: float,
    n0: int,
    levels: int,
    ratio: int,
) -> List[float]:
    values: List[float] = []
    for i in range(levels):
        n = n0 * (ratio**i)
        values.append(trapezoidal_rule(f, a, b, n))
    return values


def summarize_table(table: np.ndarray, truth: float) -> List[Tuple[int, float, float, float, float]]:
    """Return rows: (level, raw, diag, raw_err, diag_err)."""
    rows: List[Tuple[int, float, float, float, float]] = []
    for i in range(table.shape[0]):
        raw = float(table[i, 0])
        diag = float(table[i, i])
        rows.append((i, raw, diag, abs(raw - truth), abs(diag - truth)))
    return rows


def print_summary(title: str, rows: Sequence[Tuple[int, float, float, float, float]]) -> None:
    print(f"\n=== {title} ===")
    print("level | raw_estimate        | diag_estimate       | |raw-error|        | |diag-error|")
    for level, raw, diag, raw_err, diag_err in rows:
        print(
            f"{level:>5d} | {raw:>19.12f} | {diag:>19.12f} |"
            f" {raw_err:>18.10e} | {diag_err:>18.10e}"
        )


def main() -> None:
    ratio = 2.0
    levels = 6
    order = 2
    error_stride = 2

    # Experiment A: derivative of exp(x) at x=1, true value = e
    f_scalar = math.exp
    x0 = 1.0
    derivative_truth = math.e
    raw_derivative = build_raw_sequence_derivative(
        f=f_scalar, x=x0, h0=0.4, levels=levels, ratio=ratio
    )
    derivative_table = richardson_table(
        raw_estimates=raw_derivative,
        order=order,
        ratio=ratio,
        error_stride=error_stride,
    )
    derivative_rows = summarize_table(derivative_table, derivative_truth)
    print_summary("Derivative: f(x)=exp(x), f'(1)=e", derivative_rows)
    final_derivative = derivative_table[levels - 1, levels - 1]

    # Experiment B: integral of exp(x) on [0,1], true value = e - 1
    f_vector = np.exp
    integral_truth = math.e - 1.0
    raw_integral = build_raw_sequence_integral(
        f=f_vector, a=0.0, b=1.0, n0=4, levels=levels, ratio=int(ratio)
    )
    integral_table = richardson_table(
        raw_estimates=raw_integral,
        order=order,
        ratio=ratio,
        error_stride=error_stride,
    )
    integral_rows = summarize_table(integral_table, integral_truth)
    print_summary("Integral: ∫_0^1 exp(x) dx = e - 1", integral_rows)
    final_integral = integral_table[levels - 1, levels - 1]

    print("\n--- Final diagonal estimates ---")
    print(
        f"Derivative final: {final_derivative:.12f}, "
        f"abs error = {abs(final_derivative - derivative_truth):.10e}"
    )
    print(
        f"Integral final:   {final_integral:.12f}, "
        f"abs error = {abs(final_integral - integral_truth):.10e}"
    )


if __name__ == "__main__":
    main()

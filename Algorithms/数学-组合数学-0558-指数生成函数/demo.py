"""Exponential generating function MVP.

This demo computes Bell numbers B_n from the EGF
    A(x) = exp(exp(x) - 1)
using truncated power-series recursion, and validates the result against
an independent Stirling-number DP.
"""

from __future__ import annotations

from fractions import Fraction
from typing import List

import numpy as np
import pandas as pd


def factorial_table(n_max: int) -> List[int]:
    """Return [0!, 1!, ..., n_max!]."""

    if n_max < 0:
        raise ValueError("n_max must be non-negative")

    facts = [1] * (n_max + 1)
    for i in range(1, n_max + 1):
        facts[i] = facts[i - 1] * i
    return facts


def exp_series(g: List[Fraction], n_max: int) -> List[Fraction]:
    """Compute truncated f(x)=exp(g(x)) up to x^n_max.

    Args:
        g: coefficients of ordinary power series g(x)=sum g[n] x^n.
        n_max: truncation order.

    Returns:
        coefficients f[0..n_max] where f(x)=exp(g(x)).

    Notes:
        Uses coefficient recursion from f' = g' f:
            n * f[n] = sum_{k=1..n} k * g[k] * f[n-k].
        This implementation expects g[0] = 0 so f[0] = 1 is exact.
    """

    if n_max < 0:
        raise ValueError("n_max must be non-negative")
    if len(g) != n_max + 1:
        raise ValueError("g length must equal n_max + 1")
    if g[0] != 0:
        raise ValueError("this exact-rational MVP expects g[0] == 0")

    f: List[Fraction] = [Fraction(0, 1) for _ in range(n_max + 1)]
    f[0] = Fraction(1, 1)

    for n in range(1, n_max + 1):
        acc = Fraction(0, 1)
        for k in range(1, n + 1):
            acc += Fraction(k, 1) * g[k] * f[n - k]
        f[n] = acc / n

    return f


def bell_numbers_via_egf(n_max: int) -> tuple[List[int], List[Fraction], List[int]]:
    """Compute Bell numbers B_0..B_n_max using EGF A(x)=exp(exp(x)-1)."""

    facts = factorial_table(n_max)

    # g(x) = exp(x) - 1 => g[0]=0, g[n]=1/n! for n>=1
    g = [Fraction(0, 1) for _ in range(n_max + 1)]
    for n in range(1, n_max + 1):
        g[n] = Fraction(1, facts[n])

    coeff = exp_series(g, n_max)  # coeff[n] = B_n / n!

    bell: List[int] = []
    for n in range(n_max + 1):
        val = coeff[n] * facts[n]
        if val.denominator != 1:
            raise RuntimeError(f"B_{n} is not an integer: {val}")
        bell.append(val.numerator)

    return bell, coeff, facts


def bell_numbers_via_stirling(n_max: int) -> List[int]:
    """Reference implementation using Stirling numbers of the second kind."""

    if n_max < 0:
        raise ValueError("n_max must be non-negative")

    # S(n, k): partition n labeled items into k non-empty blocks.
    S = np.zeros((n_max + 1, n_max + 1), dtype=np.int64)
    S[0, 0] = 1

    for n in range(1, n_max + 1):
        for k in range(1, n + 1):
            S[n, k] = S[n - 1, k - 1] + k * S[n - 1, k]

    bell = [int(S[n, : n + 1].sum()) for n in range(n_max + 1)]
    return bell


def frac_to_text(x: Fraction) -> str:
    """Pretty string for rational coefficients."""

    if x.denominator == 1:
        return str(x.numerator)
    return f"{x.numerator}/{x.denominator}"


def main() -> None:
    n_max = 10

    bell_egf, coeff, facts = bell_numbers_via_egf(n_max)
    bell_ref = bell_numbers_via_stirling(n_max)

    bell_egf_arr = np.array(bell_egf, dtype=np.int64)
    bell_ref_arr = np.array(bell_ref, dtype=np.int64)

    rows = []
    for n in range(n_max + 1):
        rows.append(
            {
                "n": n,
                "coef_[x^n]A(x)=B_n/n!": frac_to_text(coeff[n]),
                "B_n_from_egf": int(bell_egf_arr[n]),
                "B_n_from_stirling": int(bell_ref_arr[n]),
                "n!": facts[n],
            }
        )

    df = pd.DataFrame(rows)

    print("=== Exponential Generating Function Demo ===")
    print("A(x) = exp(exp(x) - 1)")
    print("Interpretation: B_n = n! * [x^n]A(x) (Bell numbers)")
    print("\nCoefficient table:")
    print(df.to_string(index=False))

    print("\nSequence summary:")
    print(f"Bell by EGF      : {bell_egf}")
    print(f"Bell by Stirling : {bell_ref}")

    matches = bool(np.array_equal(bell_egf_arr, bell_ref_arr))
    print(f"Exact match      : {matches}")

    if not matches:
        raise RuntimeError("EGF result does not match Stirling reference")


if __name__ == "__main__":
    main()

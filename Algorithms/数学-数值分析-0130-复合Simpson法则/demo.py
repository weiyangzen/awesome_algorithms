"""Minimal runnable MVP for Composite Simpson Rule (MATH-0130)."""

from __future__ import annotations

import math
from typing import Callable, List, Tuple

import numpy as np


ArrayFunc = Callable[[np.ndarray], np.ndarray]


def composite_simpson(f: ArrayFunc, a: float, b: float, n: int) -> float:
    """Approximate integral of f on [a, b] by composite Simpson rule.

    Parameters
    ----------
    f : callable
        Vectorized function: accepts numpy array and returns numpy array.
    a, b : float
        Integration interval endpoints.
    n : int
        Number of subintervals, must be a positive even number.
    """
    if n <= 0 or n % 2 != 0:
        raise ValueError("n must be a positive even integer for composite Simpson rule")
    if not (math.isfinite(a) and math.isfinite(b)):
        raise ValueError("a and b must be finite numbers")
    if a == b:
        return 0.0

    sign = 1.0
    left, right = float(a), float(b)
    if right < left:
        left, right = right, left
        sign = -1.0

    h = (right - left) / n
    x = np.linspace(left, right, n + 1, dtype=float)
    y = np.asarray(f(x), dtype=float)

    if y.shape != x.shape:
        raise ValueError("f(x) must return an array with the same shape as x")

    sum_odd = np.sum(y[1:n:2])
    sum_even = np.sum(y[2:n:2])
    integral = (h / 3.0) * (y[0] + y[-1] + 4.0 * sum_odd + 2.0 * sum_even)
    return sign * float(integral)


def simpson_error_bound(max_f4: float, a: float, b: float, n: int) -> float:
    """Return deterministic bound |E| <= ((b-a)/180) * h^4 * max|f''''(x)|."""
    if n <= 0 or n % 2 != 0:
        raise ValueError("n must be a positive even integer")
    length = abs(b - a)
    h = length / n
    return (length / 180.0) * (h ** 4) * abs(max_f4)


def run_fixed_cases() -> None:
    """Run fixed test cases with known exact integrals."""
    cases: List[Tuple[str, ArrayFunc, float, float, int, float]] = [
        ("sin(x) on [0, pi]", np.sin, 0.0, math.pi, 400, 2.0),
        ("exp(x) on [0, 1]", np.exp, 0.0, 1.0, 80, math.e - 1.0),
        ("x^4 on [0, 1]", lambda x: x ** 4, 0.0, 1.0, 40, 1.0 / 5.0),
    ]

    print("Fixed cases:")
    for name, f, a, b, n, exact in cases:
        approx = composite_simpson(f, a, b, n)
        abs_err = abs(approx - exact)
        print(
            f"  {name:<20} n={n:>4}  approx={approx:.12f}  "
            f"exact={exact:.12f}  abs_err={abs_err:.3e}"
        )

    sin_err = abs(composite_simpson(np.sin, 0.0, math.pi, 400) - 2.0)
    exp_err = abs(composite_simpson(np.exp, 0.0, 1.0, 80) - (math.e - 1.0))
    poly_err = abs(composite_simpson(lambda x: x ** 4, 0.0, 1.0, 40) - 0.2)

    assert sin_err < 1e-9, f"sin integral error too large: {sin_err}"
    assert exp_err < 1e-9, f"exp integral error too large: {exp_err}"
    assert poly_err < 1e-6, f"x^4 integral error too large: {poly_err}"


def run_convergence_check() -> None:
    """Show approximately 4th-order convergence for a smooth function."""
    print("\nConvergence check for exp(x) on [0,1]:")
    exact = math.e - 1.0
    n_values = [10, 20, 40, 80]
    errors: List[float] = []

    for n in n_values:
        approx = composite_simpson(np.exp, 0.0, 1.0, n)
        err = abs(approx - exact)
        errors.append(err)

    for i, n in enumerate(n_values):
        ratio_text = "-"
        if i > 0 and errors[i] > 0:
            ratio_text = f"{errors[i - 1] / errors[i]:.2f}"
        print(f"  n={n:>3}, abs_err={errors[i]:.3e}, prev/cur={ratio_text}")

    # With halved h, error should shrink close to 16x asymptotically.
    assert errors[-1] < errors[0] / 1000.0, "convergence is weaker than expected"


def run_error_bound_check() -> None:
    """Verify theoretical bound on exp(x): f''''(x)=exp(x), max on [0,1] is e."""
    print("\nError-bound check for exp(x) on [0,1]:")
    n = 20
    approx = composite_simpson(np.exp, 0.0, 1.0, n)
    exact = math.e - 1.0
    actual_err = abs(approx - exact)
    bound = simpson_error_bound(max_f4=math.e, a=0.0, b=1.0, n=n)

    print(f"  n={n}, actual_err={actual_err:.3e}, bound={bound:.3e}")
    assert actual_err <= bound + 1e-15, "error bound check failed"


def run_edge_cases() -> None:
    """Run sanity checks for edge behavior."""
    print("\nEdge cases:")

    reverse = composite_simpson(np.sin, math.pi, 0.0, 400)
    print(f"  reverse interval integral of sin on [pi,0]: {reverse:.12f}")
    assert abs(reverse + 2.0) < 1e-10

    zero_len = composite_simpson(np.exp, 1.5, 1.5, 2)
    print(f"  zero-length interval integral: {zero_len:.12f}")
    assert zero_len == 0.0

    try:
        _ = composite_simpson(np.sin, 0.0, math.pi, 9)
        raise AssertionError("odd n should raise ValueError")
    except ValueError:
        print("  odd n check: ValueError raised as expected")


def main() -> None:
    print("Composite Simpson Rule MVP (MATH-0130)")
    print("=" * 72)

    run_fixed_cases()
    run_convergence_check()
    run_error_bound_check()
    run_edge_cases()

    print("=" * 72)
    print("All checks passed.")


if __name__ == "__main__":
    main()

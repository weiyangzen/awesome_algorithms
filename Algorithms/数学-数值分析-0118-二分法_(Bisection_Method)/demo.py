"""Minimal runnable MVP for the Bisection Method root finder."""

from __future__ import annotations

import math
from typing import Callable, List, Tuple

IterationRecord = Tuple[int, float, float, float, float, float]


def check_finite_number(value: float, name: str) -> None:
    """Raise ValueError if value is not a finite real number."""
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")


def bisection(
    f: Callable[[float], float],
    a: float,
    b: float,
    max_iter: int = 100,
    tol: float = 1e-12,
) -> Tuple[float, List[IterationRecord]]:
    """Find a root of f in [a, b] with the bisection method.

    Preconditions:
    - f is continuous on [a, b]
    - f(a) and f(b) have opposite signs or one endpoint is already a root
    """
    check_finite_number(a, "a")
    check_finite_number(b, "b")
    if not a < b:
        raise ValueError(f"Require a < b, got a={a}, b={b}")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if tol <= 0.0:
        raise ValueError("tol must be positive")

    fa = f(a)
    fb = f(b)
    check_finite_number(fa, "f(a)")
    check_finite_number(fb, "f(b)")

    if fa == 0.0:
        return a, [(0, a, b, a, fa, 0.0)]
    if fb == 0.0:
        return b, [(0, a, b, b, fb, 0.0)]
    if fa * fb > 0.0:
        raise ValueError(
            "Bisection requires a bracketing interval: f(a) and f(b) must have opposite signs"
        )

    left = a
    right = b
    f_left = fa
    history: List[IterationRecord] = []

    for k in range(max_iter):
        mid = left + 0.5 * (right - left)
        f_mid = f(mid)
        check_finite_number(f_mid, "f(mid)")

        half_width = 0.5 * (right - left)
        history.append((k, left, right, mid, f_mid, half_width))

        if abs(f_mid) <= tol or half_width <= tol:
            return mid, history

        if f_left * f_mid < 0.0:
            right = mid
        else:
            left = mid
            f_left = f_mid

    return history[-1][3], history


def relative_error(estimate: float, reference: float) -> float:
    """Compute relative error with a safe branch at zero reference."""
    if reference == 0.0:
        return abs(estimate)
    return abs(estimate - reference) / abs(reference)


def print_trace(history: List[IterationRecord], max_lines: int = 8) -> None:
    """Print the first few iteration records."""
    shown = history[:max_lines]
    for k, left, right, mid, f_mid, half_width in shown:
        print(
            "    "
            f"iter={k:2d} "
            f"a={left:.10f} "
            f"b={right:.10f} "
            f"mid={mid:.10f} "
            f"f(mid)={f_mid:.3e} "
            f"half_width={half_width:.3e}"
        )
    if len(history) > max_lines:
        print(f"    ... ({len(history) - max_lines} more iterations)")


def run_case(
    name: str,
    f: Callable[[float], float],
    a: float,
    b: float,
    reference: float,
    max_iter: int,
    tol: float,
) -> None:
    """Run one bisection demo case and print diagnostics."""
    print("=" * 80)
    print(name)
    print(f"interval=[{a}, {b}], max_iter={max_iter}, tol={tol:g}")

    root, history = bisection(f, a, b, max_iter=max_iter, tol=tol)
    err = relative_error(root, reference)

    print(f"  estimate       = {root:.16e}")
    print(f"  reference      = {reference:.16e}")
    print(f"  relative_error = {err:.3e}")
    print(f"  iterations     = {len(history)}")
    print_trace(history)


def run_invalid_interval_demo() -> None:
    """Show the expected failure mode for an invalid bracket."""
    print("=" * 80)
    print("Invalid bracket demo")
    try:
        bisection(lambda x: x * x + 1.0, -1.0, 1.0)
    except ValueError as exc:
        print(f"  expected_error = {exc}")


def main() -> None:
    max_iter = 100
    tol = 1e-12

    cases = [
        (
            "Case 1: solve x^3 - x - 2 = 0",
            lambda x: x * x * x - x - 2.0,
            1.0,
            2.0,
            1.5213797068045676,
        ),
        (
            "Case 2: solve cos(x) - x = 0",
            lambda x: math.cos(x) - x,
            0.0,
            1.0,
            0.7390851332151607,
        ),
        (
            "Case 3: solve x^2 - 2 = 0",
            lambda x: x * x - 2.0,
            1.0,
            2.0,
            math.sqrt(2.0),
        ),
        (
            "Case 4: solve exp(x) - 3 = 0",
            lambda x: math.exp(x) - 3.0,
            0.0,
            2.0,
            math.log(3.0),
        ),
        (
            "Case 5: endpoint root for x - 5 = 0",
            lambda x: x - 5.0,
            5.0,
            8.0,
            5.0,
        ),
    ]

    for name, func, a, b, reference in cases:
        run_case(name, func, a, b, reference, max_iter=max_iter, tol=tol)

    run_invalid_interval_demo()


if __name__ == "__main__":
    main()

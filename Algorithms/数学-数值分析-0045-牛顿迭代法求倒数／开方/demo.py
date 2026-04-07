"""Minimal runnable MVP: Newton iteration for reciprocal and square root."""

from __future__ import annotations

import math
from typing import List, Tuple

IterationRecord = Tuple[int, float, float]


def check_finite_number(value: float, name: str) -> None:
    """Raise ValueError if value is not finite."""
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")


def reciprocal_initial_guess(a: float) -> float:
    """Build an inexpensive initial guess for 1/a via mantissa-exponent decomposition."""
    if a == 0.0:
        raise ValueError("a must be non-zero")

    sign = -1.0 if a < 0.0 else 1.0
    mantissa, exponent = math.frexp(abs(a))  # abs(a) = mantissa * 2**exponent

    # Affine approximation of 1/m on m in [0.5, 1): line through (0.5, 2) and (1, 1).
    inv_mantissa_approx = 3.0 - 2.0 * mantissa
    return sign * math.ldexp(inv_mantissa_approx, -exponent)


def sqrt_initial_guess(s: float) -> float:
    """Build a positive initial guess for sqrt(s) via frexp/ldexp."""
    if s < 0.0:
        raise ValueError("s must be >= 0")
    if s == 0.0:
        return 0.0

    mantissa, exponent = math.frexp(s)  # mantissa in [0.5, 1)
    if exponent % 2 != 0:
        mantissa *= 2.0
        exponent -= 1

    # Now mantissa in [1, 2). Approximate sqrt(m) with a line through endpoints.
    sqrt_mantissa_approx = 0.5857864376269049 + 0.4142135623730951 * mantissa
    return math.ldexp(sqrt_mantissa_approx, exponent // 2)


def newton_reciprocal(
    a: float,
    max_iter: int = 30,
    tol: float = 1e-15,
) -> Tuple[float, List[IterationRecord]]:
    """Compute reciprocal using x_{k+1} = x_k * (2 - a*x_k)."""
    check_finite_number(a, "a")
    if a == 0.0:
        raise ValueError("a must be non-zero")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if tol <= 0.0:
        raise ValueError("tol must be positive")

    x = reciprocal_initial_guess(a)
    history: List[IterationRecord] = []

    for k in range(max_iter):
        residual = abs(1.0 - a * x)
        history.append((k, x, residual))
        if residual <= tol:
            break

        x_next = x * (2.0 - a * x)
        if not math.isfinite(x_next):
            raise RuntimeError("reciprocal Newton iteration diverged to non-finite value")

        if abs(x_next - x) <= tol * max(1.0, abs(x_next)):
            x = x_next
            history.append((k + 1, x, abs(1.0 - a * x)))
            break

        x = x_next

    return x, history


def newton_sqrt(
    s: float,
    max_iter: int = 30,
    tol: float = 1e-15,
) -> Tuple[float, List[IterationRecord]]:
    """Compute square root using x_{k+1} = 0.5*(x_k + s/x_k)."""
    check_finite_number(s, "s")
    if s < 0.0:
        raise ValueError("s must be >= 0")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if tol <= 0.0:
        raise ValueError("tol must be positive")
    if s == 0.0:
        return 0.0, [(0, 0.0, 0.0)]

    x = sqrt_initial_guess(s)
    history: List[IterationRecord] = []

    for k in range(max_iter):
        residual = abs(x * x - s)
        history.append((k, x, residual))
        if residual <= tol * max(1.0, s):
            break

        x_next = 0.5 * (x + s / x)
        if not math.isfinite(x_next):
            raise RuntimeError("sqrt Newton iteration diverged to non-finite value")

        if abs(x_next - x) <= tol * max(1.0, abs(x_next)):
            x = x_next
            history.append((k + 1, x, abs(x * x - s)))
            break

        x = x_next

    return x, history


def relative_error(estimate: float, reference: float) -> float:
    """Compute relative error with a safe branch at zero reference."""
    if reference == 0.0:
        return abs(estimate)
    return abs(estimate - reference) / abs(reference)


def print_trace(history: List[IterationRecord], max_lines: int = 6) -> None:
    """Print the first few iteration records."""
    shown = history[:max_lines]
    for k, x, residual in shown:
        print(f"    iter={k:2d}  estimate={x:.16e}  residual={residual:.3e}")
    if len(history) > max_lines:
        print(f"    ... ({len(history) - max_lines} more iterations)")


def run_reciprocal_demo(max_iter: int, tol: float) -> None:
    """Run reciprocal demos on several representative values."""
    cases = [0.2, 0.75, 3.0, -12.5, 1024.0]
    print("=" * 80)
    print("Newton reciprocal demo: x ~= 1/a")
    print("=" * 80)

    for a in cases:
        estimate, history = newton_reciprocal(a, max_iter=max_iter, tol=tol)
        reference = 1.0 / a
        err = relative_error(estimate, reference)

        print(f"a={a:.10g}")
        print(f"  estimate       = {estimate:.16e}")
        print(f"  reference      = {reference:.16e}")
        print(f"  relative_error = {err:.3e}")
        print(f"  iterations     = {len(history)}")
        print_trace(history)


def run_sqrt_demo(max_iter: int, tol: float) -> None:
    """Run square-root demos on several representative values."""
    cases = [0.0, 2.0, 10.0, 1e-6, 1e6]
    print("=" * 80)
    print("Newton square-root demo: x ~= sqrt(s)")
    print("=" * 80)

    for s in cases:
        estimate, history = newton_sqrt(s, max_iter=max_iter, tol=tol)
        reference = math.sqrt(s)
        err = relative_error(estimate, reference)

        print(f"s={s:.10g}")
        print(f"  estimate       = {estimate:.16e}")
        print(f"  reference      = {reference:.16e}")
        print(f"  relative_error = {err:.3e}")
        print(f"  iterations     = {len(history)}")
        print_trace(history)


def main() -> None:
    max_iter = 30
    tol = 1e-15

    run_reciprocal_demo(max_iter=max_iter, tol=tol)
    run_sqrt_demo(max_iter=max_iter, tol=tol)


if __name__ == "__main__":
    main()

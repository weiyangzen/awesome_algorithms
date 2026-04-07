"""Minimal runnable MVP for composite Simpson's rule."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List

import numpy as np

try:
    from scipy.integrate import quad  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    quad = None


ArrayFunc = Callable[[np.ndarray], np.ndarray | float]


@dataclass
class SimpsonCase:
    """Deterministic test case for one integral."""

    name: str
    func: ArrayFunc
    a: float
    b: float
    n: int
    true_value: float
    tol: float


@dataclass
class SimpsonResult:
    """Run result for one case."""

    name: str
    n: int
    integral_n: float
    integral_2n: float
    true_value: float
    abs_error: float
    richardson_est: float
    tol: float
    scipy_abs_error: float | None


def evaluate_scalar(func: ArrayFunc, t: float) -> float:
    """Evaluate func at one scalar point and return a finite float."""
    try:
        out = func(np.array([t], dtype=float))
        arr = np.asarray(out, dtype=float)
        value = float(arr.reshape(-1)[0])
    except Exception:
        value = float(func(float(t)))

    if not np.isfinite(value):
        raise ValueError(f"Function produced non-finite value at x={t}.")
    return value


def validate_interval_and_n(a: float, b: float, n: int) -> None:
    """Validate interval and mesh count constraints."""
    if not np.isfinite(a) or not np.isfinite(b):
        raise ValueError("Interval endpoints must be finite.")
    if a >= b:
        raise ValueError(f"Require a < b, got a={a}, b={b}.")
    if n <= 0:
        raise ValueError(f"Require n > 0, got n={n}.")
    if n % 2 != 0:
        raise ValueError(f"Composite Simpson requires even n, got n={n}.")


def evaluate_on_grid(func: ArrayFunc, x: np.ndarray) -> np.ndarray:
    """Safely evaluate function values on grid x with finite-value checks."""
    try:
        y = np.asarray(func(x), dtype=float)
        if y.shape != x.shape:
            raise ValueError("Function output shape mismatch; fallback to vectorize.")
    except Exception:
        y = np.vectorize(lambda t: evaluate_scalar(func, float(t)), otypes=[float])(x)
        y = np.asarray(y, dtype=float)

    if not np.all(np.isfinite(y)):
        raise ValueError("Function produced non-finite values on the integration grid.")
    return y


def composite_simpson(func: ArrayFunc, a: float, b: float, n: int) -> float:
    """Compute integral by composite Simpson's rule over [a, b] with even n."""
    validate_interval_and_n(a=a, b=b, n=n)

    x = np.linspace(a, b, n + 1, dtype=float)
    y = evaluate_on_grid(func=func, x=x)

    h = (b - a) / n
    endpoint_sum = y[0] + y[-1]
    odd_sum = np.sum(y[1:-1:2])
    even_sum = np.sum(y[2:-1:2])

    integral = (h / 3.0) * (endpoint_sum + 4.0 * odd_sum + 2.0 * even_sum)
    return float(integral)


def richardson_error_estimate(i_n: float, i_2n: float) -> float:
    """Estimate Simpson discretization error via Richardson extrapolation."""
    return abs(i_2n - i_n) / 15.0


def build_cases() -> List[SimpsonCase]:
    """Build deterministic integration tasks with known analytic answers."""
    return [
        SimpsonCase(
            name="sin(x) on [0, pi]",
            func=np.sin,
            a=0.0,
            b=math.pi,
            n=120,
            true_value=2.0,
            tol=1e-9,
        ),
        SimpsonCase(
            name="exp(x) on [0, 1]",
            func=np.exp,
            a=0.0,
            b=1.0,
            n=40,
            true_value=math.e - 1.0,
            tol=1e-9,
        ),
        SimpsonCase(
            name="1/(1+x^2) on [0, 1]",
            func=lambda x: 1.0 / (1.0 + x * x),
            a=0.0,
            b=1.0,
            n=60,
            true_value=math.atan(1.0),
            tol=1e-10,
        ),
        SimpsonCase(
            name="x^3 - 2x + 1 on [-1, 2]",
            func=lambda x: x**3 - 2.0 * x + 1.0,
            a=-1.0,
            b=2.0,
            n=20,
            true_value=3.75,
            tol=1e-12,
        ),
    ]


def run_case(case: SimpsonCase) -> SimpsonResult:
    """Run one case, perform checks, and return summarized metrics."""
    i_n = composite_simpson(case.func, case.a, case.b, case.n)
    i_2n = composite_simpson(case.func, case.a, case.b, case.n * 2)

    abs_error = abs(i_2n - case.true_value)
    richardson_est = richardson_error_estimate(i_n=i_n, i_2n=i_2n)

    if abs_error > case.tol:
        raise AssertionError(
            f"Case '{case.name}' failed: abs_error={abs_error:.3e} exceeds tol={case.tol:.3e}"
        )

    scipy_abs_error: float | None = None
    if quad is not None:
        ref, _ = quad(lambda t: evaluate_scalar(case.func, float(t)), case.a, case.b)
        scipy_abs_error = abs(i_2n - float(ref))

    return SimpsonResult(
        name=case.name,
        n=case.n,
        integral_n=i_n,
        integral_2n=i_2n,
        true_value=case.true_value,
        abs_error=abs_error,
        richardson_est=richardson_est,
        tol=case.tol,
        scipy_abs_error=scipy_abs_error,
    )


def format_results_table(results: List[SimpsonResult]) -> str:
    """Create a compact plain-text result table."""
    header = (
        f"{'case':34s} {'n':>4s} {'abs_error':>12s} "
        f"{'richardson':>12s} {'tol':>12s} {'scipy_err':>12s}"
    )
    rows = [header, "-" * len(header)]

    for r in results:
        scipy_text = "N/A" if r.scipy_abs_error is None else f"{r.scipy_abs_error:.3e}"
        rows.append(
            f"{r.name:34.34s} {r.n:4d} {r.abs_error:12.3e} "
            f"{r.richardson_est:12.3e} {r.tol:12.3e} {scipy_text:>12s}"
        )
    return "\n".join(rows)


def verify_guardrail_for_odd_n() -> None:
    """Ensure odd n raises an exception as required by Simpson's rule."""
    try:
        composite_simpson(np.sin, 0.0, 1.0, 9)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for odd n, but no error was raised.")


def main() -> None:
    verify_guardrail_for_odd_n()

    cases = build_cases()
    results = [run_case(case) for case in cases]

    print("Composite Simpson Rule MVP")
    print(f"scipy_reference_available: {quad is not None}")
    print(format_results_table(results))

    worst_case = max(results, key=lambda r: r.abs_error)
    print("\nSummary")
    print(f"cases_total           : {len(results)}")
    print(f"worst_case            : {worst_case.name}")
    print(f"worst_abs_error       : {worst_case.abs_error:.3e}")
    print("all_checks_passed     : True")


if __name__ == "__main__":
    main()

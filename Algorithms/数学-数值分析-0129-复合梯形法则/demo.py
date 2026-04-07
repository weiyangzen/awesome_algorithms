"""Minimal runnable MVP: Composite Trapezoidal Rule for numerical integration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

ArrayFunc = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class IntegralCase:
    """A benchmark integration case with closed-form reference value."""

    name: str
    func: ArrayFunc
    a: float
    b: float
    exact: float


def _check_finite(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")


def _evaluate_on_grid(func: ArrayFunc, x: np.ndarray) -> np.ndarray:
    """Evaluate a callable on grid x, with a scalar-loop fallback."""
    try:
        y = np.asarray(func(x), dtype=float)
        if y.shape == x.shape:
            return y
    except Exception:
        pass

    # Fallback for non-vectorized callables.
    values = []
    for xi in x:
        try:
            yi = func(np.asarray([xi]))
            yi = np.asarray(yi, dtype=float).reshape(-1)
            values.append(float(yi[0]))
        except Exception:
            values.append(float(func(float(xi))))
    return np.asarray(values, dtype=float)


def composite_trapezoidal(func: ArrayFunc, a: float, b: float, n: int) -> float:
    """Compute integral_{a}^{b} f(x) dx by the composite trapezoidal rule."""
    _check_finite(a, "a")
    _check_finite(b, "b")
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if a == b:
        return 0.0

    h = (b - a) / n
    x = np.linspace(a, b, n + 1, dtype=float)
    y = _evaluate_on_grid(func, x)
    weighted_sum = y[0] + y[-1] + 2.0 * np.sum(y[1:-1])
    return 0.5 * h * float(weighted_sum)


def richardson_error_estimate(i_n: float, i_2n: float, order: int = 2) -> float:
    """Estimate truncation error from two mesh levels via Richardson extrapolation."""
    if order <= 0:
        raise ValueError("order must be positive")
    return abs(i_2n - i_n) / (2**order - 1)


def relative_error(estimate: float, exact: float) -> float:
    if exact == 0.0:
        return abs(estimate)
    return abs(estimate - exact) / abs(exact)


def run_case(case: IntegralCase, n_values: Sequence[int]) -> None:
    print("=" * 96)
    print(f"{case.name}: integral on [{case.a}, {case.b}]")
    print(f"exact = {case.exact:.16e}")
    print("-" * 96)
    print(
        " n      estimate                 abs_error        rel_error      "
        "richardson_est   observed_order"
    )

    prev_estimate: float | None = None
    prev_abs_error: float | None = None

    for n in n_values:
        estimate = composite_trapezoidal(case.func, case.a, case.b, n)
        abs_err = abs(estimate - case.exact)
        rel_err = relative_error(estimate, case.exact)

        if prev_estimate is None:
            richardson = float("nan")
            observed_order = float("nan")
        else:
            richardson = richardson_error_estimate(prev_estimate, estimate, order=2)
            if prev_abs_error is None or abs_err == 0.0 or prev_abs_error == 0.0:
                observed_order = float("nan")
            else:
                observed_order = math.log(prev_abs_error / abs_err, 2.0)

        richardson_txt = f"{richardson:.3e}" if math.isfinite(richardson) else "   -   "
        order_txt = f"{observed_order:.3f}" if math.isfinite(observed_order) else "  -  "

        print(
            f"{n:4d}  {estimate: .16e}  {abs_err: .3e}   {rel_err: .3e}    "
            f"{richardson_txt:>12}      {order_txt:>8}"
        )

        prev_estimate = estimate
        prev_abs_error = abs_err


def main() -> None:
    cases = [
        IntegralCase(
            name="Case 1: f(x) = sin(x)",
            func=lambda x: np.sin(x),
            a=0.0,
            b=math.pi,
            exact=2.0,
        ),
        IntegralCase(
            name="Case 2: f(x) = exp(-x^2)",
            func=lambda x: np.exp(-(x**2)),
            a=0.0,
            b=1.0,
            exact=0.5 * math.sqrt(math.pi) * math.erf(1.0),
        ),
        IntegralCase(
            name="Case 3: f(x) = 1/(1+x^2)",
            func=lambda x: 1.0 / (1.0 + x**2),
            a=0.0,
            b=1.0,
            exact=math.atan(1.0),
        ),
    ]

    n_values = [4, 8, 16, 32, 64, 128, 256]

    for case in cases:
        run_case(case, n_values=n_values)


if __name__ == "__main__":
    main()

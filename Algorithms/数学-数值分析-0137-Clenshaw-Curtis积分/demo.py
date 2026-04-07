"""Clenshaw-Curtis quadrature MVP.

This script implements Clenshaw-Curtis integration from scratch:
- Build Chebyshev-Lobatto nodes on [-1, 1]
- Compute Clenshaw-Curtis weights with cosine-series formulas
- Affine-map nodes to [a, b] and apply weighted summation
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np


ArrayFunc = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class IntegralCase:
    """A benchmark integral with known exact value."""

    name: str
    func: ArrayFunc
    a: float
    b: float
    exact: float


def _check_finite(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")


def _evaluate_function(func: ArrayFunc, x: np.ndarray) -> np.ndarray:
    """Evaluate func on x, preferring vectorized call and falling back to scalar loop."""
    try:
        y = np.asarray(func(x), dtype=float)
        if y.shape == x.shape and np.all(np.isfinite(y)):
            return y
    except Exception:
        pass

    values: list[float] = []
    for xi in x:
        try:
            yi = np.asarray(func(np.array([xi], dtype=float)), dtype=float)
            if yi.size == 1:
                values.append(float(yi.reshape(-1)[0]))
                continue
        except Exception:
            pass

        yi_scalar = np.asarray(func(float(xi)), dtype=float)
        if yi_scalar.size != 1:
            raise ValueError("Scalar fallback evaluation must return a scalar value.")
        values.append(float(yi_scalar.reshape(-1)[0]))

    y = np.array(values, dtype=float)
    if not np.all(np.isfinite(y)):
        raise ValueError("Function evaluation produced non-finite values.")
    return y


def clenshaw_curtis_nodes_weights(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return nodes and weights for n-th order Clenshaw-Curtis rule on [-1, 1]."""
    if n < 0:
        raise ValueError("n must be >= 0.")
    if n == 0:
        return np.array([1.0]), np.array([2.0])

    theta = math.pi * np.arange(n + 1, dtype=float) / n
    x = np.cos(theta)
    w = np.zeros(n + 1, dtype=float)

    interior = np.arange(1, n)
    v = np.ones(n - 1, dtype=float)

    if n % 2 == 0:
        endpoint_weight = 1.0 / (n * n - 1.0)
        w[0] = endpoint_weight
        w[-1] = endpoint_weight

        for k in range(1, n // 2):
            v -= 2.0 * np.cos(2.0 * k * theta[interior]) / (4.0 * k * k - 1.0)
        v -= np.cos(n * theta[interior]) / (n * n - 1.0)
    else:
        endpoint_weight = 1.0 / (n * n)
        w[0] = endpoint_weight
        w[-1] = endpoint_weight

        for k in range(1, (n - 1) // 2 + 1):
            v -= 2.0 * np.cos(2.0 * k * theta[interior]) / (4.0 * k * k - 1.0)

    w[interior] = 2.0 * v / n
    return x, w


def clenshaw_curtis_integrate(func: ArrayFunc, a: float, b: float, n: int) -> float:
    """Approximate integral of func over [a, b] with n-th order Clenshaw-Curtis."""
    _check_finite(a, "a")
    _check_finite(b, "b")
    if n < 0:
        raise ValueError("n must be >= 0.")
    if a == b:
        return 0.0

    x_std, w_std = clenshaw_curtis_nodes_weights(n)
    scale = 0.5 * (b - a)
    shift = 0.5 * (a + b)
    x_mapped = shift + scale * x_std
    y = _evaluate_function(func, x_mapped)
    return float(scale * np.dot(w_std, y))


def _relative_error(estimate: float, exact: float) -> float:
    if exact == 0.0:
        return abs(estimate - exact)
    return abs(estimate - exact) / abs(exact)


def _observed_order(prev_err: float, curr_err: float) -> float:
    if prev_err <= 0.0 or curr_err <= 0.0:
        return float("nan")
    return math.log(prev_err / curr_err, 2.0)


def _format_float(x: float) -> str:
    if math.isnan(x):
        return "nan"
    return f"{x:.6e}"


def run_case(case: IntegralCase, n_values: list[int]) -> None:
    print("=" * 88)
    print(f"{case.name}")
    print(f"Interval = [{case.a}, {case.b}], exact = {case.exact:.12f}")
    print("-" * 88)
    print(
        f"{'n':>6} | {'estimate':>16} | {'abs_error':>12} | "
        f"{'rel_error':>12} | {'obs_order':>10}"
    )
    print("-" * 88)

    prev_abs_error: float | None = None
    for n in n_values:
        estimate = clenshaw_curtis_integrate(case.func, case.a, case.b, n)
        abs_error = abs(estimate - case.exact)
        rel_error = _relative_error(estimate, case.exact)
        order = float("nan") if prev_abs_error is None else _observed_order(prev_abs_error, abs_error)

        print(
            f"{n:6d} | {estimate:16.10f} | {_format_float(abs_error):>12} | "
            f"{_format_float(rel_error):>12} | {_format_float(order):>10}"
        )
        prev_abs_error = abs_error
    print()


def main() -> None:
    cases = [
        IntegralCase(
            name="Case 1: ∫[0,π] sin(x) dx",
            func=lambda x: np.sin(x),
            a=0.0,
            b=math.pi,
            exact=2.0,
        ),
        IntegralCase(
            name="Case 2: ∫[-1,1] exp(x) dx",
            func=lambda x: np.exp(x),
            a=-1.0,
            b=1.0,
            exact=math.e - math.exp(-1.0),
        ),
        IntegralCase(
            name="Case 3: ∫[-1,1] 1/(1+25x^2) dx",
            func=lambda x: 1.0 / (1.0 + 25.0 * x * x),
            a=-1.0,
            b=1.0,
            exact=0.4 * math.atan(5.0),
        ),
        IntegralCase(
            name="Case 4: ∫[-1,1] |x| dx (non-smooth benchmark)",
            func=lambda x: np.abs(x),
            a=-1.0,
            b=1.0,
            exact=1.0,
        ),
    ]
    n_values = [2, 4, 8, 16, 32, 64, 128]

    print("Clenshaw-Curtis Quadrature MVP")
    print("All runs are non-interactive and deterministic.")
    print()
    for case in cases:
        run_case(case, n_values)


if __name__ == "__main__":
    main()

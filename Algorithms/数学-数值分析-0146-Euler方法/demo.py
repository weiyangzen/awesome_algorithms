"""Minimal runnable MVP for Euler's method (explicit first-order ODE solver)."""

from __future__ import annotations

import math
from typing import Callable, List, Sequence, Tuple

Derivative = Callable[[float, float], float]


def validate_inputs(t0: float, y0: float, h: float, steps: int) -> None:
    """Validate scalar inputs for the Euler integrator."""
    for name, value in (("t0", t0), ("y0", y0), ("h", h)):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")
    if h <= 0.0:
        raise ValueError(f"h must be positive, got {h}")
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")


def euler_solve(
    f: Derivative,
    t0: float,
    y0: float,
    h: float,
    steps: int,
) -> Tuple[List[float], List[float]]:
    """Solve y' = f(t, y) with explicit Euler for a fixed number of steps."""
    validate_inputs(t0=t0, y0=y0, h=h, steps=steps)

    t_values: List[float] = [t0]
    y_values: List[float] = [y0]

    t = t0
    y = y0
    for _ in range(steps):
        slope = f(t, y)
        if not math.isfinite(slope):
            raise RuntimeError("non-finite slope encountered during Euler iteration")

        y = y + h * slope
        t = t + h
        t_values.append(t)
        y_values.append(y)

    return t_values, y_values


def ode_rhs(t: float, y: float) -> float:
    """Example ODE right-hand side: y' = y - t^2 + 1."""
    return y - t * t + 1.0


def ode_exact(t: float) -> float:
    """Exact solution for the demo ODE with y(0)=0.5: y=(t+1)^2-0.5*e^t."""
    return (t + 1.0) ** 2 - 0.5 * math.exp(t)


def max_abs_error(t_values: Sequence[float], y_values: Sequence[float]) -> float:
    """Compute max absolute error against the exact solution on the sampled grid."""
    if len(t_values) != len(y_values):
        raise ValueError("t_values and y_values must have the same length")
    return max(abs(y - ode_exact(t)) for t, y in zip(t_values, y_values))


def run_single_resolution(h: float, t_end: float = 2.0) -> Tuple[int, float, float]:
    """Run one Euler solve and return (steps, final_error, max_error)."""
    steps_float = t_end / h
    steps = int(round(steps_float))
    if abs(steps - steps_float) > 1e-12:
        raise ValueError("t_end / h must be close to an integer for this fixed-step demo")

    t_values, y_values = euler_solve(f=ode_rhs, t0=0.0, y0=0.5, h=h, steps=steps)

    final_error = abs(y_values[-1] - ode_exact(t_values[-1]))
    max_error = max_abs_error(t_values, y_values)
    return steps, final_error, max_error


def print_trajectory_sample(h: float, rows: int = 6) -> None:
    """Print a short trajectory table for the chosen step size."""
    t_end = 2.0
    steps = int(round(t_end / h))
    t_values, y_values = euler_solve(f=ode_rhs, t0=0.0, y0=0.5, h=h, steps=steps)

    print("-" * 80)
    print(f"Trajectory sample (h={h}, first {rows} rows)")
    print(" n      t           y_euler         y_exact         abs_error")
    print("-" * 80)

    show = min(rows, len(t_values))
    for i in range(show):
        t = t_values[i]
        y_num = y_values[i]
        y_ref = ode_exact(t)
        err = abs(y_num - y_ref)
        print(f"{i:2d}  {t:8.4f}  {y_num:14.8f}  {y_ref:14.8f}  {err:12.4e}")


def estimate_orders(results: Sequence[Tuple[float, int, float, float]]) -> List[Tuple[float, float]]:
    """Estimate empirical convergence order p from consecutive step sizes."""
    orders: List[Tuple[float, float]] = []
    for i in range(len(results) - 1):
        h1, _, _, e1 = results[i]
        h2, _, _, e2 = results[i + 1]
        if e1 <= 0.0 or e2 <= 0.0:
            continue
        p = math.log(e1 / e2) / math.log(h1 / h2)
        orders.append((h2, p))
    return orders


def main() -> None:
    print("Euler method demo for y' = y - t^2 + 1, y(0)=0.5, t in [0, 2]")
    print("Exact solution: y(t) = (t+1)^2 - 0.5*exp(t)")

    h_values = [0.2, 0.1, 0.05, 0.025]
    results: List[Tuple[float, int, float, float]] = []

    print("=" * 80)
    print("Convergence table")
    print(" h        steps    final_abs_error    max_abs_error")
    print("=" * 80)
    for h in h_values:
        steps, final_error, max_error = run_single_resolution(h)
        results.append((h, steps, final_error, max_error))
        print(f"{h:7.3f}  {steps:5d}    {final_error:14.6e}   {max_error:14.6e}")

    print("=" * 80)
    print("Empirical order (using max_abs_error across consecutive h)")
    for h, p in estimate_orders(results):
        print(f"h={h:7.3f} -> estimated order p={p:.4f}")

    print_trajectory_sample(h=0.2, rows=8)


if __name__ == "__main__":
    main()

"""Minimal runnable MVP for classic Runge-Kutta 4th-order (RK4) ODE solving."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

Derivative = Callable[[float, float], float]


@dataclass
class RK4StepTrace:
    """Trace record for one RK4 step (useful for debugging and teaching)."""

    n: int
    t_n: float
    y_n: float
    k1: float
    k2: float
    k3: float
    k4: float
    y_next: float


def validate_inputs(t0: float, y0: float, h: float, steps: int) -> None:
    """Validate scalar inputs for the fixed-step integrator."""
    for name, value in (("t0", t0), ("y0", y0), ("h", h)):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")
    if h <= 0.0:
        raise ValueError(f"h must be positive, got {h}")
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")


def rk4_step(f: Derivative, t: float, y: float, h: float) -> Tuple[float, float, float, float, float]:
    """Advance one RK4 step and return (y_next, k1, k2, k3, k4)."""
    k1 = f(t, y)
    k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
    k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
    k4 = f(t + h, y + h * k3)

    for name, value in (("k1", k1), ("k2", k2), ("k3", k3), ("k4", k4)):
        if not math.isfinite(value):
            raise RuntimeError(f"non-finite slope {name} encountered: {value!r}")

    y_next = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    if not math.isfinite(y_next):
        raise RuntimeError("non-finite state produced by RK4 update")

    return y_next, k1, k2, k3, k4


def rk4_solve(
    f: Derivative,
    t0: float,
    y0: float,
    h: float,
    steps: int,
) -> Tuple[List[float], List[float], List[RK4StepTrace]]:
    """Solve y' = f(t, y) with fixed-step classic RK4."""
    validate_inputs(t0=t0, y0=y0, h=h, steps=steps)

    t_values: List[float] = [t0]
    y_values: List[float] = [y0]
    traces: List[RK4StepTrace] = []

    t = t0
    y = y0
    for n in range(steps):
        y_next, k1, k2, k3, k4 = rk4_step(f=f, t=t, y=y, h=h)
        traces.append(
            RK4StepTrace(
                n=n,
                t_n=t,
                y_n=y,
                k1=k1,
                k2=k2,
                k3=k3,
                k4=k4,
                y_next=y_next,
            )
        )
        t = t + h
        y = y_next
        t_values.append(t)
        y_values.append(y)

    return t_values, y_values, traces


def ode_rhs(t: float, y: float) -> float:
    """Example ODE: y' = y - t^2 + 1."""
    return y - t * t + 1.0


def ode_exact(t: float) -> float:
    """Exact solution for y(0)=0.5: y(t) = (t+1)^2 - 0.5*exp(t)."""
    return (t + 1.0) ** 2 - 0.5 * math.exp(t)


def max_abs_error(t_values: Sequence[float], y_values: Sequence[float]) -> float:
    """Compute max absolute error on the sampled grid."""
    if len(t_values) != len(y_values):
        raise ValueError("t_values and y_values must have the same length")
    return max(abs(y - ode_exact(t)) for t, y in zip(t_values, y_values))


def run_single_resolution(h: float, t_end: float = 2.0) -> Tuple[int, float, float]:
    """Run one RK4 solve and return (steps, final_error, max_error)."""
    steps_float = t_end / h
    steps = int(round(steps_float))
    if abs(steps - steps_float) > 1e-12:
        raise ValueError("t_end / h must be close to an integer for this fixed-step demo")

    t_values, y_values, _ = rk4_solve(f=ode_rhs, t0=0.0, y0=0.5, h=h, steps=steps)

    final_error = abs(y_values[-1] - ode_exact(t_values[-1]))
    global_error = max_abs_error(t_values, y_values)
    return steps, final_error, global_error


def estimate_orders(results: Sequence[Tuple[float, int, float, float]]) -> List[Tuple[float, float]]:
    """Estimate convergence order from consecutive step sizes using max error."""
    orders: List[Tuple[float, float]] = []
    for i in range(len(results) - 1):
        h1, _, _, e1 = results[i]
        h2, _, _, e2 = results[i + 1]
        if e1 <= 0.0 or e2 <= 0.0:
            continue
        p = math.log(e1 / e2) / math.log(h1 / h2)
        orders.append((h2, p))
    return orders


def print_trajectory_sample(h: float, rows: int = 6) -> None:
    """Print a short RK4 trajectory table."""
    t_end = 2.0
    steps = int(round(t_end / h))
    t_values, y_values, _ = rk4_solve(f=ode_rhs, t0=0.0, y0=0.5, h=h, steps=steps)

    print("-" * 92)
    print(f"Trajectory sample (h={h}, first {rows} rows)")
    print(" n      t           y_rk4           y_exact         abs_error")
    print("-" * 92)

    show = min(rows, len(t_values))
    for i in range(show):
        t = t_values[i]
        y_num = y_values[i]
        y_ref = ode_exact(t)
        err = abs(y_num - y_ref)
        print(f"{i:2d}  {t:8.4f}  {y_num:14.8f}  {y_ref:14.8f}  {err:12.4e}")


def print_step_trace(h: float, trace_rows: int = 3) -> None:
    """Print first few RK4 internal slopes (k1..k4) to expose source-level flow."""
    t_end = 2.0
    steps = int(round(t_end / h))
    _, _, traces = rk4_solve(f=ode_rhs, t0=0.0, y0=0.5, h=h, steps=steps)

    print("-" * 92)
    print(f"Internal RK4 slopes preview (h={h}, first {trace_rows} steps)")
    print(" n      t_n         y_n            k1            k2            k3            k4")
    print("-" * 92)

    show = min(trace_rows, len(traces))
    for i in range(show):
        tr = traces[i]
        print(
            f"{tr.n:2d}  {tr.t_n:8.4f}  {tr.y_n:12.8f}  "
            f"{tr.k1:12.8f}  {tr.k2:12.8f}  {tr.k3:12.8f}  {tr.k4:12.8f}"
        )


def main() -> None:
    print("RK4 demo for y' = y - t^2 + 1, y(0)=0.5, t in [0, 2]")
    print("Exact solution: y(t) = (t+1)^2 - 0.5*exp(t)")

    h_values = [0.2, 0.1, 0.05, 0.025]
    results: List[Tuple[float, int, float, float]] = []

    print("=" * 92)
    print("Convergence table")
    print(" h        steps    final_abs_error    max_abs_error")
    print("=" * 92)
    for h in h_values:
        steps, final_error, max_error = run_single_resolution(h=h)
        results.append((h, steps, final_error, max_error))
        print(f"{h:7.3f}  {steps:5d}    {final_error:14.6e}   {max_error:14.6e}")

    print("=" * 92)
    print("Empirical order p (using max_abs_error across consecutive h)")
    for h, p in estimate_orders(results):
        print(f"h={h:7.3f} -> estimated order p={p:.4f}")

    print_trajectory_sample(h=0.2, rows=8)
    print_step_trace(h=0.2, trace_rows=4)


if __name__ == "__main__":
    main()

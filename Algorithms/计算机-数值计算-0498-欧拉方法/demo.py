"""Euler method MVP for solving a simple ODE initial value problem."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np


def ode_rhs(t: float, y: float) -> float:
    """Right-hand side of y' = y."""
    _ = t
    return y


def analytic_solution(t: float, y0: float = 1.0) -> float:
    """Exact solution for y' = y, y(0) = y0."""
    return y0 * math.exp(t)


def euler_solve(
    f: Callable[[float, float], float],
    t0: float,
    y0: float,
    h: float,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve y' = f(t, y) with the explicit Euler method."""
    if h <= 0:
        raise ValueError("Step size h must be positive.")
    if n_steps <= 0:
        raise ValueError("n_steps must be a positive integer.")

    ts = np.empty(n_steps + 1, dtype=float)
    ys = np.empty(n_steps + 1, dtype=float)
    ts[0], ys[0] = t0, y0

    for n in range(n_steps):
        slope = f(ts[n], ys[n])
        ys[n + 1] = ys[n] + h * slope
        ts[n + 1] = ts[n] + h

    return ts, ys


def run_case(h: float, t_end: float = 1.0) -> dict[str, float | np.ndarray]:
    """Run one Euler configuration and return key metrics."""
    t0, y0 = 0.0, 1.0
    n_steps = int(round((t_end - t0) / h))
    ts, ys = euler_solve(ode_rhs, t0=t0, y0=y0, h=h, n_steps=n_steps)
    exact = analytic_solution(ts[-1], y0=y0)
    error = abs(ys[-1] - exact)
    return {
        "h": h,
        "ts": ts,
        "ys": ys,
        "y_end": float(ys[-1]),
        "y_exact": float(exact),
        "abs_error": float(error),
    }


def main() -> None:
    case_h = run_case(h=0.1)
    case_h2 = run_case(h=0.05)
    order = math.log(case_h["abs_error"] / case_h2["abs_error"], 2.0)

    print("Euler Method MVP")
    print("Problem: y' = y, y(0)=1, t in [0,1]")
    print("-" * 62)

    for case in (case_h, case_h2):
        h = case["h"]
        y_end = case["y_end"]
        y_exact = case["y_exact"]
        abs_error = case["abs_error"]
        ts = case["ts"][:5]
        ys = case["ys"][:5]
        print(f"h = {h:.3f}")
        print(
            f"  y_num(1) = {y_end:.10f}, y_exact(1) = {y_exact:.10f}, "
            f"abs_error = {abs_error:.10f}"
        )
        print(f"  first 5 grid points t: {np.array2string(ts, precision=3)}")
        print(f"  first 5 values      y: {np.array2string(ys, precision=6)}")
        print("-" * 62)

    print(f"Estimated convergence order p ≈ {order:.4f} (expected near 1)")


if __name__ == "__main__":
    main()

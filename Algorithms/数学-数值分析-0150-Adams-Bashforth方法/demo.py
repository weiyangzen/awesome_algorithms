"""Minimal runnable MVP for Adams-Bashforth method (MATH-0150)."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Dict, List, Tuple

import numpy as np


RHS = Callable[[float, float], float]


AB_COEFFS: Dict[int, np.ndarray] = {
    1: np.array([1.0]),
    2: np.array([3.0 / 2.0, -1.0 / 2.0]),
    3: np.array([23.0 / 12.0, -16.0 / 12.0, 5.0 / 12.0]),
    4: np.array([55.0 / 24.0, -59.0 / 24.0, 37.0 / 24.0, -9.0 / 24.0]),
}


@dataclass
class ExpODE:
    """y' = y with an evaluation counter."""

    nfev: int = 0

    def __call__(self, t: float, y: float) -> float:
        self.nfev += 1
        _ = t
        return y


@dataclass
class SolveResult:
    t: np.ndarray
    y: np.ndarray
    nfev: int


def rk4_step(f: RHS, t: float, y: float, h: float) -> float:
    """One classical RK4 step used for AB startup."""
    k1 = f(t, y)
    k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
    k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
    k4 = f(t + h, y + h * k3)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate_adams_bashforth(
    f: RHS,
    t0: float,
    y0: float,
    h: float,
    n_steps: int,
    order: int,
) -> SolveResult:
    """Integrate ODE by explicit Adams-Bashforth method of given order (1..4)."""
    if order not in AB_COEFFS:
        raise ValueError(f"order must be in {sorted(AB_COEFFS)}, got {order}")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if not math.isfinite(t0) or not math.isfinite(y0):
        raise ValueError("t0 and y0 must be finite")
    if not math.isfinite(h) or h <= 0.0:
        raise ValueError("h must be finite and positive")

    t = t0 + h * np.arange(n_steps + 1, dtype=float)
    y = np.empty(n_steps + 1, dtype=float)
    y[0] = y0

    # Bootstrap first (order - 1) points with RK4.
    startup_steps = min(order - 1, n_steps)
    for i in range(startup_steps):
        y[i + 1] = rk4_step(f, float(t[i]), float(y[i]), h)
        if not math.isfinite(float(y[i + 1])):
            raise RuntimeError("non-finite value encountered during RK4 startup")

    if n_steps <= startup_steps:
        nfev = getattr(f, "nfev", -1)
        return SolveResult(t=t, y=y, nfev=nfev)

    coeffs = AB_COEFFS[order]

    # Cache derivative history at available points.
    f_values = np.empty(n_steps + 1, dtype=float)
    for i in range(startup_steps + 1):
        f_values[i] = f(float(t[i]), float(y[i]))

    for n in range(startup_steps, n_steps):
        weighted_sum = 0.0
        for j, c in enumerate(coeffs):
            weighted_sum += float(c) * float(f_values[n - j])

        y[n + 1] = y[n] + h * weighted_sum
        if not math.isfinite(float(y[n + 1])):
            raise RuntimeError("non-finite value encountered in AB update")

        f_values[n + 1] = f(float(t[n + 1]), float(y[n + 1]))

    nfev = getattr(f, "nfev", -1)
    return SolveResult(t=t, y=y, nfev=nfev)


def exact_exp_solution(t: np.ndarray) -> np.ndarray:
    """Exact solution of y' = y, y(0)=1."""
    return np.exp(t)


def run_convergence_demo(order: int) -> Tuple[List[int], List[float]]:
    """Check endpoint error trend under mesh refinement."""
    print(f"\nConvergence demo for AB{order} on y'=y, y(0)=1")
    print("N      h         endpoint_abs_error    prev/cur")

    n_list = [20, 40, 80, 160]
    errors: List[float] = []

    for idx, n_steps in enumerate(n_list):
        h = 1.0 / n_steps
        ode = ExpODE()
        result = integrate_adams_bashforth(
            f=ode,
            t0=0.0,
            y0=1.0,
            h=h,
            n_steps=n_steps,
            order=order,
        )

        exact_end = math.e
        err = abs(float(result.y[-1]) - exact_end)
        errors.append(err)

        ratio_text = "-"
        if idx > 0 and err > 0.0:
            ratio_text = f"{errors[idx - 1] / err:.3f}"

        print(f"{n_steps:<6d} {h:<9.5f} {err:<21.6e} {ratio_text}")

    # Asymptotically, error ratio should approach 2^order.
    # We use relaxed checks to keep MVP robust across environments.
    last_ratio = errors[-2] / errors[-1]
    if order == 2:
        assert last_ratio > 3.0, f"AB2 ratio too small: {last_ratio}"
    elif order == 4:
        assert last_ratio > 10.0, f"AB4 ratio too small: {last_ratio}"

    return n_list, errors


def run_accuracy_and_cost_demo() -> None:
    """Compare AB2 and AB4 at same step count on accuracy and RHS call count."""
    print("\nAccuracy and cost comparison at N=80")

    n_steps = 80
    h = 1.0 / n_steps

    for order in (2, 4):
        ode = ExpODE()
        result = integrate_adams_bashforth(
            f=ode,
            t0=0.0,
            y0=1.0,
            h=h,
            n_steps=n_steps,
            order=order,
        )
        err = abs(float(result.y[-1]) - math.e)
        print(f"AB{order}: endpoint_err={err:.6e}, nfev={result.nfev}")

    # At same N, AB4 should generally be more accurate than AB2 on smooth problems.
    ode2 = ExpODE()
    r2 = integrate_adams_bashforth(ode2, 0.0, 1.0, h, n_steps, 2)
    e2 = abs(float(r2.y[-1]) - math.e)

    ode4 = ExpODE()
    r4 = integrate_adams_bashforth(ode4, 0.0, 1.0, h, n_steps, 4)
    e4 = abs(float(r4.y[-1]) - math.e)

    assert e4 < e2, "AB4 should outperform AB2 on this smooth test"


def main() -> None:
    print("Adams-Bashforth Method MVP (MATH-0150)")
    print("=" * 72)

    run_convergence_demo(order=2)
    run_convergence_demo(order=4)
    run_accuracy_and_cost_demo()

    print("=" * 72)
    print("All checks passed.")


if __name__ == "__main__":
    main()

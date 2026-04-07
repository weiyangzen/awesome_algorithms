"""Minimal runnable MVP for Runge-Kutta method (classical RK4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import math

import numpy as np

State = np.ndarray
RhsFunc = Callable[[float, State], State]


@dataclass(frozen=True)
class IntegrationResult:
    """Container for ODE integration trajectory."""

    t: np.ndarray  # shape: (n_steps + 1,)
    y: np.ndarray  # shape: (n_steps + 1, dim)


def _ensure_state_vector(y0: float | np.ndarray) -> State:
    arr = np.asarray(y0, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1:
        raise ValueError("Initial state must be scalar or 1-D array.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Initial state contains non-finite values.")
    return arr


def _eval_rhs(f: RhsFunc, t: float, y: State) -> State:
    out = np.asarray(f(float(t), y.copy()), dtype=float)
    if out.shape != y.shape:
        raise ValueError(f"RHS shape mismatch: expected {y.shape}, got {out.shape}.")
    if not np.all(np.isfinite(out)):
        raise ValueError("RHS evaluation produced non-finite values.")
    return out


def rk4_step(f: RhsFunc, t: float, y: State, h: float) -> State:
    """Single RK4 step: y_{n+1} = y_n + h/6 * (k1 + 2k2 + 2k3 + k4)."""
    k1 = _eval_rhs(f, t, y)
    k2 = _eval_rhs(f, t + 0.5 * h, y + 0.5 * h * k1)
    k3 = _eval_rhs(f, t + 0.5 * h, y + 0.5 * h * k2)
    k4 = _eval_rhs(f, t + h, y + h * k3)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rk4_integrate(
    f: RhsFunc,
    t0: float,
    y0: float | np.ndarray,
    t_end: float,
    n_steps: int,
) -> IntegrationResult:
    """Integrate y' = f(t, y) on [t0, t_end] with fixed-step RK4."""
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    if not (math.isfinite(t0) and math.isfinite(t_end)):
        raise ValueError("t0 and t_end must be finite.")
    if t_end <= t0:
        raise ValueError("Require t_end > t0 for this MVP.")

    y0_vec = _ensure_state_vector(y0)
    h = (t_end - t0) / float(n_steps)
    if not math.isfinite(h) or h <= 0.0:
        raise ValueError("Computed step size h must be finite and positive.")

    t_grid = np.linspace(t0, t_end, n_steps + 1, dtype=float)
    y_grid = np.empty((n_steps + 1, y0_vec.size), dtype=float)
    y_grid[0] = y0_vec

    t = t0
    y = y0_vec.copy()
    for i in range(n_steps):
        y = rk4_step(f, t, y, h)
        t = t_grid[i + 1]
        y_grid[i + 1] = y

    return IntegrationResult(t=t_grid, y=y_grid)


def scalar_rhs(t: float, y: State) -> State:
    """ODE: y' = y - t^2 + 1, y(0)=0.5."""
    return np.array([y[0] - t * t + 1.0], dtype=float)


def scalar_exact_solution(t: np.ndarray) -> np.ndarray:
    """Exact solution for scalar_rhs."""
    return (t + 1.0) ** 2 - 0.5 * np.exp(t)


def oscillator_rhs(_: float, y: State) -> State:
    """Harmonic oscillator: x' = v, v' = -x."""
    x, v = float(y[0]), float(y[1])
    return np.array([v, -x], dtype=float)


def observed_orders(errors: list[float]) -> list[float]:
    orders: list[float] = []
    for i in range(len(errors) - 1):
        e1, e2 = errors[i], errors[i + 1]
        if e1 <= 0.0 or e2 <= 0.0:
            orders.append(float("nan"))
        else:
            orders.append(math.log(e1 / e2, 2.0))
    return orders


def run_scalar_convergence_demo() -> None:
    t0, t_end, y0 = 0.0, 2.0, 0.5
    step_list = [10, 20, 40, 80]
    terminal_errors: list[float] = []

    print("=== Scalar ODE: RK4 convergence demo ===")
    print("ODE: y' = y - t^2 + 1, y(0)=0.5, interval=[0, 2]")
    print(f"{'steps':>8} {'h':>10} {'|error(T)|':>14}")

    for n_steps in step_list:
        result = rk4_integrate(scalar_rhs, t0, y0, t_end, n_steps)
        y_exact_t = scalar_exact_solution(np.array([t_end], dtype=float))[0]
        err = abs(float(result.y[-1, 0]) - float(y_exact_t))
        terminal_errors.append(err)
        h = (t_end - t0) / n_steps
        print(f"{n_steps:8d} {h:10.5f} {err:14.6e}")

    orders = observed_orders(terminal_errors)
    print("Observed order p from error ratio e(h)/e(h/2):")
    for i, p in enumerate(orders):
        print(f"  {step_list[i]:>3d} -> {step_list[i + 1]:>3d}: p = {p:.4f}")
    print()


def run_vector_demo() -> None:
    t0, t_end = 0.0, 2.0 * math.pi
    n_steps = 200
    y0 = np.array([0.0, 1.0], dtype=float)  # x(0)=0, v(0)=1

    result = rk4_integrate(oscillator_rhs, t0, y0, t_end, n_steps)
    x_num, v_num = result.y[:, 0], result.y[:, 1]

    x_exact = np.sin(result.t)
    v_exact = np.cos(result.t)
    max_state_error = np.max(np.sqrt((x_num - x_exact) ** 2 + (v_num - v_exact) ** 2))

    energy = 0.5 * (x_num * x_num + v_num * v_num)
    energy_drift = float(np.max(np.abs(energy - energy[0])))

    print("=== Vector ODE: harmonic oscillator demo ===")
    print("ODE: x' = v, v' = -x, interval=[0, 2*pi]")
    print(f"steps = {n_steps}, h = {(t_end - t0) / n_steps:.5f}")
    print(f"max state error      = {max_state_error:.6e}")
    print(f"max energy drift     = {energy_drift:.6e}")
    print(
        "final state (num)    = "
        f"[{x_num[-1]: .6f}, {v_num[-1]: .6f}] ; expected [0.000000, 1.000000]"
    )


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)
    run_scalar_convergence_demo()
    run_vector_demo()


if __name__ == "__main__":
    main()

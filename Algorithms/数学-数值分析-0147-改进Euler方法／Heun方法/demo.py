"""Minimal runnable MVP for Improved Euler (Heun) method."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class ODESolution:
    method: str
    t_values: np.ndarray
    y_values: np.ndarray
    exact_values: np.ndarray
    max_abs_error: float


def rhs(t: float, y: float) -> float:
    """Example ODE right-hand side: y' = y - t^2 + 1."""
    return y - t * t + 1.0


def exact_solution(t: np.ndarray) -> np.ndarray:
    """Exact solution for y' = y - t^2 + 1, y(0)=0.5."""
    return (t + 1.0) ** 2 - 0.5 * np.exp(t)


def check_inputs(t0: float, t_end: float, y0: float, n_steps: int) -> None:
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    if not np.isfinite(t0) or not np.isfinite(t_end) or not np.isfinite(y0):
        raise ValueError("t0, t_end, y0 must be finite")
    if t_end <= t0:
        raise ValueError("t_end must be greater than t0")


def euler_method(
    func: Callable[[float, float], float],
    t0: float,
    y0: float,
    t_end: float,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    check_inputs(t0=t0, t_end=t_end, y0=y0, n_steps=n_steps)

    h = (t_end - t0) / n_steps
    t_values = np.linspace(t0, t_end, n_steps + 1)
    y_values = np.empty(n_steps + 1, dtype=float)
    y_values[0] = y0

    for i in range(n_steps):
        slope = func(float(t_values[i]), float(y_values[i]))
        if not np.isfinite(slope):
            raise ValueError(f"non-finite slope at step {i}")
        y_values[i + 1] = y_values[i] + h * slope

    return t_values, y_values


def heun_method(
    func: Callable[[float, float], float],
    t0: float,
    y0: float,
    t_end: float,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    check_inputs(t0=t0, t_end=t_end, y0=y0, n_steps=n_steps)

    h = (t_end - t0) / n_steps
    t_values = np.linspace(t0, t_end, n_steps + 1)
    y_values = np.empty(n_steps + 1, dtype=float)
    y_values[0] = y0

    for i in range(n_steps):
        t_n = float(t_values[i])
        y_n = float(y_values[i])

        k1 = func(t_n, y_n)
        y_pred = y_n + h * k1
        k2 = func(t_n + h, y_pred)

        if not np.isfinite(k1) or not np.isfinite(k2):
            raise ValueError(f"non-finite slope at step {i}")

        y_values[i + 1] = y_n + 0.5 * h * (k1 + k2)

    return t_values, y_values


def build_solution(
    method: str,
    t_values: np.ndarray,
    y_values: np.ndarray,
    exact_func: Callable[[np.ndarray], np.ndarray],
) -> ODESolution:
    exact_values = exact_func(t_values)
    max_abs_error = float(np.max(np.abs(y_values - exact_values)))
    return ODESolution(
        method=method,
        t_values=t_values,
        y_values=y_values,
        exact_values=exact_values,
        max_abs_error=max_abs_error,
    )


def estimate_order(h_values: np.ndarray, err_values: np.ndarray) -> float:
    mask = (h_values > 0.0) & (err_values > 0.0) & np.isfinite(err_values)
    if int(np.count_nonzero(mask)) < 2:
        return float("nan")

    x = np.log(h_values[mask])
    y = np.log(err_values[mask])
    slope, _ = np.polyfit(x, y, deg=1)
    return float(slope)


def main() -> None:
    t0 = 0.0
    t_end = 2.0
    y0 = 0.5
    step_list = [10, 20, 40, 80, 160]

    h_values = []
    euler_errors = []
    heun_errors = []

    print("ODE: y' = y - t^2 + 1, y(0)=0.5, t in [0, 2]")
    print("-" * 74)
    print(f"{'N':>6} {'h':>12} {'Euler_max_err':>22} {'Heun_max_err':>22}")

    sample_heun: ODESolution | None = None

    for n_steps in step_list:
        t_euler, y_euler = euler_method(rhs, t0=t0, y0=y0, t_end=t_end, n_steps=n_steps)
        t_heun, y_heun = heun_method(rhs, t0=t0, y0=y0, t_end=t_end, n_steps=n_steps)

        euler_sol = build_solution("Euler", t_euler, y_euler, exact_solution)
        heun_sol = build_solution("Heun", t_heun, y_heun, exact_solution)

        h = (t_end - t0) / n_steps
        h_values.append(h)
        euler_errors.append(euler_sol.max_abs_error)
        heun_errors.append(heun_sol.max_abs_error)

        if n_steps == 20:
            sample_heun = heun_sol

        print(
            f"{n_steps:6d} {h:12.6f} "
            f"{euler_sol.max_abs_error:22.10e} {heun_sol.max_abs_error:22.10e}"
        )

    h_arr = np.array(h_values, dtype=float)
    euler_arr = np.array(euler_errors, dtype=float)
    heun_arr = np.array(heun_errors, dtype=float)

    euler_order = estimate_order(h_arr, euler_arr)
    heun_order = estimate_order(h_arr, heun_arr)

    print("-" * 74)
    print(f"Estimated order (Euler): {euler_order:.4f}")
    print(f"Estimated order (Heun) : {heun_order:.4f}")

    if sample_heun is not None:
        print("\nHeun trajectory preview (N=20, first 5 points):")
        print(f"{'t':>10} {'y_num':>16} {'y_exact':>16} {'abs_err':>16}")
        for i in range(5):
            t_i = float(sample_heun.t_values[i])
            y_num = float(sample_heun.y_values[i])
            y_ex = float(sample_heun.exact_values[i])
            err = abs(y_num - y_ex)
            print(f"{t_i:10.4f} {y_num:16.8f} {y_ex:16.8f} {err:16.8e}")


if __name__ == "__main__":
    main()

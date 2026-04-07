"""Minimal runnable MVP for Verlet integration (velocity-Verlet form)."""

from __future__ import annotations

import math
from typing import Callable, List, Sequence, Tuple

import numpy as np

Acceleration = Callable[[float, float], float]


def validate_scalar_inputs(t0: float, x0: float, v0: float, h: float, steps: int) -> None:
    """Validate scalar inputs for fixed-step second-order ODE integration."""
    for name, value in (("t0", t0), ("x0", x0), ("v0", v0), ("h", h)):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")
    if h <= 0.0:
        raise ValueError(f"h must be positive, got {h}")
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")


def require_integer_steps(t0: float, t_end: float, h: float) -> int:
    """Compute number of fixed steps and require grid alignment."""
    span = t_end - t0
    if span <= 0.0:
        raise ValueError(f"t_end must be greater than t0, got t0={t0}, t_end={t_end}")
    steps_float = span / h
    steps = int(round(steps_float))
    if abs(steps - steps_float) > 1e-12:
        raise ValueError("(t_end - t0) / h must be close to an integer in this demo")
    return steps


def sho_acceleration_factory(omega: float) -> Acceleration:
    """Return acceleration function a(x, t) = -omega^2 * x for SHO."""
    if not math.isfinite(omega) or omega <= 0.0:
        raise ValueError(f"omega must be positive and finite, got {omega!r}")

    omega2 = omega * omega

    def acc(x: float, _: float) -> float:
        return -omega2 * x

    return acc


def exact_sho_state(
    t_values: np.ndarray,
    x0: float,
    v0: float,
    omega: float,
    t0: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Exact state for x'' + omega^2 x = 0 with initial state at t0."""
    phase = omega * (t_values - t0)
    cos_term = np.cos(phase)
    sin_term = np.sin(phase)

    x_exact = x0 * cos_term + (v0 / omega) * sin_term
    v_exact = -x0 * omega * sin_term + v0 * cos_term
    return x_exact, v_exact


def velocity_verlet(
    acc: Acceleration,
    t0: float,
    x0: float,
    v0: float,
    h: float,
    steps: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate x'' = a(x, t) with velocity-Verlet."""
    validate_scalar_inputs(t0=t0, x0=x0, v0=v0, h=h, steps=steps)

    t = np.empty(steps + 1, dtype=float)
    x = np.empty(steps + 1, dtype=float)
    v = np.empty(steps + 1, dtype=float)

    t[0] = t0
    x[0] = x0
    v[0] = v0

    a_curr = acc(x0, t0)
    if not math.isfinite(a_curr):
        raise RuntimeError("non-finite acceleration at initial state")

    for n in range(steps):
        t_next = t[n] + h
        x_next = x[n] + h * v[n] + 0.5 * h * h * a_curr

        a_next = acc(float(x_next), float(t_next))
        if not math.isfinite(a_next):
            raise RuntimeError("non-finite acceleration encountered in Verlet iteration")

        v_next = v[n] + 0.5 * h * (a_curr + a_next)

        t[n + 1] = t_next
        x[n + 1] = x_next
        v[n + 1] = v_next
        a_curr = a_next

    return t, x, v


def explicit_euler_second_order(
    acc: Acceleration,
    t0: float,
    x0: float,
    v0: float,
    h: float,
    steps: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference baseline: explicit Euler on (x, v) first-order system."""
    validate_scalar_inputs(t0=t0, x0=x0, v0=v0, h=h, steps=steps)

    t = np.empty(steps + 1, dtype=float)
    x = np.empty(steps + 1, dtype=float)
    v = np.empty(steps + 1, dtype=float)

    t[0] = t0
    x[0] = x0
    v[0] = v0

    for n in range(steps):
        a_curr = acc(float(x[n]), float(t[n]))
        if not math.isfinite(a_curr):
            raise RuntimeError("non-finite acceleration encountered in Euler iteration")

        t[n + 1] = t[n] + h
        x[n + 1] = x[n] + h * v[n]
        v[n + 1] = v[n] + h * a_curr

    return t, x, v


def sho_energy(x: np.ndarray, v: np.ndarray, omega: float, mass: float = 1.0) -> np.ndarray:
    """Compute harmonic oscillator total energy on a trajectory."""
    return 0.5 * mass * v * v + 0.5 * mass * (omega * omega) * x * x


def run_verlet_accuracy_case(h: float, t_end: float, omega: float) -> Tuple[int, float, float]:
    """Run one fixed-step Verlet solve and report final/max position error."""
    t0 = 0.0
    x0 = 1.0
    v0 = 0.0
    steps = require_integer_steps(t0=t0, t_end=t_end, h=h)

    acc = sho_acceleration_factory(omega)
    t, x, _ = velocity_verlet(acc=acc, t0=t0, x0=x0, v0=v0, h=h, steps=steps)
    x_exact, _ = exact_sho_state(t_values=t, x0=x0, v0=v0, omega=omega, t0=t0)

    abs_err = np.abs(x - x_exact)
    final_abs_error = float(abs_err[-1])
    max_abs_error = float(np.max(abs_err))
    return steps, final_abs_error, max_abs_error


def estimate_orders(results: Sequence[Tuple[float, int, float, float]]) -> List[Tuple[float, float]]:
    """Estimate empirical order from consecutive max errors."""
    orders: List[Tuple[float, float]] = []
    for i in range(len(results) - 1):
        h1, _, _, e1 = results[i]
        h2, _, _, e2 = results[i + 1]
        if e1 <= 0.0 or e2 <= 0.0:
            continue
        p = math.log(e1 / e2) / math.log(h1 / h2)
        orders.append((h2, p))
    return orders


def run_long_horizon_comparison(long_h: float, long_steps: int, omega: float) -> None:
    """Compare Verlet vs explicit Euler on long-time error and energy drift."""
    t0 = 0.0
    x0 = 1.0
    v0 = 0.0

    acc = sho_acceleration_factory(omega)

    t_vv, x_vv, v_vv = velocity_verlet(
        acc=acc,
        t0=t0,
        x0=x0,
        v0=v0,
        h=long_h,
        steps=long_steps,
    )
    t_eu, x_eu, v_eu = explicit_euler_second_order(
        acc=acc,
        t0=t0,
        x0=x0,
        v0=v0,
        h=long_h,
        steps=long_steps,
    )

    x_ref_vv, _ = exact_sho_state(t_values=t_vv, x0=x0, v0=v0, omega=omega, t0=t0)
    x_ref_eu, _ = exact_sho_state(t_values=t_eu, x0=x0, v0=v0, omega=omega, t0=t0)

    err_vv = np.abs(x_vv - x_ref_vv)
    err_eu = np.abs(x_eu - x_ref_eu)

    e_vv = sho_energy(x=x_vv, v=v_vv, omega=omega)
    e_eu = sho_energy(x=x_eu, v=v_eu, omega=omega)

    rel_drift_vv = np.abs(e_vv - e_vv[0]) / e_vv[0]
    rel_drift_eu = np.abs(e_eu - e_eu[0]) / e_eu[0]

    print("=" * 96)
    print(f"Long-horizon comparison (h={long_h}, steps={long_steps}, t_end={long_h * long_steps:.1f})")
    print(" method          max_abs_pos_error      max_rel_energy_drift    final_rel_energy_drift")
    print("-" * 96)
    print(
        f" Verlet      {float(np.max(err_vv)):18.6e}      "
        f"{float(np.max(rel_drift_vv)):18.6e}      {float(rel_drift_vv[-1]):18.6e}"
    )
    print(
        f" Euler       {float(np.max(err_eu)):18.6e}      "
        f"{float(np.max(rel_drift_eu)):18.6e}      {float(rel_drift_eu[-1]):18.6e}"
    )


def print_trajectory_sample(h: float, omega: float, rows: int = 8) -> None:
    """Print early trajectory rows for quick sanity check."""
    t0 = 0.0
    t_end = 2.0
    x0 = 1.0
    v0 = 0.0

    steps = require_integer_steps(t0=t0, t_end=t_end, h=h)
    acc = sho_acceleration_factory(omega)

    t, x, v = velocity_verlet(acc=acc, t0=t0, x0=x0, v0=v0, h=h, steps=steps)
    x_exact, v_exact = exact_sho_state(t_values=t, x0=x0, v0=v0, omega=omega, t0=t0)

    print("=" * 96)
    print(f"Trajectory sample by velocity-Verlet (h={h}, first {rows} rows)")
    print(" n      t            x_num           v_num          x_exact         v_exact        abs_err_x")
    print("-" * 96)

    show = min(rows, len(t))
    for i in range(show):
        abs_err_x = abs(x[i] - x_exact[i])
        print(
            f"{i:2d}  {t[i]:8.4f}   {x[i]:13.8f}   {v[i]:13.8f}   "
            f"{x_exact[i]:13.8f}   {v_exact[i]:13.8f}   {abs_err_x:10.3e}"
        )


def main() -> None:
    omega = 1.0
    t_end = 20.0
    h_values = [0.2, 0.1, 0.05, 0.025]

    print("Velocity-Verlet demo on harmonic oscillator: x'' = -omega^2 * x")
    print(f"Initial state: x(0)=1, v(0)=0, omega={omega}")

    results: List[Tuple[float, int, float, float]] = []
    print("=" * 96)
    print(f"Convergence table on [0, {t_end}] (position error)")
    print(" h        steps    final_abs_error      max_abs_error")
    print("-" * 96)

    for h in h_values:
        steps, final_abs_error, max_abs_error = run_verlet_accuracy_case(
            h=h,
            t_end=t_end,
            omega=omega,
        )
        results.append((h, steps, final_abs_error, max_abs_error))
        print(f"{h:7.3f}  {steps:6d}   {final_abs_error:16.6e}   {max_abs_error:16.6e}")

    print("=" * 96)
    print("Empirical order (based on max_abs_error)")
    for h, p in estimate_orders(results):
        print(f"h={h:7.3f} -> estimated order p={p:.4f}")

    run_long_horizon_comparison(long_h=0.1, long_steps=20_000, omega=omega)
    print_trajectory_sample(h=0.1, omega=omega, rows=8)


if __name__ == "__main__":
    main()

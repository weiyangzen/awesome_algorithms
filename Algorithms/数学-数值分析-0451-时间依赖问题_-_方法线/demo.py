"""Method of Lines MVP for a time-dependent PDE.

Problem:
    u_t = alpha * u_xx, x in (0, 1), t > 0
    u(0, t) = u(1, t) = 0
    u(x, 0) = sin(pi x)

This script demonstrates:
1) Space discretization (finite differences) -> ODE system.
2) Time integration of the semi-discrete ODE (RK4, no black-box solver).
3) Validation against the analytic solution.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite, log, pi
from typing import List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class MOLConfig:
    """Configuration for a single Method of Lines run."""

    alpha: float
    n_space: int
    t_end: float
    dt: float
    x_left: float = 0.0
    x_right: float = 1.0


def validate_config(config: MOLConfig) -> None:
    """Validate scalar configuration values."""
    if config.n_space < 3:
        raise ValueError("n_space must be >= 3.")
    if config.t_end <= 0.0 or not isfinite(config.t_end):
        raise ValueError("t_end must be positive and finite.")
    if config.dt <= 0.0 or not isfinite(config.dt):
        raise ValueError("dt must be positive and finite.")
    if config.alpha <= 0.0 or not isfinite(config.alpha):
        raise ValueError("alpha must be positive and finite.")
    if not isfinite(config.x_left) or not isfinite(config.x_right):
        raise ValueError("Domain boundaries must be finite.")
    if config.x_right <= config.x_left:
        raise ValueError("x_right must be greater than x_left.")


def make_grid(config: MOLConfig) -> Tuple[np.ndarray, float]:
    """Build a uniform 1D spatial grid."""
    x = np.linspace(config.x_left, config.x_right, config.n_space, dtype=float)
    dx = float((config.x_right - config.x_left) / (config.n_space - 1))
    return x, dx


def initial_condition(x: np.ndarray) -> np.ndarray:
    """Initial state u(x, 0)."""
    return np.sin(pi * x)


def exact_solution(x: np.ndarray, t: float, alpha: float) -> np.ndarray:
    """Analytic solution for the selected benchmark."""
    return np.exp(-alpha * (pi**2) * t) * np.sin(pi * x)


def apply_laplacian_dirichlet_zero(state: np.ndarray, dx: float) -> np.ndarray:
    """Apply 1D Laplacian to interior nodes with zero Dirichlet boundaries.

    For interior vector y (corresponding to x_1 ... x_{n-2}):
        (Ly)_i = (y_{i-1} - 2y_i + y_{i+1}) / dx^2
    with virtual boundary values y_0 = y_{n-1} = 0.
    """
    n_interior = state.size
    if n_interior <= 0:
        raise ValueError("state must have at least one interior value.")
    if dx <= 0.0 or not isfinite(dx):
        raise ValueError("dx must be positive and finite.")

    lap = np.empty_like(state)
    inv_dx2 = 1.0 / (dx * dx)

    if n_interior == 1:
        lap[0] = -2.0 * state[0] * inv_dx2
        return lap

    lap[0] = (-2.0 * state[0] + state[1]) * inv_dx2
    lap[1:-1] = (state[:-2] - 2.0 * state[1:-1] + state[2:]) * inv_dx2
    lap[-1] = (state[-2] - 2.0 * state[-1]) * inv_dx2
    return lap


def semi_discrete_rhs(state: np.ndarray, dx: float, alpha: float) -> np.ndarray:
    """RHS of y' = alpha * L * y."""
    rhs = alpha * apply_laplacian_dirichlet_zero(state, dx)
    if not np.all(np.isfinite(rhs)):
        raise RuntimeError("Non-finite RHS encountered.")
    return rhs


def rk4_step(state: np.ndarray, dt: float, dx: float, alpha: float) -> np.ndarray:
    """One classic RK4 step for the semi-discrete ODE."""
    k1 = semi_discrete_rhs(state, dx, alpha)
    k2 = semi_discrete_rhs(state + 0.5 * dt * k1, dx, alpha)
    k3 = semi_discrete_rhs(state + 0.5 * dt * k2, dx, alpha)
    k4 = semi_discrete_rhs(state + dt * k3, dx, alpha)
    updated = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    if not np.all(np.isfinite(updated)):
        raise RuntimeError("Non-finite state encountered after RK4 update.")
    return updated


def integrate_mol(config: MOLConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate the Method of Lines system and return trajectory."""
    validate_config(config)
    x, dx = make_grid(config)

    u0 = initial_condition(x)
    y0 = u0[1:-1]  # interior unknowns
    n_interior = y0.size

    steps_float = config.t_end / config.dt
    steps = int(round(steps_float))
    if steps < 1:
        raise ValueError("At least one time step is required.")
    if not np.isclose(steps_float, steps, atol=1e-12, rtol=1e-12):
        raise ValueError("t_end must be an integer multiple of dt in this MVP.")

    t_values = np.linspace(0.0, config.t_end, steps + 1, dtype=float)
    y_history = np.empty((steps + 1, n_interior), dtype=float)
    y_history[0] = y0

    y = y0.copy()
    for idx in range(steps):
        y = rk4_step(y, config.dt, dx, config.alpha)
        y_history[idx + 1] = y

    return x, t_values, y_history


def reconstruct_full_state(y_interior: np.ndarray, n_space: int) -> np.ndarray:
    """Insert zero Dirichlet boundaries back to full grid state."""
    full = np.zeros(n_space, dtype=float)
    full[1:-1] = y_interior
    return full


def choose_stable_dt(alpha: float, dx: float, t_end: float, safety: float = 0.10) -> Tuple[float, int]:
    """Choose dt proportional to dx^2 and force exact grid alignment."""
    if safety <= 0.0:
        raise ValueError("safety must be positive.")
    target_dt = safety * dx * dx / alpha
    steps = max(1, int(ceil(t_end / target_dt)))
    dt = t_end / steps
    return dt, steps


def run_single_case(alpha: float, n_space: int, t_end: float) -> Tuple[float, int, float, float]:
    """Run one discretization level and return (dx, steps, max_err, l2_err)."""
    dx = 1.0 / (n_space - 1)
    dt, steps = choose_stable_dt(alpha=alpha, dx=dx, t_end=t_end, safety=0.10)
    config = MOLConfig(alpha=alpha, n_space=n_space, t_end=t_end, dt=dt)

    x, _, y_history = integrate_mol(config)
    u_num_final = reconstruct_full_state(y_history[-1], n_space)
    u_exact_final = exact_solution(x, t_end, alpha)
    err = np.abs(u_num_final - u_exact_final)

    max_err = float(np.max(err))
    l2_err = float(np.sqrt(np.mean(err * err)))
    return dx, steps, max_err, l2_err


def estimate_orders(dx_values: Sequence[float], error_values: Sequence[float]) -> List[float]:
    """Estimate convergence orders from consecutive (dx, error) pairs."""
    orders: List[float] = []
    for i in range(len(error_values) - 1):
        e1 = error_values[i]
        e2 = error_values[i + 1]
        h1 = dx_values[i]
        h2 = dx_values[i + 1]
        if e1 <= 0.0 or e2 <= 0.0:
            orders.append(float("nan"))
            continue
        orders.append(log(e1 / e2) / log(h1 / h2))
    return orders


def print_report(
    alpha: float,
    t_end: float,
    n_space_levels: Sequence[int],
    rows: Sequence[Tuple[float, int, float, float]],
    orders: Sequence[float],
) -> None:
    """Print a concise convergence report."""
    print("Method of Lines demo: 1D heat equation")
    print(f"alpha={alpha:.3f}, t_end={t_end:.3f}, boundary=Dirichlet(0,0)")
    print("")
    print("Convergence table at final time")
    print("n_space | dx        | steps | max_abs_error | l2_error")
    print("--------+-----------+-------+---------------+---------------")
    for n_space, (dx, steps, max_err, l2_err) in zip(n_space_levels, rows):
        print(f"{n_space:7d} | {dx:9.6f} | {steps:5d} | {max_err:13.6e} | {l2_err:13.6e}")

    if orders:
        print("")
        print("Observed orders from max_abs_error")
        for i, p in enumerate(orders):
            left = n_space_levels[i]
            right = n_space_levels[i + 1]
            print(f"{left:2d}->{right:2d}: p={p:.4f}")


def run_quality_checks(rows: Sequence[Tuple[float, int, float, float]], orders: Sequence[float]) -> None:
    """Basic sanity checks for this deterministic demo."""
    max_errors = [row[2] for row in rows]
    if not (max_errors[0] > max_errors[1] > max_errors[2]):
        raise AssertionError("Expected strictly decreasing max error with refinement.")

    if orders and min(orders) < 1.8:
        raise AssertionError(f"Observed order too low: min order={min(orders):.4f}")


def main() -> None:
    alpha = 1.0
    t_end = 0.1
    n_space_levels = [21, 41, 61]

    rows = [run_single_case(alpha=alpha, n_space=n, t_end=t_end) for n in n_space_levels]
    dx_values = [row[0] for row in rows]
    max_errors = [row[2] for row in rows]
    orders = estimate_orders(dx_values, max_errors)

    print_report(
        alpha=alpha,
        t_end=t_end,
        n_space_levels=n_space_levels,
        rows=rows,
        orders=orders,
    )
    run_quality_checks(rows, orders)
    print("")
    print("All checks passed.")


if __name__ == "__main__":
    main()

"""1D hyperbolic finite-difference MVP: explicit upwind for linear advection."""

from __future__ import annotations

import numpy as np


def initial_condition(x: np.ndarray) -> np.ndarray:
    """Smooth periodic initial profile."""
    gaussian = np.exp(-((x - 0.3) ** 2) / 0.005)
    sine = 0.15 * np.sin(2.0 * np.pi * x)
    return gaussian + sine


def upwind_step(u: np.ndarray, a: float, cfl: float) -> np.ndarray:
    """Advance one time step by first-order upwind scheme with periodic BC."""
    if a >= 0.0:
        return u - cfl * (u - np.roll(u, 1))
    return u - cfl * (np.roll(u, -1) - u)


def solve_advection_upwind(
    nx: int,
    a: float,
    t_end: float,
    cfl_target: float,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Solve u_t + a u_x = 0 on [0, 1) with periodic boundary."""
    if nx < 3:
        raise ValueError("nx must be >= 3.")
    if t_end <= 0.0:
        raise ValueError("t_end must be positive.")
    if abs(a) < 1e-14:
        raise ValueError("a must be non-zero for transport dynamics.")

    x = np.linspace(0.0, 1.0, nx, endpoint=False)
    u = initial_condition(x)

    dx = 1.0 / nx
    dt0 = cfl_target * dx / abs(a)
    n_steps = max(1, int(np.round(t_end / dt0)))
    dt = t_end / n_steps
    cfl = a * dt / dx
    if abs(cfl) > 1.0 + 1e-12:
        raise ValueError(f"Unstable CFL={cfl:.6f}. Need |CFL|<=1.")

    for _ in range(n_steps):
        u = upwind_step(u, a=a, cfl=cfl)

    return x, u, cfl, n_steps


def exact_solution_periodic(x: np.ndarray, t: float, a: float) -> np.ndarray:
    """Exact periodic solution: u(x,t) = u0((x - a t) mod 1)."""
    x_shift = (x - a * t) % 1.0
    return initial_condition(x_shift)


def compute_error_metrics(
    u_num: np.ndarray,
    u_exact: np.ndarray,
    dx: float,
) -> tuple[float, float, float]:
    """Return (L1, L2, Linf) errors."""
    err = u_num - u_exact
    l1 = float(np.sum(np.abs(err)) * dx)
    l2 = float(np.sqrt(np.sum(err**2) * dx))
    linf = float(np.max(np.abs(err)))
    return l1, l2, linf


def main() -> None:
    nx = 200
    a = 1.0
    t_end = 0.5
    cfl_target = 0.8

    x, u_num, cfl, n_steps = solve_advection_upwind(
        nx=nx,
        a=a,
        t_end=t_end,
        cfl_target=cfl_target,
    )
    u_exact = exact_solution_periodic(x, t=t_end, a=a)
    dx = 1.0 / nx
    l1, l2, linf = compute_error_metrics(u_num, u_exact, dx=dx)

    mass_num = float(np.sum(u_num) * dx)
    mass_exact = float(np.sum(initial_condition(x)) * dx)
    mass_error = mass_num - mass_exact

    print("=== Hyperbolic Finite Difference: Upwind (1D Advection) ===")
    print(f"nx={nx}, a={a}, t_end={t_end}, n_steps={n_steps}")
    print(f"target_cfl={cfl_target:.6f}, actual_cfl={cfl:.6f}")
    print(f"L1 error   = {l1:.6e}")
    print(f"L2 error   = {l2:.6e}")
    print(f"Linf error = {linf:.6e}")
    print(f"mass_error = {mass_error:.6e}")


if __name__ == "__main__":
    main()

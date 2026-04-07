"""1D Lax-Wendroff MVP for linear advection with periodic boundary."""

from __future__ import annotations

import numpy as np


def initial_condition(x: np.ndarray) -> np.ndarray:
    """Smooth periodic profile used as the reference initial condition."""
    gaussian = np.exp(-((x - 0.25) ** 2) / 0.004)
    cosine = 0.2 * np.cos(2.0 * np.pi * x)
    return gaussian + cosine


def lax_wendroff_step(u: np.ndarray, cfl: float) -> np.ndarray:
    """Advance one step by the second-order Lax-Wendroff scheme."""
    u_right = np.roll(u, -1)
    u_left = np.roll(u, 1)
    return (
        u
        - 0.5 * cfl * (u_right - u_left)
        + 0.5 * (cfl**2) * (u_right - 2.0 * u + u_left)
    )


def solve_advection_lax_wendroff(
    nx: int,
    a: float,
    t_end: float,
    cfl_target: float,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Solve u_t + a u_x = 0 on x in [0, 1) with periodic boundary."""
    if nx < 5:
        raise ValueError("nx must be >= 5.")
    if t_end <= 0.0:
        raise ValueError("t_end must be positive.")
    if abs(a) < 1e-14:
        raise ValueError("a must be non-zero for advection.")

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
        u = lax_wendroff_step(u, cfl=cfl)

    return x, u, cfl, n_steps


def exact_solution_periodic(x: np.ndarray, t: float, a: float) -> np.ndarray:
    """Exact periodic transport solution u(x,t)=u0((x-a t) mod 1)."""
    x_shift = (x - a * t) % 1.0
    return initial_condition(x_shift)


def compute_error_metrics(
    u_num: np.ndarray,
    u_exact: np.ndarray,
    dx: float,
) -> tuple[float, float, float]:
    """Return L1/L2/Linf error norms."""
    err = u_num - u_exact
    l1 = float(np.sum(np.abs(err)) * dx)
    l2 = float(np.sqrt(np.sum(err**2) * dx))
    linf = float(np.max(np.abs(err)))
    return l1, l2, linf


def run_case(nx: int, a: float, t_end: float, cfl_target: float) -> dict[str, float]:
    """Run one grid case and return key metrics."""
    x, u_num, cfl, n_steps = solve_advection_lax_wendroff(
        nx=nx,
        a=a,
        t_end=t_end,
        cfl_target=cfl_target,
    )
    u_exact = exact_solution_periodic(x=x, t=t_end, a=a)
    dx = 1.0 / nx
    l1, l2, linf = compute_error_metrics(u_num=u_num, u_exact=u_exact, dx=dx)

    mass0 = float(np.sum(initial_condition(x)) * dx)
    mass1 = float(np.sum(u_num) * dx)
    return {
        "nx": float(nx),
        "n_steps": float(n_steps),
        "cfl": cfl,
        "l1": l1,
        "l2": l2,
        "linf": linf,
        "mass_error": mass1 - mass0,
    }


def main() -> None:
    a = 1.0
    t_end = 0.4
    cfl_target = 0.8

    grid_sizes = [100, 200, 400]
    results = [run_case(nx=n, a=a, t_end=t_end, cfl_target=cfl_target) for n in grid_sizes]

    print("=== Lax-Wendroff Scheme: 1D Linear Advection ===")
    print(f"a={a}, t_end={t_end}, target_cfl={cfl_target}")
    print("nx    n_steps   actual_cfl      L1            L2            Linf         mass_error")
    for r in results:
        print(
            f"{int(r['nx']):<5d} {int(r['n_steps']):<8d} {r['cfl']:<13.6f} "
            f"{r['l1']:<13.6e} {r['l2']:<13.6e} {r['linf']:<13.6e} {r['mass_error']:<13.6e}"
        )

    if len(results) >= 2:
        print("\nEstimated convergence order (based on L2):")
        for i in range(len(results) - 1):
            e_coarse = results[i]["l2"]
            e_fine = results[i + 1]["l2"]
            order = np.log(e_coarse / e_fine) / np.log(2.0)
            nx0 = int(results[i]["nx"])
            nx1 = int(results[i + 1]["nx"])
            print(f"from nx={nx0} to nx={nx1}: p ≈ {order:.4f}")


if __name__ == "__main__":
    main()

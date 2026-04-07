"""Minimal runnable MVP for Lax-Friedrichs scheme (1D linear advection)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


ArrayFunc = Callable[[np.ndarray], np.ndarray]


@dataclass
class ResolutionResult:
    nx: int
    n_steps: int
    cfl: float
    l1: float
    l2: float
    linf: float
    mass_error: float


def ensure_finite_array(name: str, arr: np.ndarray) -> np.ndarray:
    """Validate all entries are finite."""
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def initial_condition_smooth(x: np.ndarray) -> np.ndarray:
    """Smooth periodic profile for convergence tests."""
    gaussian = np.exp(-((x - 0.30) ** 2) / 0.003)
    sine = 0.20 * np.sin(2.0 * np.pi * x)
    return gaussian + sine


def initial_condition_square(x: np.ndarray) -> np.ndarray:
    """Discontinuous profile to show numerical diffusion."""
    return np.where((x >= 0.2) & (x <= 0.5), 1.0, 0.0)


def linear_flux(u: np.ndarray, a: float) -> np.ndarray:
    """Flux f(u)=a*u for linear advection."""
    return a * u


def lax_friedrichs_step(
    u: np.ndarray,
    flux: ArrayFunc,
    dt: float,
    dx: float,
    alpha: float,
) -> np.ndarray:
    """One conservative Lax-Friedrichs step with periodic boundary."""
    if u.ndim != 1:
        raise ValueError("u must be a 1D array")
    if dt <= 0.0 or dx <= 0.0:
        raise ValueError("dt and dx must be positive")

    u_right = np.roll(u, -1)
    f_u = flux(u)
    f_right = flux(u_right)

    # Numerical interface flux F_{j+1/2}.
    interface_flux = 0.5 * (f_u + f_right) - 0.5 * alpha * (u_right - u)
    u_next = u - (dt / dx) * (interface_flux - np.roll(interface_flux, 1))

    return ensure_finite_array("u_next", u_next)


def solve_lax_friedrichs(
    nx: int,
    a: float,
    t_end: float,
    cfl_target: float,
    u0_func: ArrayFunc,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int, float]:
    """Solve u_t + (a u)_x = 0 on [0,1) with periodic boundary."""
    if nx < 8:
        raise ValueError("nx must be >= 8")
    if abs(a) < 1e-14:
        raise ValueError("a must be non-zero")
    if t_end <= 0.0:
        raise ValueError("t_end must be positive")
    if cfl_target <= 0.0:
        raise ValueError("cfl_target must be positive")

    x = np.linspace(0.0, 1.0, nx, endpoint=False)
    u0 = ensure_finite_array("u0", u0_func(x).astype(float))

    dx = 1.0 / nx
    dt0 = cfl_target * dx / abs(a)
    n_steps = max(1, int(np.ceil(t_end / dt0)))
    dt = t_end / n_steps
    cfl = a * dt / dx

    if abs(cfl) > 1.0 + 1e-12:
        raise ValueError(f"Unstable actual CFL={cfl:.6f}; require |CFL|<=1")

    alpha = dx / dt
    flux = lambda u: linear_flux(u, a=a)

    u = u0.copy()
    for _ in range(n_steps):
        u = lax_friedrichs_step(u=u, flux=flux, dt=dt, dx=dx, alpha=alpha)

    return x, u0, u, cfl, n_steps, dt


def exact_periodic_solution(x: np.ndarray, t: float, a: float, u0_func: ArrayFunc) -> np.ndarray:
    """Exact periodic solution u(x,t)=u0((x-a*t) mod 1)."""
    x_back = (x - a * t) % 1.0
    return ensure_finite_array("u_exact", u0_func(x_back).astype(float))


def error_norms(u_num: np.ndarray, u_exact: np.ndarray, dx: float) -> tuple[float, float, float]:
    """Compute L1/L2/Linf norms for the error vector."""
    err = u_num - u_exact
    l1 = float(np.sum(np.abs(err)) * dx)
    l2 = float(np.sqrt(np.sum(err**2) * dx))
    linf = float(np.max(np.abs(err)))
    return l1, l2, linf


def total_variation(u: np.ndarray) -> float:
    """Discrete total variation under periodic indexing."""
    return float(np.sum(np.abs(np.roll(u, -1) - u)))


def run_resolution_case(nx: int, a: float, t_end: float, cfl_target: float) -> ResolutionResult:
    """Run smooth-case simulation and return diagnostics."""
    x, u0, u_num, cfl, n_steps, _dt = solve_lax_friedrichs(
        nx=nx,
        a=a,
        t_end=t_end,
        cfl_target=cfl_target,
        u0_func=initial_condition_smooth,
    )
    dx = 1.0 / nx
    u_exact = exact_periodic_solution(x, t=t_end, a=a, u0_func=initial_condition_smooth)
    l1, l2, linf = error_norms(u_num=u_num, u_exact=u_exact, dx=dx)

    mass_error = float(np.sum(u_num - u0) * dx)
    return ResolutionResult(
        nx=nx,
        n_steps=n_steps,
        cfl=cfl,
        l1=l1,
        l2=l2,
        linf=linf,
        mass_error=mass_error,
    )


def convergence_order(err_coarse: float, err_fine: float, ratio: float) -> float:
    """Estimate empirical order p from err ~ h^p."""
    if err_coarse <= 0.0 or err_fine <= 0.0 or ratio <= 1.0:
        raise ValueError("invalid inputs for convergence order")
    return float(np.log(err_coarse / err_fine) / np.log(ratio))


def main() -> None:
    a = 1.0
    t_end = 0.4
    cfl_target = 0.85
    resolutions = [100, 200, 400]

    print("=== Lax-Friedrichs MVP: 1D Linear Advection ===")
    print(f"a={a}, t_end={t_end}, cfl_target={cfl_target}")

    results = [run_resolution_case(nx, a=a, t_end=t_end, cfl_target=cfl_target) for nx in resolutions]

    print("\nSmooth initial condition convergence table")
    print("nx | n_steps | actual_cfl | L1_error    | L2_error    | Linf_error  | mass_error")
    print("---+---------+------------+-------------+-------------+-------------+------------")
    for r in results:
        print(
            f"{r.nx:3d} | {r.n_steps:7d} | {r.cfl:10.6f} | {r.l1:11.4e} |"
            f" {r.l2:11.4e} | {r.linf:11.4e} | {r.mass_error:10.2e}"
        )

    p1 = convergence_order(results[0].l1, results[1].l1, ratio=2.0)
    p2 = convergence_order(results[1].l1, results[2].l1, ratio=2.0)
    print(f"\nEstimated order from L1: p(100->200)={p1:.3f}, p(200->400)={p2:.3f}")

    if p1 < 0.70 or p2 < 0.70:
        raise AssertionError("Observed order is too low for expected first-order behavior")

    # Extra diagnostic on discontinuous data: TV should not increase for this setup.
    nx_tv = 400
    x, u0_sq, u_sq, cfl_sq, n_steps_sq, _dt = solve_lax_friedrichs(
        nx=nx_tv,
        a=a,
        t_end=t_end,
        cfl_target=cfl_target,
        u0_func=initial_condition_square,
    )
    _ = x  # keep interface explicit for readability
    tv0 = total_variation(u0_sq)
    tvt = total_variation(u_sq)
    mass_error_sq = float(np.sum(u_sq - u0_sq) * (1.0 / nx_tv))

    print("\nDiscontinuous initial condition diagnostic")
    print(f"nx={nx_tv}, n_steps={n_steps_sq}, actual_cfl={cfl_sq:.6f}")
    print(f"TV0={tv0:.6f}, TVT={tvt:.6f}, TV_drop={tv0 - tvt:.6f}")
    print(f"mass_error={mass_error_sq:.2e}")

    if tvt > tv0 + 1e-10:
        raise AssertionError("Total variation increased unexpectedly")

    print("\nAll Lax-Friedrichs MVP checks passed.")


if __name__ == "__main__":
    main()

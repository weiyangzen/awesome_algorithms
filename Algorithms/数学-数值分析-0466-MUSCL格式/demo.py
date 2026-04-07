"""Minimal runnable MVP for MUSCL scheme (1D linear advection)."""

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


def ensure_finite(name: str, arr: np.ndarray) -> np.ndarray:
    """Ensure array has only finite values."""
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def initial_condition_smooth(x: np.ndarray) -> np.ndarray:
    """Smooth profile used for convergence tests."""
    gaussian = np.exp(-((x - 0.30) ** 2) / 0.003)
    sine = 0.20 * np.sin(2.0 * np.pi * x)
    return gaussian + sine


def initial_condition_square(x: np.ndarray) -> np.ndarray:
    """Discontinuous profile used to inspect TVD behavior."""
    return np.where((x >= 0.20) & (x <= 0.50), 1.0, 0.0)


def linear_flux(u: np.ndarray, a: float) -> np.ndarray:
    """Linear advection flux f(u)=a*u."""
    return a * u


def minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minmod limiter on two arrays."""
    same_sign = (a * b) > 0.0
    return np.where(same_sign, np.where(np.abs(a) < np.abs(b), a, b), 0.0)


def muscl_reconstruct(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct interface states (left/right) by MUSCL + minmod slope."""
    if u.ndim != 1:
        raise ValueError("u must be 1D")
    u_left = np.roll(u, 1)
    u_right = np.roll(u, -1)

    # Piecewise linear slope with TVD minmod limiting.
    slope = minmod(u - u_left, u_right - u)

    # Interface j+1/2: left state from cell j, right state from cell j+1.
    u_face_left = u + 0.5 * slope
    u_face_right = np.roll(u, -1) - 0.5 * np.roll(slope, -1)
    return u_face_left, u_face_right


def rusanov_flux_linear(u_l: np.ndarray, u_r: np.ndarray, a: float) -> np.ndarray:
    """Rusanov flux for scalar linear flux."""
    alpha = abs(a)
    f_l = linear_flux(u_l, a=a)
    f_r = linear_flux(u_r, a=a)
    return 0.5 * (f_l + f_r) - 0.5 * alpha * (u_r - u_l)


def muscl_spatial_operator(u: np.ndarray, dx: float, a: float) -> np.ndarray:
    """Semi-discrete operator du/dt = L(u) under periodic boundary."""
    if dx <= 0.0:
        raise ValueError("dx must be positive")
    u_l, u_r = muscl_reconstruct(u)
    flux = rusanov_flux_linear(u_l=u_l, u_r=u_r, a=a)
    return -(flux - np.roll(flux, 1)) / dx


def ssp_rk2_step(u: np.ndarray, dt: float, dx: float, a: float) -> np.ndarray:
    """One SSP-RK2 step for du/dt = L(u)."""
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    k1 = muscl_spatial_operator(u=u, dx=dx, a=a)
    u1 = u + dt * k1
    k2 = muscl_spatial_operator(u=u1, dx=dx, a=a)
    u_next = 0.5 * u + 0.5 * (u1 + dt * k2)
    return ensure_finite("u_next", u_next)


def solve_muscl(
    nx: int,
    a: float,
    t_end: float,
    cfl_target: float,
    u0_func: ArrayFunc,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int, float]:
    """Solve 1D advection by MUSCL finite volume + SSP-RK2."""
    if nx < 16:
        raise ValueError("nx must be >= 16")
    if abs(a) < 1e-14:
        raise ValueError("a must be non-zero")
    if t_end <= 0.0:
        raise ValueError("t_end must be positive")
    if cfl_target <= 0.0:
        raise ValueError("cfl_target must be positive")

    x = np.linspace(0.0, 1.0, nx, endpoint=False)
    u0 = ensure_finite("u0", u0_func(x).astype(float))
    u = u0.copy()

    dx = 1.0 / nx
    dt0 = cfl_target * dx / abs(a)
    n_steps = max(1, int(np.ceil(t_end / dt0)))
    dt = t_end / n_steps
    cfl = abs(a) * dt / dx

    # SSP-RK2 with TVD spatial discretization usually uses CFL <= 1 for scalar advection.
    if cfl > 1.0 + 1e-12:
        raise ValueError(f"Unstable actual CFL={cfl:.6f}; require CFL<=1")

    for _ in range(n_steps):
        u = ssp_rk2_step(u=u, dt=dt, dx=dx, a=a)

    return x, u0, u, cfl, n_steps, dt


def exact_periodic_solution(x: np.ndarray, t: float, a: float, u0_func: ArrayFunc) -> np.ndarray:
    """Exact periodic solution: u(x,t)=u0((x-a*t) mod 1)."""
    x_back = (x - a * t) % 1.0
    return ensure_finite("u_exact", u0_func(x_back).astype(float))


def error_norms(u_num: np.ndarray, u_exact: np.ndarray, dx: float) -> tuple[float, float, float]:
    """Compute L1/L2/Linf error norms."""
    err = u_num - u_exact
    l1 = float(np.sum(np.abs(err)) * dx)
    l2 = float(np.sqrt(np.sum(err**2) * dx))
    linf = float(np.max(np.abs(err)))
    return l1, l2, linf


def total_variation(u: np.ndarray) -> float:
    """Discrete periodic total variation."""
    return float(np.sum(np.abs(np.roll(u, -1) - u)))


def run_resolution_case(nx: int, a: float, t_end: float, cfl_target: float) -> ResolutionResult:
    """Run one smooth test case and return diagnostics."""
    x, u0, u_num, cfl, n_steps, _dt = solve_muscl(
        nx=nx,
        a=a,
        t_end=t_end,
        cfl_target=cfl_target,
        u0_func=initial_condition_smooth,
    )
    dx = 1.0 / nx
    u_exact = exact_periodic_solution(x=x, t=t_end, a=a, u0_func=initial_condition_smooth)
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
    """Estimate order p from err ~ h^p."""
    if err_coarse <= 0.0 or err_fine <= 0.0 or ratio <= 1.0:
        raise ValueError("invalid inputs for convergence order")
    return float(np.log(err_coarse / err_fine) / np.log(ratio))


def main() -> None:
    a = 1.0
    t_end = 0.4
    cfl_target = 0.8
    resolutions = [80, 160, 320]

    print("=== MUSCL MVP: 1D Linear Advection (Finite Volume) ===")
    print("Spatial reconstruction: MUSCL + minmod limiter")
    print("Time integrator: SSP-RK2")
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
    print(f"\nEstimated order from L1: p(80->160)={p1:.3f}, p(160->320)={p2:.3f}")

    if not (results[0].l1 > results[1].l1 > results[2].l1):
        raise AssertionError("L1 error did not decrease monotonically with refinement")
    if p1 < 0.90 or p2 < 0.90:
        raise AssertionError("Observed order is unexpectedly low for MUSCL with limiter")

    nx_tv = 400
    x, u0_sq, u_sq, cfl_sq, n_steps_sq, _dt = solve_muscl(
        nx=nx_tv,
        a=a,
        t_end=t_end,
        cfl_target=cfl_target,
        u0_func=initial_condition_square,
    )
    _ = x
    tv0 = total_variation(u0_sq)
    tvt = total_variation(u_sq)
    mass_error_sq = float(np.sum(u_sq - u0_sq) * (1.0 / nx_tv))

    print("\nDiscontinuous initial condition diagnostic")
    print(f"nx={nx_tv}, n_steps={n_steps_sq}, actual_cfl={cfl_sq:.6f}")
    print(f"TV0={tv0:.6f}, TVT={tvt:.6f}, TV_change={tvt - tv0:.6f}")
    print(f"mass_error={mass_error_sq:.2e}")

    # Allow tiny floating noise only.
    if tvt > tv0 + 1e-9:
        raise AssertionError("Total variation increased unexpectedly")

    print("\nAll MUSCL MVP checks passed.")


if __name__ == "__main__":
    main()

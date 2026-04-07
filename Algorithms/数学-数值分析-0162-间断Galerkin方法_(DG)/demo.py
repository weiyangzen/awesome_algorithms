"""1D linear advection solved by a minimal P1 discontinuous Galerkin (DG) MVP.

Equation:
    u_t + a u_x = 0,  x in [0, 1], periodic boundary.

Discretization:
- Space: DG with piecewise linear basis on each cell (P1).
- Numerical flux: Rusanov flux (equivalent to upwind for scalar linear advection).
- Time: SSP-RK3 explicit integrator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class DGConfig:
    num_cells: int
    advection_speed: float
    final_time: float
    cfl: float


@dataclass(frozen=True)
class DGResult:
    num_cells: int
    time_steps: int
    dt: float
    l2_error: float
    rel_l2_error: float
    mass_initial: float
    mass_final: float


def initial_condition(x: Array) -> Array:
    return np.sin(2.0 * np.pi * x)


def exact_solution(x: Array, t: float, a: float) -> Array:
    x_shift = np.mod(x - a * t, 1.0)
    return np.sin(2.0 * np.pi * x_shift)


def gauss_legendre_3() -> tuple[Array, Array]:
    xi = np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)], dtype=float)
    w = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0], dtype=float)
    return xi, w


def gauss_legendre_5() -> tuple[Array, Array]:
    xi = np.array(
        [
            -0.9061798459386640,
            -0.5384693101056831,
            0.0,
            0.5384693101056831,
            0.9061798459386640,
        ],
        dtype=float,
    )
    w = np.array(
        [
            0.2369268850561891,
            0.4786286704993665,
            0.5688888888888889,
            0.4786286704993665,
            0.2369268850561891,
        ],
        dtype=float,
    )
    return xi, w


def project_to_p1(num_cells: int, f: Callable[[Array], Array]) -> tuple[Array, float]:
    """L2-project f(x) onto local basis {1, xi} on each cell."""
    h = 1.0 / num_cells
    centers = (np.arange(num_cells, dtype=float) + 0.5) * h
    xi, w = gauss_legendre_3()

    xq = centers[:, None] + 0.5 * h * xi[None, :]
    uq = f(np.mod(xq, 1.0))

    c0 = 0.5 * np.sum(w[None, :] * uq, axis=1)
    c1 = 1.5 * np.sum(w[None, :] * uq * xi[None, :], axis=1)
    coeff = np.stack((c0, c1), axis=1)

    mass = h * np.sum(c0)
    return coeff, mass


def rusanov_flux(u_left: Array, u_right: Array, a: float) -> Array:
    return 0.5 * (a * u_left + a * u_right) - 0.5 * abs(a) * (u_right - u_left)


def dg_rhs(coeff: Array, a: float, h: float) -> Array:
    """Semi-discrete RHS for P1 DG in weak form with basis {1, xi}."""
    c0 = coeff[:, 0]
    c1 = coeff[:, 1]

    # Traces from each cell.
    u_left_trace = c0 - c1
    u_right_trace = c0 + c1

    # Interface j+1/2 uses right trace of cell j and left trace of cell j+1.
    flux_right = rusanov_flux(u_right_trace, np.roll(u_left_trace, -1), a)
    flux_left = np.roll(flux_right, 1)

    rhs0 = (flux_left - flux_right) / h
    rhs1 = (3.0 / h) * (2.0 * a * c0 - flux_right - flux_left)

    return np.stack((rhs0, rhs1), axis=1)


def ssp_rk3_step(coeff: Array, dt: float, rhs_func: Callable[[Array], Array]) -> Array:
    k1 = rhs_func(coeff)
    u1 = coeff + dt * k1

    k2 = rhs_func(u1)
    u2 = 0.75 * coeff + 0.25 * (u1 + dt * k2)

    k3 = rhs_func(u2)
    u_next = (1.0 / 3.0) * coeff + (2.0 / 3.0) * (u2 + dt * k3)
    return u_next


def l2_error(coeff: Array, t: float, a: float) -> tuple[float, float]:
    num_cells = coeff.shape[0]
    h = 1.0 / num_cells
    centers = (np.arange(num_cells, dtype=float) + 0.5) * h
    xi, w = gauss_legendre_5()

    xq = centers[:, None] + 0.5 * h * xi[None, :]
    uhq = coeff[:, 0][:, None] + coeff[:, 1][:, None] * xi[None, :]
    ueq = exact_solution(np.mod(xq, 1.0), t, a)

    diff2 = (uhq - ueq) ** 2
    exact2 = ueq**2

    err_sq = np.sum(0.5 * h * np.sum(w[None, :] * diff2, axis=1))
    ref_sq = np.sum(0.5 * h * np.sum(w[None, :] * exact2, axis=1))

    abs_l2 = float(np.sqrt(err_sq))
    rel_l2 = float(abs_l2 / np.sqrt(ref_sq)) if ref_sq > 0.0 else 0.0
    return abs_l2, rel_l2


def run_dg(config: DGConfig) -> DGResult:
    if config.num_cells < 4:
        raise ValueError("num_cells must be >= 4")
    if config.final_time <= 0.0:
        raise ValueError("final_time must be > 0")
    if config.cfl <= 0.0:
        raise ValueError("cfl must be > 0")
    if config.advection_speed == 0.0:
        raise ValueError("advection_speed must be non-zero")

    num_cells = config.num_cells
    a = config.advection_speed
    h = 1.0 / num_cells

    coeff, mass0 = project_to_p1(num_cells, initial_condition)

    # DG P1 explicit CFL scale: dt <= O(h / ((2p+1)|a|)), p=1 => factor 3.
    dt_cfl = config.cfl * h / (3.0 * abs(a))
    nsteps = int(np.ceil(config.final_time / dt_cfl))
    dt = config.final_time / nsteps

    rhs = lambda u: dg_rhs(u, a, h)
    for _ in range(nsteps):
        coeff = ssp_rk3_step(coeff, dt, rhs)

    l2_abs, l2_rel = l2_error(coeff, config.final_time, a)
    mass_final = h * np.sum(coeff[:, 0])

    return DGResult(
        num_cells=num_cells,
        time_steps=nsteps,
        dt=dt,
        l2_error=l2_abs,
        rel_l2_error=l2_rel,
        mass_initial=float(mass0),
        mass_final=float(mass_final),
    )


def convergence_rate(e_coarse: float, e_fine: float) -> float:
    if e_fine <= 0.0:
        return float("nan")
    return float(np.log(e_coarse / e_fine) / np.log(2.0))


def main() -> None:
    print("DG MVP: 1D linear advection u_t + a u_x = 0 on [0,1] with periodic BC")

    base_cfg = {
        "advection_speed": 1.0,
        "final_time": 0.5,
        "cfl": 0.18,
    }

    levels = [40, 80, 160]
    results: list[DGResult] = []

    for n in levels:
        cfg = DGConfig(num_cells=n, **base_cfg)
        result = run_dg(cfg)
        results.append(result)

    print(
        "{:<8} {:<8} {:<12} {:<14} {:<14} {:<14}".format(
            "cells", "steps", "dt", "L2_error", "rel_L2", "mass_drift"
        )
    )
    for r in results:
        mass_drift = r.mass_final - r.mass_initial
        print(
            "{:<8d} {:<8d} {:<12.4e} {:<14.6e} {:<14.6e} {:<14.6e}".format(
                r.num_cells,
                r.time_steps,
                r.dt,
                r.l2_error,
                r.rel_l2_error,
                mass_drift,
            )
        )

    if len(results) >= 2:
        print("\nObserved convergence rates (L2, refinement x2):")
        for i in range(1, len(results)):
            rc = results[i - 1]
            rf = results[i]
            rate = convergence_rate(rc.l2_error, rf.l2_error)
            print(f"N={rc.num_cells:>3d} -> N={rf.num_cells:>3d}: rate={rate:.3f}")


if __name__ == "__main__":
    main()

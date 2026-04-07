"""WENO/ENO minimal runnable MVP.

This demo solves 1D periodic linear advection:
    u_t + a u_x = 0
using finite-volume-style interface reconstruction plus SSP-RK3.

Implemented reconstructions:
- ENO3-like stencil picking (minimal-beta candidate)
- WENO5-JS nonlinear weighted reconstruction
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np


Array = np.ndarray


@dataclass
class SimulationSummary:
    method: str
    case_name: str
    n_cells: int
    cfl: float
    final_time: float
    l1_error: float
    linf_error: float
    total_variation: float
    u_min: float
    u_max: float


def initial_smooth(x: Array) -> Array:
    """Smooth periodic initial condition."""
    return np.sin(x) + 0.5 * np.sin(2.0 * x)


def initial_square(x: Array) -> Array:
    """Discontinuous periodic initial condition."""
    return np.where((x >= 0.5 * math.pi) & (x < 1.5 * math.pi), 1.0, 0.0)


def exact_periodic_shift(init_fn: Callable[[Array], Array], x: Array, a: float, t: float, period: float) -> Array:
    x0 = (x - a * t) % period
    return init_fn(x0)


def weno5_left_state(u: Array, eps: float = 1e-6, power: int = 2) -> Array:
    """Reconstruct left state at interfaces i+1/2 for all i (periodic)."""
    um2 = np.roll(u, 2)
    um1 = np.roll(u, 1)
    ui = u
    up1 = np.roll(u, -1)
    up2 = np.roll(u, -2)

    q0 = (1.0 / 3.0) * um2 - (7.0 / 6.0) * um1 + (11.0 / 6.0) * ui
    q1 = -(1.0 / 6.0) * um1 + (5.0 / 6.0) * ui + (1.0 / 3.0) * up1
    q2 = (1.0 / 3.0) * ui + (5.0 / 6.0) * up1 - (1.0 / 6.0) * up2

    beta0 = (13.0 / 12.0) * (um2 - 2.0 * um1 + ui) ** 2 + 0.25 * (um2 - 4.0 * um1 + 3.0 * ui) ** 2
    beta1 = (13.0 / 12.0) * (um1 - 2.0 * ui + up1) ** 2 + 0.25 * (um1 - up1) ** 2
    beta2 = (13.0 / 12.0) * (ui - 2.0 * up1 + up2) ** 2 + 0.25 * (3.0 * ui - 4.0 * up1 + up2) ** 2

    d0, d1, d2 = 0.1, 0.6, 0.3
    alpha0 = d0 / (eps + beta0) ** power
    alpha1 = d1 / (eps + beta1) ** power
    alpha2 = d2 / (eps + beta2) ** power
    alpha_sum = alpha0 + alpha1 + alpha2

    w0 = alpha0 / alpha_sum
    w1 = alpha1 / alpha_sum
    w2 = alpha2 / alpha_sum

    return w0 * q0 + w1 * q1 + w2 * q2


def eno3_like_left_state(u: Array) -> Array:
    """ENO3-like reconstruction by picking the smoothest candidate stencil.

    Note:
    - Classic ENO chooses stencils hierarchically by divided differences.
    - This MVP uses a simplified, transparent variant: build 3 third-order
      candidates then select the one with minimum smoothness indicator beta.
    """
    um2 = np.roll(u, 2)
    um1 = np.roll(u, 1)
    ui = u
    up1 = np.roll(u, -1)
    up2 = np.roll(u, -2)

    q0 = (1.0 / 3.0) * um2 - (7.0 / 6.0) * um1 + (11.0 / 6.0) * ui
    q1 = -(1.0 / 6.0) * um1 + (5.0 / 6.0) * ui + (1.0 / 3.0) * up1
    q2 = (1.0 / 3.0) * ui + (5.0 / 6.0) * up1 - (1.0 / 6.0) * up2

    beta0 = (13.0 / 12.0) * (um2 - 2.0 * um1 + ui) ** 2 + 0.25 * (um2 - 4.0 * um1 + 3.0 * ui) ** 2
    beta1 = (13.0 / 12.0) * (um1 - 2.0 * ui + up1) ** 2 + 0.25 * (um1 - up1) ** 2
    beta2 = (13.0 / 12.0) * (ui - 2.0 * up1 + up2) ** 2 + 0.25 * (3.0 * ui - 4.0 * up1 + up2) ** 2

    choose0 = (beta0 <= beta1) & (beta0 <= beta2)
    choose1 = (beta1 < beta0) & (beta1 <= beta2)

    return np.where(choose0, q0, np.where(choose1, q1, q2))


def spatial_operator(u: Array, dx: float, a: float, method: str) -> Array:
    if method == "weno5":
        u_half = weno5_left_state(u)
    elif method == "eno3":
        u_half = eno3_like_left_state(u)
    else:
        raise ValueError(f"unknown method: {method}")

    flux_iphalf = a * u_half
    flux_imhalf = np.roll(flux_iphalf, 1)
    return -(flux_iphalf - flux_imhalf) / dx


def rk3_step(u: Array, dt: float, dx: float, a: float, method: str) -> Array:
    l1 = spatial_operator(u, dx, a, method)
    u1 = u + dt * l1

    l2 = spatial_operator(u1, dx, a, method)
    u2 = 0.75 * u + 0.25 * (u1 + dt * l2)

    l3 = spatial_operator(u2, dx, a, method)
    return (1.0 / 3.0) * u + (2.0 / 3.0) * (u2 + dt * l3)


def solve_advection(u0: Array, dx: float, a: float, final_time: float, cfl: float, method: str) -> Array:
    if cfl <= 0.0:
        raise ValueError("cfl must be positive")
    if dx <= 0.0:
        raise ValueError("dx must be positive")
    if final_time < 0.0:
        raise ValueError("final_time must be non-negative")

    u = u0.copy()
    t = 0.0
    max_speed = abs(a)
    if max_speed == 0.0:
        return u

    base_dt = cfl * dx / max_speed
    if base_dt <= 0.0:
        raise ValueError("invalid dt from cfl/dx/speed")

    max_steps = 2_000_000
    step = 0
    while t < final_time - 1e-15:
        dt = min(base_dt, final_time - t)
        u = rk3_step(u, dt, dx, a, method)
        t += dt
        step += 1
        if step > max_steps:
            raise RuntimeError("step limit exceeded; check parameters")

    return u


def compute_total_variation(u: Array) -> float:
    return float(np.sum(np.abs(np.roll(u, -1) - u)))


def run_case(
    method: str,
    case_name: str,
    init_fn: Callable[[Array], Array],
    n_cells: int,
    domain_length: float,
    a: float,
    final_time: float,
    cfl: float,
) -> SimulationSummary:
    dx = domain_length / n_cells
    x = np.linspace(0.0, domain_length, n_cells, endpoint=False)

    u0 = init_fn(x)
    u_num = solve_advection(u0, dx=dx, a=a, final_time=final_time, cfl=cfl, method=method)
    u_ref = exact_periodic_shift(init_fn, x, a=a, t=final_time, period=domain_length)

    return SimulationSummary(
        method=method,
        case_name=case_name,
        n_cells=n_cells,
        cfl=cfl,
        final_time=final_time,
        l1_error=float(np.mean(np.abs(u_num - u_ref))),
        linf_error=float(np.max(np.abs(u_num - u_ref))),
        total_variation=compute_total_variation(u_num),
        u_min=float(np.min(u_num)),
        u_max=float(np.max(u_num)),
    )


def print_summary(summary: SimulationSummary) -> None:
    print(
        f"method={summary.method:>5s} | case={summary.case_name:>13s} | "
        f"N={summary.n_cells:4d} | CFL={summary.cfl:.2f} | T={summary.final_time:.3f}"
    )
    print(
        f"  L1={summary.l1_error:.6e} | Linf={summary.linf_error:.6e} | "
        f"TV={summary.total_variation:.6f} | min={summary.u_min:.6f} | max={summary.u_max:.6f}"
    )


def main() -> None:
    domain_length = 2.0 * math.pi
    a = 1.0  # positive speed for left-biased reconstruction
    final_time = 2.0 * math.pi  # one full period
    cfl = 0.45
    n_cells = 240

    cases: list[tuple[str, Callable[[Array], Array]]] = [
        ("smooth-sine", initial_smooth),
        ("discontinuous", initial_square),
    ]

    methods = ["eno3", "weno5"]

    print("WENO/ENO 1D periodic advection demo")
    print("Equation: u_t + a u_x = 0, finite-volume style + SSP-RK3")
    print("-" * 78)

    for case_name, init_fn in cases:
        print(f"Case: {case_name}")
        for method in methods:
            summary = run_case(
                method=method,
                case_name=case_name,
                init_fn=init_fn,
                n_cells=n_cells,
                domain_length=domain_length,
                a=a,
                final_time=final_time,
                cfl=cfl,
            )
            print_summary(summary)
        print("-" * 78)


if __name__ == "__main__":
    main()

"""Minimal DG(P1) MVP for 1D linear advection with periodic boundary.

Model:
    u_t + a u_x = 0,  x in [0, 1), t in [0, T]

Discretization:
- Spatial: discontinuous Galerkin, piecewise linear basis (P1)
- Flux: local Lax-Friedrichs / upwind for linear advection
- Time: SSP-RK3 explicit integrator

The script runs several mesh sizes and prints error metrics against exact solution.
"""

from __future__ import annotations

import numpy as np


def initial_condition(x: np.ndarray) -> np.ndarray:
    """Smooth periodic initial condition for convergence verification."""
    return np.sin(2.0 * np.pi * x) + 0.25 * np.cos(4.0 * np.pi * x)


def project_to_p1_dg(
    n_cells: int,
    x_left: float = 0.0,
    x_right: float = 1.0,
) -> tuple[np.ndarray, float, np.ndarray]:
    """L2-project initial condition to local P1 basis {1, xi} on each cell.

    Returns:
        coeffs: shape (n_cells, 2), local coefficients [u0, u1]
        h: cell size
        centers: cell centers
    """
    if n_cells < 4:
        raise ValueError("n_cells must be >= 4.")

    length = x_right - x_left
    h = length / n_cells
    centers = x_left + (np.arange(n_cells) + 0.5) * h

    # 4-point Gauss-Legendre quadrature on reference cell [-1, 1].
    xi_q, w_q = np.polynomial.legendre.leggauss(4)
    x_q = centers[:, None] + 0.5 * h * xi_q[None, :]

    # Periodic wrap to keep the initial-condition interface consistent.
    x_q_wrapped = ((x_q - x_left) % length) + x_left
    u_q = initial_condition(x_q_wrapped)

    # For basis phi0=1, phi1=xi:
    # u0 = (1/2) * integral u dxi, u1 = (3/2) * integral u*xi dxi.
    u0 = 0.5 * np.sum(u_q * w_q[None, :], axis=1)
    u1 = 1.5 * np.sum(u_q * xi_q[None, :] * w_q[None, :], axis=1)

    coeffs = np.column_stack((u0, u1))
    return coeffs, h, centers


def numerical_flux(u_left: np.ndarray, u_right: np.ndarray, a: float) -> np.ndarray:
    """Local Lax-Friedrichs flux (reduces to upwind for linear advection)."""
    return 0.5 * a * (u_left + u_right) - 0.5 * abs(a) * (u_right - u_left)


def dg_rhs(coeffs: np.ndarray, a: float, h: float) -> np.ndarray:
    """Compute semi-discrete DG right-hand side for P1 modal coefficients.

    coeffs[:, 0] is modal coefficient on basis phi0=1.
    coeffs[:, 1] is modal coefficient on basis phi1=xi.
    """
    u0 = coeffs[:, 0]
    u1 = coeffs[:, 1]

    # Traces at each cell boundary from inside the cell.
    u_right_trace = u0 + u1  # value at xi=+1
    u_left_trace = u0 - u1  # value at xi=-1

    # Interface j+1/2: left state from cell j right trace,
    # right state from cell j+1 left trace.
    state_left = u_right_trace
    state_right = np.roll(u_left_trace, -1)
    flux_right = numerical_flux(state_left, state_right, a)

    # For cell j, left interface is (j-1/2).
    flux_left = np.roll(flux_right, 1)

    rhs = np.empty_like(coeffs)
    # (h/2) * M * dU/dt = -a*S*U + boundary terms
    # With basis {1, xi}: M=diag(2, 2/3), S=[[0,0],[2,0]].
    rhs[:, 0] = -(flux_right - flux_left) / h
    rhs[:, 1] = (3.0 / h) * (2.0 * a * u0 - flux_right - flux_left)
    return rhs


def ssp_rk3_step(coeffs: np.ndarray, dt: float, a: float, h: float) -> np.ndarray:
    """Advance one time step by SSP-RK3."""
    k1 = dg_rhs(coeffs, a=a, h=h)
    u1 = coeffs + dt * k1

    k2 = dg_rhs(u1, a=a, h=h)
    u2 = 0.75 * coeffs + 0.25 * (u1 + dt * k2)

    k3 = dg_rhs(u2, a=a, h=h)
    return (1.0 / 3.0) * coeffs + (2.0 / 3.0) * (u2 + dt * k3)


def solve_dg_advection(
    n_cells: int,
    a: float,
    t_end: float,
    cfl_target: float,
    degree: int = 1,
) -> tuple[np.ndarray, float, np.ndarray, int, float]:
    """Solve 1D periodic advection by DG(P1)+SSP-RK3.

    Returns:
        coeffs: final modal coefficients
        h: cell size
        centers: cell centers
        n_steps: number of RK steps
        cfl_actual: effective DG CFL = |a|*dt/h*(2p+1)
    """
    if abs(a) < 1e-14:
        raise ValueError("a must be non-zero.")
    if t_end <= 0.0:
        raise ValueError("t_end must be positive.")
    if degree != 1:
        raise ValueError("This MVP currently implements degree=1 only.")

    coeffs, h, centers = project_to_p1_dg(n_cells)

    # DG explicit stability scaling: dt ~ h / ((2p+1)|a|).
    dt_guess = cfl_target * h / (abs(a) * (2 * degree + 1))
    n_steps = max(1, int(np.ceil(t_end / dt_guess)))
    dt = t_end / n_steps
    cfl_actual = abs(a) * dt / h * (2 * degree + 1)

    for _ in range(n_steps):
        coeffs = ssp_rk3_step(coeffs, dt=dt, a=a, h=h)

    return coeffs, h, centers, n_steps, cfl_actual


def error_metrics(
    coeffs: np.ndarray,
    h: float,
    centers: np.ndarray,
    a: float,
    t_end: float,
    x_left: float = 0.0,
    x_right: float = 1.0,
) -> tuple[float, float, float]:
    """Compute L1/L2/Linf errors by quadrature inside each cell."""
    xi_q, w_q = np.polynomial.legendre.leggauss(8)

    u_num = coeffs[:, 0][:, None] + coeffs[:, 1][:, None] * xi_q[None, :]
    x_q = centers[:, None] + 0.5 * h * xi_q[None, :]

    length = x_right - x_left
    x_shift = ((x_q - a * t_end - x_left) % length) + x_left
    u_ex = initial_condition(x_shift)

    err = u_num - u_ex
    jac = 0.5 * h

    l1 = float(np.sum(np.abs(err) * w_q[None, :]) * jac)
    l2 = float(np.sqrt(np.sum((err**2) * w_q[None, :]) * jac))
    linf = float(np.max(np.abs(err)))
    return l1, l2, linf


def mass_from_coeffs(coeffs: np.ndarray, h: float) -> float:
    """Total mass integral for DG(P1): integral u dx = sum_j h*u0_j."""
    return float(h * np.sum(coeffs[:, 0]))


def run_case(n_cells: int, a: float, t_end: float, cfl_target: float) -> dict[str, float]:
    coeffs_init, h, centers = project_to_p1_dg(n_cells)
    mass0 = mass_from_coeffs(coeffs_init, h)

    coeffs, h, centers, n_steps, cfl_actual = solve_dg_advection(
        n_cells=n_cells,
        a=a,
        t_end=t_end,
        cfl_target=cfl_target,
    )

    l1, l2, linf = error_metrics(coeffs=coeffs, h=h, centers=centers, a=a, t_end=t_end)
    mass1 = mass_from_coeffs(coeffs, h)

    return {
        "n_cells": float(n_cells),
        "n_steps": float(n_steps),
        "cfl": cfl_actual,
        "l1": l1,
        "l2": l2,
        "linf": linf,
        "mass_error": mass1 - mass0,
    }


def main() -> None:
    a = 1.0
    t_end = 0.3
    cfl_target = 0.25
    grid_sizes = [40, 80, 160]

    results = [run_case(n, a=a, t_end=t_end, cfl_target=cfl_target) for n in grid_sizes]

    print("=== DG(P1) for 1D Linear Advection (Periodic) ===")
    print(f"a={a}, t_end={t_end}, cfl_target={cfl_target}")
    print("cells  steps    cfl(actual)    L1            L2            Linf          mass_error")

    for r in results:
        print(
            f"{int(r['n_cells']):<6d} {int(r['n_steps']):<8d} {r['cfl']:<14.6f} "
            f"{r['l1']:<13.6e} {r['l2']:<13.6e} {r['linf']:<13.6e} {r['mass_error']:<13.6e}"
        )

    print("\nEstimated convergence order (L2):")
    for i in range(len(results) - 1):
        e_coarse = results[i]["l2"]
        e_fine = results[i + 1]["l2"]
        order = np.log(e_coarse / e_fine) / np.log(2.0)
        n0 = int(results[i]["n_cells"])
        n1 = int(results[i + 1]["n_cells"])
        print(f"from {n0} to {n1}: p ~= {order:.4f}")


if __name__ == "__main__":
    main()

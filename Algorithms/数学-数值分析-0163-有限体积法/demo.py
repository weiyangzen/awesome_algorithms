"""Finite Volume Method (FVM) MVP for 1D steady diffusion.

Model problem:
    -d/dx(k du/dx) = q(x), x in [0, 1]
    u(0) = u_L, u(1) = u_R

This demo uses:
    k = 1, u_L = u_R = 0,
    exact solution u*(x) = sin(pi x),
    q(x) = pi^2 sin(pi x).

Discretization:
- Cell-centered finite volume on a uniform mesh.
- Two-point flux approximation for diffusion fluxes.
- Dirichlet boundaries enforced via half-cell distance at boundary faces.
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np


def exact_solution(x: np.ndarray) -> np.ndarray:
    """Analytical solution used for verification."""
    return np.sin(math.pi * x)


def source_term(x: np.ndarray) -> np.ndarray:
    """Source q(x) corresponding to u*=sin(pi x) for k=1."""
    return (math.pi**2) * np.sin(math.pi * x)


def build_fvm_system(
    n_cells: int,
    kappa: float,
    u_left: float,
    u_right: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Assemble A u = b for cell-centered FVM on [0,1]."""
    if n_cells < 2:
        raise ValueError("n_cells must be >= 2")
    if kappa <= 0.0:
        raise ValueError("kappa must be positive")

    dx = 1.0 / n_cells
    xc = (np.arange(n_cells, dtype=float) + 0.5) * dx

    a = np.zeros((n_cells, n_cells), dtype=float)
    b = source_term(xc) * dx

    for i in range(n_cells):
        if i == 0:
            a_e = kappa / dx
            a_wb = 2.0 * kappa / dx  # boundary face uses half-cell distance
            a[i, i] = a_e + a_wb
            a[i, i + 1] = -a_e
            b[i] += a_wb * u_left
        elif i == n_cells - 1:
            a_w = kappa / dx
            a_eb = 2.0 * kappa / dx
            a[i, i] = a_w + a_eb
            a[i, i - 1] = -a_w
            b[i] += a_eb * u_right
        else:
            a_w = kappa / dx
            a_e = kappa / dx
            a[i, i] = a_w + a_e
            a[i, i - 1] = -a_w
            a[i, i + 1] = -a_e

    return a, b, xc, dx


def compute_face_fluxes(
    u: np.ndarray,
    kappa: float,
    dx: float,
    u_left: float,
    u_right: float,
) -> np.ndarray:
    """Compute diffusive face flux F=-k du/dx at all faces (size n_cells+1)."""
    n_cells = u.size
    flux = np.zeros(n_cells + 1, dtype=float)

    flux[0] = -kappa * (u[0] - u_left) / (0.5 * dx)

    for face in range(1, n_cells):
        i_w = face - 1
        i_e = face
        flux[face] = -kappa * (u[i_e] - u[i_w]) / dx

    flux[-1] = -kappa * (u_right - u[-1]) / (0.5 * dx)
    return flux


def cell_balance_residuals(
    u: np.ndarray,
    xc: np.ndarray,
    kappa: float,
    dx: float,
    u_left: float,
    u_right: float,
) -> np.ndarray:
    """Conservation residual per cell: (F_e - F_w) - Q_i."""
    flux = compute_face_fluxes(u, kappa, dx, u_left, u_right)
    q_cell = source_term(xc) * dx
    residual = (flux[1:] - flux[:-1]) - q_cell
    return residual


def run_case(n_cells: int) -> Dict[str, float]:
    """Solve one mesh size and return error/conservation metrics."""
    kappa = 1.0
    u_left = 0.0
    u_right = 0.0

    a, b, xc, dx = build_fvm_system(n_cells, kappa, u_left, u_right)
    u_num = np.linalg.solve(a, b)

    u_ref = exact_solution(xc)
    rel_l2 = float(np.linalg.norm(u_num - u_ref) / np.linalg.norm(u_ref))

    balance = cell_balance_residuals(u_num, xc, kappa, dx, u_left, u_right)
    max_balance = float(np.max(np.abs(balance)))

    return {
        "n_cells": float(n_cells),
        "dx": dx,
        "relative_l2_error": rel_l2,
        "max_balance_residual": max_balance,
    }


def print_results_table(results: List[Dict[str, float]]) -> None:
    """Pretty print mesh convergence results."""
    print("Finite Volume Method MVP: 1D steady diffusion")
    print("PDE: -d/dx(k du/dx)=q(x), k=1, u(0)=u(1)=0")
    print("Exact: u(x)=sin(pi x), q(x)=pi^2 sin(pi x)")
    print()

    header = (
        f"{'N':>6} {'dx':>12} {'relative_l2_error':>20} "
        f"{'max_balance_residual':>24} {'observed_order':>16}"
    )
    print(header)
    print("-" * len(header))

    prev_error = None
    for item in results:
        err = item["relative_l2_error"]
        order = float("nan")
        if prev_error is not None:
            order = math.log(prev_error / err, 2.0)
        order_text = f"{order:16.6f}" if math.isfinite(order) else f"{'-':>16}"

        print(
            f"{int(item['n_cells']):6d} {item['dx']:12.6e} "
            f"{err:20.6e} {item['max_balance_residual']:24.6e} {order_text}"
        )
        prev_error = err


def main() -> None:
    mesh_list = [10, 20, 40, 80, 160]
    results = [run_case(n) for n in mesh_list]
    print_results_table(results)


if __name__ == "__main__":
    main()

"""Minimal runnable MVP for 1D Euler equations (compressible gas dynamics).

Model (conservative form):
    U = [rho, rho*u, E]^T
    dU/dt + dF(U)/dx = 0

with ideal-gas EOS:
    p = (gamma - 1) * (E - 0.5 * rho * u^2)

This script solves the Sod shock-tube problem using a first-order
finite-volume method with Rusanov (local Lax-Friedrichs) numerical flux.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SolverConfig:
    gamma: float = 1.4
    cfl: float = 0.45
    n_cells: int = 400
    x_min: float = 0.0
    x_max: float = 1.0
    t_final: float = 0.2
    x_discontinuity: float = 0.5

    # Sod shock-tube initial condition.
    rho_left: float = 1.0
    u_left: float = 0.0
    p_left: float = 1.0

    rho_right: float = 0.125
    u_right: float = 0.0
    p_right: float = 0.1

    rho_floor: float = 1e-8
    p_floor: float = 1e-8


def primitive_from_conservative(
    U: np.ndarray, gamma: float, rho_floor: float, p_floor: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert conservative variables to primitive variables (rho, u, p)."""
    rho = np.maximum(U[:, 0], rho_floor)
    mom = U[:, 1]
    E = U[:, 2]

    u = mom / rho
    kinetic = 0.5 * rho * u * u
    p = (gamma - 1.0) * (E - kinetic)
    p = np.maximum(p, p_floor)
    return rho, u, p


def physical_flux(U: np.ndarray, cfg: SolverConfig) -> np.ndarray:
    """Physical flux F(U) for the 1D Euler equations."""
    rho, u, p = primitive_from_conservative(U, cfg.gamma, cfg.rho_floor, cfg.p_floor)
    E = U[:, 2]

    f1 = rho * u
    f2 = rho * u * u + p
    f3 = u * (E + p)
    return np.column_stack((f1, f2, f3))


def rusanov_flux(U_left: np.ndarray, U_right: np.ndarray, cfg: SolverConfig) -> np.ndarray:
    """Rusanov (local Lax-Friedrichs) interface flux."""
    FL = physical_flux(U_left, cfg)
    FR = physical_flux(U_right, cfg)

    rho_L, u_L, p_L = primitive_from_conservative(
        U_left, cfg.gamma, cfg.rho_floor, cfg.p_floor
    )
    rho_R, u_R, p_R = primitive_from_conservative(
        U_right, cfg.gamma, cfg.rho_floor, cfg.p_floor
    )

    a_L = np.sqrt(cfg.gamma * p_L / rho_L)
    a_R = np.sqrt(cfg.gamma * p_R / rho_R)
    s_max = np.maximum(np.abs(u_L) + a_L, np.abs(u_R) + a_R)

    return 0.5 * (FL + FR) - 0.5 * s_max[:, None] * (U_right - U_left)


def max_wave_speed(U: np.ndarray, cfg: SolverConfig) -> float:
    """Global max characteristic speed max(|u| + a)."""
    rho, u, p = primitive_from_conservative(U, cfg.gamma, cfg.rho_floor, cfg.p_floor)
    a = np.sqrt(cfg.gamma * p / rho)
    return float(np.max(np.abs(u) + a))


def apply_physical_floors(U: np.ndarray, cfg: SolverConfig) -> np.ndarray:
    """Enforce rho > 0 and p > 0 by minimal conservative correction."""
    rho = np.maximum(U[:, 0], cfg.rho_floor)
    mom = U[:, 1]
    E = U[:, 2]

    u = mom / rho
    kinetic = 0.5 * rho * u * u
    E_min = kinetic + cfg.p_floor / (cfg.gamma - 1.0)
    E = np.maximum(E, E_min)

    return np.column_stack((rho, mom, E))


def finite_volume_step(U: np.ndarray, dx: float, dt: float, cfg: SolverConfig) -> np.ndarray:
    """One first-order finite-volume step with outflow boundary conditions."""
    # Zero-gradient outflow BC via ghost-cell replication.
    U_ext = np.vstack((U[0:1, :], U, U[-1:, :]))

    F_int = rusanov_flux(U_ext[:-1, :], U_ext[1:, :], cfg)
    U_new = U - (dt / dx) * (F_int[1:, :] - F_int[:-1, :])
    return apply_physical_floors(U_new, cfg)


def initial_condition(x: np.ndarray, cfg: SolverConfig) -> np.ndarray:
    """Sod shock-tube initial condition in conservative variables."""
    left = x < cfg.x_discontinuity

    rho = np.where(left, cfg.rho_left, cfg.rho_right)
    u = np.where(left, cfg.u_left, cfg.u_right)
    p = np.where(left, cfg.p_left, cfg.p_right)

    E = p / (cfg.gamma - 1.0) + 0.5 * rho * u * u
    return np.column_stack((rho, rho * u, E))


def simulate(cfg: SolverConfig) -> tuple[np.ndarray, np.ndarray]:
    """Run the finite-volume simulation until t_final."""
    edges = np.linspace(cfg.x_min, cfg.x_max, cfg.n_cells + 1)
    x = 0.5 * (edges[:-1] + edges[1:])
    dx = (cfg.x_max - cfg.x_min) / cfg.n_cells

    U = initial_condition(x, cfg)

    t = 0.0
    while t < cfg.t_final:
        speed = max_wave_speed(U, cfg)
        dt = cfg.cfl * dx / max(speed, 1e-12)
        if t + dt > cfg.t_final:
            dt = cfg.t_final - t

        U = finite_volume_step(U, dx, dt, cfg)
        t += dt

    return x, U


def save_csv(path: Path, x: np.ndarray, U: np.ndarray, cfg: SolverConfig) -> None:
    """Save x, rho, u, p, E to CSV."""
    rho, u, p = primitive_from_conservative(U, cfg.gamma, cfg.rho_floor, cfg.p_floor)
    E = U[:, 2]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "rho", "u", "p", "E"])
        for xi, rhoi, ui, pi, Ei in zip(x, rho, u, p, E):
            writer.writerow(
                [f"{xi:.8f}", f"{rhoi:.8f}", f"{ui:.8f}", f"{pi:.8f}", f"{Ei:.8f}"]
            )


def main() -> None:
    cfg = SolverConfig()

    x, U0 = None, None
    # Keep a copy of initial state for conservation diagnostics.
    edges = np.linspace(cfg.x_min, cfg.x_max, cfg.n_cells + 1)
    x0 = 0.5 * (edges[:-1] + edges[1:])
    U0 = initial_condition(x0, cfg)

    x, U = simulate(cfg)

    out_path = Path(__file__).with_name("result.csv")
    save_csv(out_path, x, U, cfg)

    dx = (cfg.x_max - cfg.x_min) / cfg.n_cells
    mass0 = float(np.sum(U0[:, 0]) * dx)
    mass1 = float(np.sum(U[:, 0]) * dx)
    energy0 = float(np.sum(U0[:, 2]) * dx)
    energy1 = float(np.sum(U[:, 2]) * dx)

    rho, u, p = primitive_from_conservative(U, cfg.gamma, cfg.rho_floor, cfg.p_floor)

    print("欧拉方程 (Euler Equations) MVP 运行完成")
    print(f"网格数: {cfg.n_cells}, 终止时间: {cfg.t_final}")
    print(f"密度范围: [{np.min(rho):.6f}, {np.max(rho):.6f}]")
    print(f"压力范围: [{np.min(p):.6f}, {np.max(p):.6f}]")
    print(f"速度范围: [{np.min(u):.6f}, {np.max(u):.6f}]")
    print(f"质量守恒误差: {abs(mass1 - mass0):.6e}")
    print(f"总能守恒误差: {abs(energy1 - energy0):.6e}")
    print(f"结果文件: {out_path}")


if __name__ == "__main__":
    main()

"""Roe approximate Riemann solver MVP for 1D shallow-water equations.

Model:
    U = [h, hu]^T
    F(U) = [hu, hu^2 / h + 0.5 * g * h^2]^T

This script runs a dam-break test and writes results to `result.csv`.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SolverConfig:
    g: float = 9.81
    cfl: float = 0.45
    n_cells: int = 400
    x_min: float = 0.0
    x_max: float = 1.0
    t_final: float = 0.15
    h_left: float = 2.0
    h_right: float = 1.0
    velocity_left: float = 0.0
    velocity_right: float = 0.0
    h_floor: float = 1e-8
    entropy_eps: float = 1e-2


def flux(U: np.ndarray, g: float, h_floor: float) -> np.ndarray:
    """Physical flux F(U) for shallow-water equations."""
    h = np.maximum(U[:, 0], h_floor)
    m = U[:, 1]
    u = m / h
    return np.column_stack((m, m * u + 0.5 * g * h * h))


def entropy_fix_abs(lam: np.ndarray, eps: float) -> np.ndarray:
    """Harten entropy fix for absolute eigenvalue."""
    abs_lam = np.abs(lam)
    mask = abs_lam < eps
    abs_lam[mask] = 0.5 * (lam[mask] * lam[mask] / eps + eps)
    return abs_lam


def roe_flux(U_left: np.ndarray, U_right: np.ndarray, cfg: SolverConfig) -> np.ndarray:
    """Roe flux at each interface for vectorized left/right states."""
    h_floor = cfg.h_floor
    g = cfg.g

    hL = np.maximum(U_left[:, 0], h_floor)
    hR = np.maximum(U_right[:, 0], h_floor)
    mL = U_left[:, 1]
    mR = U_right[:, 1]

    uL = mL / hL
    uR = mR / hR

    FL = flux(U_left, g, h_floor)
    FR = flux(U_right, g, h_floor)

    sqrt_hL = np.sqrt(hL)
    sqrt_hR = np.sqrt(hR)
    denom = np.maximum(sqrt_hL + sqrt_hR, h_floor)

    u_tilde = (sqrt_hL * uL + sqrt_hR * uR) / denom
    h_tilde = 0.5 * (hL + hR)
    c_tilde = np.sqrt(np.maximum(g * h_tilde, h_floor))

    lam1 = u_tilde - c_tilde
    lam2 = u_tilde + c_tilde
    abs_lam1 = entropy_fix_abs(lam1, cfg.entropy_eps)
    abs_lam2 = entropy_fix_abs(lam2, cfg.entropy_eps)

    dU = U_right - U_left
    dh = dU[:, 0]
    dm = dU[:, 1]

    two_c = np.maximum(2.0 * c_tilde, h_floor)
    alpha1 = ((u_tilde + c_tilde) * dh - dm) / two_c
    alpha2 = (dm - (u_tilde - c_tilde) * dh) / two_c

    r1_0 = np.ones_like(u_tilde)
    r1_1 = u_tilde - c_tilde
    r2_0 = np.ones_like(u_tilde)
    r2_1 = u_tilde + c_tilde

    diss_0 = abs_lam1 * alpha1 * r1_0 + abs_lam2 * alpha2 * r2_0
    diss_1 = abs_lam1 * alpha1 * r1_1 + abs_lam2 * alpha2 * r2_1
    diss = np.column_stack((diss_0, diss_1))

    return 0.5 * (FL + FR) - 0.5 * diss


def max_wave_speed(U: np.ndarray, cfg: SolverConfig) -> float:
    h = np.maximum(U[:, 0], cfg.h_floor)
    u = U[:, 1] / h
    c = np.sqrt(cfg.g * h)
    return float(np.max(np.abs(u) + c))


def finite_volume_step(U: np.ndarray, dx: float, dt: float, cfg: SolverConfig) -> np.ndarray:
    """One first-order Godunov step with Roe interface fluxes."""
    # Outflow BC by ghost-cell extrapolation.
    U_ext = np.vstack((U[0:1, :], U, U[-1:, :]))
    F_int = roe_flux(U_ext[:-1, :], U_ext[1:, :], cfg)
    return U - (dt / dx) * (F_int[1:, :] - F_int[:-1, :])


def initial_condition(x: np.ndarray, cfg: SolverConfig) -> np.ndarray:
    h = np.where(x < 0.5 * (cfg.x_min + cfg.x_max), cfg.h_left, cfg.h_right)
    u = np.where(x < 0.5 * (cfg.x_min + cfg.x_max), cfg.velocity_left, cfg.velocity_right)
    return np.column_stack((h, h * u))


def simulate(cfg: SolverConfig) -> tuple[np.ndarray, np.ndarray]:
    n = cfg.n_cells
    x = np.linspace(cfg.x_min, cfg.x_max, n, endpoint=False)
    dx = (cfg.x_max - cfg.x_min) / n
    U = initial_condition(x, cfg)

    t = 0.0
    while t < cfg.t_final:
        speed = max_wave_speed(U, cfg)
        dt = cfg.cfl * dx / max(speed, 1e-12)
        if t + dt > cfg.t_final:
            dt = cfg.t_final - t
        U = finite_volume_step(U, dx, dt, cfg)
        U[:, 0] = np.maximum(U[:, 0], cfg.h_floor)
        t += dt

    return x + 0.5 * dx, U


def save_csv(path: Path, x: np.ndarray, U: np.ndarray, h_floor: float) -> None:
    h = np.maximum(U[:, 0], h_floor)
    m = U[:, 1]
    u = m / h

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "h", "u", "momentum"])
        for xi, hi, ui, mi in zip(x, h, u, m):
            writer.writerow([f"{xi:.8f}", f"{hi:.8f}", f"{ui:.8f}", f"{mi:.8f}"])


def main() -> None:
    cfg = SolverConfig()
    x, U = simulate(cfg)

    out_path = Path(__file__).with_name("result.csv")
    save_csv(out_path, x, U, cfg.h_floor)

    h = U[:, 0]
    total_mass = np.sum(h) * ((cfg.x_max - cfg.x_min) / cfg.n_cells)

    print("Roe近似Riemann解 MVP 运行完成")
    print(f"网格数: {cfg.n_cells}, 终止时间: {cfg.t_final}")
    print(f"水深范围: [{np.min(h):.6f}, {np.max(h):.6f}]")
    print(f"总质量(近似守恒量): {total_mass:.8f}")
    print(f"结果文件: {out_path}")


if __name__ == "__main__":
    main()

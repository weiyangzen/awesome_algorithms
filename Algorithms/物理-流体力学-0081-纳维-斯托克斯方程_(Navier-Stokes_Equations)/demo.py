"""Minimal runnable MVP for incompressible 2D Navier-Stokes equations.

The demo solves the lid-driven cavity benchmark with a projection method:
- Explicit advection + diffusion for velocity
- Pressure Poisson solve for incompressibility enforcement
- No-slip walls and moving lid boundary condition
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SimulationConfig:
    nx: int = 41
    ny: int = 41
    nt: int = 300
    nit: int = 50
    length_x: float = 1.0
    length_y: float = 1.0
    rho: float = 1.0
    nu: float = 0.1
    lid_velocity: float = 1.0
    cfl: float = 0.05


@dataclass(frozen=True)
class SimulationResult:
    u: np.ndarray
    v: np.ndarray
    p: np.ndarray
    dt: float
    dx: float
    dy: float


def build_rhs(u: np.ndarray, v: np.ndarray, rho: float, dt: float, dx: float, dy: float) -> np.ndarray:
    """Build RHS for the pressure Poisson equation."""
    b = np.zeros_like(u)
    dudx = (u[1:-1, 2:] - u[1:-1, 0:-2]) / (2.0 * dx)
    dvdy = (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2.0 * dy)
    dudy = (u[2:, 1:-1] - u[0:-2, 1:-1]) / (2.0 * dy)
    dvdx = (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2.0 * dx)

    b[1:-1, 1:-1] = rho * (
        (dudx + dvdy) / dt
        - dudx**2
        - 2.0 * dudy * dvdx
        - dvdy**2
    )
    return b


def pressure_poisson(p: np.ndarray, b: np.ndarray, dx: float, dy: float, nit: int) -> np.ndarray:
    """Iteratively solve pressure Poisson equation with simple boundary conditions."""
    for _ in range(nit):
        pn = p.copy()
        p[1:-1, 1:-1] = (
            (
                (pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2
                + (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2
            )
            / (2.0 * (dx**2 + dy**2))
            - (dx**2 * dy**2) / (2.0 * (dx**2 + dy**2)) * b[1:-1, 1:-1]
        )

        # dp/dx = 0 on left/right walls
        p[:, 0] = p[:, 1]
        p[:, -1] = p[:, -2]
        # dp/dy = 0 on bottom wall
        p[0, :] = p[1, :]
        # Reference pressure on top wall
        p[-1, :] = 0.0
    return p


def cavity_flow(config: SimulationConfig) -> SimulationResult:
    dx = config.length_x / (config.nx - 1)
    dy = config.length_y / (config.ny - 1)

    # Conservative dt from advection and diffusion constraints
    adv_dt = np.inf
    if config.lid_velocity > 0:
        adv_dt = config.cfl * min(dx, dy) / config.lid_velocity
    diff_dt = 0.25 * min(dx, dy) ** 2 / config.nu
    dt = min(adv_dt, diff_dt)

    u = np.zeros((config.ny, config.nx), dtype=float)
    v = np.zeros((config.ny, config.nx), dtype=float)
    p = np.zeros((config.ny, config.nx), dtype=float)

    for _ in range(config.nt):
        un = u.copy()
        vn = v.copy()

        b = build_rhs(un, vn, config.rho, dt, dx, dy)
        p = pressure_poisson(p, b, dx, dy, config.nit)

        u[1:-1, 1:-1] = (
            un[1:-1, 1:-1]
            - un[1:-1, 1:-1] * dt / dx * (un[1:-1, 1:-1] - un[1:-1, 0:-2])
            - vn[1:-1, 1:-1] * dt / dy * (un[1:-1, 1:-1] - un[0:-2, 1:-1])
            - dt / (2.0 * config.rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2])
            + config.nu
            * (
                dt / dx**2 * (un[1:-1, 2:] - 2.0 * un[1:-1, 1:-1] + un[1:-1, 0:-2])
                + dt / dy**2 * (un[2:, 1:-1] - 2.0 * un[1:-1, 1:-1] + un[0:-2, 1:-1])
            )
        )

        v[1:-1, 1:-1] = (
            vn[1:-1, 1:-1]
            - un[1:-1, 1:-1] * dt / dx * (vn[1:-1, 1:-1] - vn[1:-1, 0:-2])
            - vn[1:-1, 1:-1] * dt / dy * (vn[1:-1, 1:-1] - vn[0:-2, 1:-1])
            - dt / (2.0 * config.rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1])
            + config.nu
            * (
                dt / dx**2 * (vn[1:-1, 2:] - 2.0 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2])
                + dt / dy**2 * (vn[2:, 1:-1] - 2.0 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])
            )
        )

        # Velocity boundary conditions
        u[0, :] = 0.0
        u[:, 0] = 0.0
        u[:, -1] = 0.0
        u[-1, :] = config.lid_velocity

        v[0, :] = 0.0
        v[-1, :] = 0.0
        v[:, 0] = 0.0
        v[:, -1] = 0.0

    return SimulationResult(u=u, v=v, p=p, dt=dt, dx=dx, dy=dy)


def compute_divergence_linf(
    u: np.ndarray,
    v: np.ndarray,
    dx: float,
    dy: float,
    trim_cells: int = 0,
) -> float:
    div = (
        (u[1:-1, 2:] - u[1:-1, 0:-2]) / (2.0 * dx)
        + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2.0 * dy)
    )

    if trim_cells > 0 and div.shape[0] > 2 * trim_cells and div.shape[1] > 2 * trim_cells:
        div = div[trim_cells:-trim_cells, trim_cells:-trim_cells]

    return float(np.max(np.abs(div)))


def nearest_cell_value(field: np.ndarray, x: float, y: float, dx: float, dy: float) -> float:
    ix = int(round(x / dx))
    iy = int(round(y / dy))
    ix = max(0, min(ix, field.shape[1] - 1))
    iy = max(0, min(iy, field.shape[0] - 1))
    return float(field[iy, ix])


def main() -> None:
    config = SimulationConfig()
    result = cavity_flow(config)

    speed = np.sqrt(result.u**2 + result.v**2)
    divergence_linf = compute_divergence_linf(result.u, result.v, result.dx, result.dy)
    divergence_core_linf = compute_divergence_linf(
        result.u,
        result.v,
        result.dx,
        result.dy,
        trim_cells=2,
    )

    probes = [
        (0.50, 0.50),
        (0.50, 0.25),
        (0.50, 0.75),
        (0.25, 0.50),
        (0.75, 0.50),
    ]

    rows: list[dict[str, float]] = []
    for x, y in probes:
        rows.append(
            {
                "x": x,
                "y": y,
                "u": nearest_cell_value(result.u, x, y, result.dx, result.dy),
                "v": nearest_cell_value(result.v, x, y, result.dx, result.dy),
                "p": nearest_cell_value(result.p, x, y, result.dx, result.dy),
                "speed": nearest_cell_value(speed, x, y, result.dx, result.dy),
            }
        )

    df = pd.DataFrame(rows)

    print("Navier-Stokes MVP (2D incompressible lid-driven cavity)")
    print(f"grid            : {config.nx} x {config.ny}")
    print(f"time steps      : nt={config.nt}, pressure iterations per step={config.nit}")
    print(f"dx, dy, dt      : {result.dx:.6f}, {result.dy:.6f}, {result.dt:.6f}")
    print(f"rho, nu         : {config.rho:.3f}, {config.nu:.3f}")
    print(f"lid velocity    : {config.lid_velocity:.3f}")
    print(f"max(|div u|) global : {divergence_linf:.6e}")
    print(f"max(|div u|) core   : {divergence_core_linf:.6e} (trim 2 cells)")
    print(f"max speed       : {float(np.max(speed)):.6f}")
    print(f"mean speed      : {float(np.mean(speed)):.6f}")
    print("\nProbe values (nearest grid cell):")
    print(df.to_string(index=False, float_format=lambda v: f"{v: .6f}"))


if __name__ == "__main__":
    main()

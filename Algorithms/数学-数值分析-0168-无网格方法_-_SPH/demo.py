"""Minimal runnable MVP for SPH (Smoothed Particle Hydrodynamics).

This script runs a small 2D weakly-compressible SPH (WCSPH) simulation
without any interactive input. It focuses on algorithm transparency rather
than ultimate performance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class SPHConfig:
    """Configuration for a tiny 2D WCSPH demo."""

    rest_density: float = 1000.0
    particle_spacing: float = 0.04
    smoothing_length: float = 0.052
    stiffness: float = 1200.0
    sound_speed: float = 20.0
    alpha_viscosity: float = 0.08
    viscosity_eps: float = 0.01

    domain_width: float = 1.0
    domain_height: float = 0.8
    boundary_padding: float = 0.02
    boundary_restitution: float = 0.35
    tangential_damping: float = 0.98

    dt: float = 0.0015
    steps: int = 220
    log_interval: int = 20

    gravity_x: float = 0.0
    gravity_y: float = -9.8

    block_origin_x: float = 0.12
    block_origin_y: float = 0.10
    block_nx: int = 12
    block_ny: int = 9

    @property
    def mass(self) -> float:
        # 2D unit-thickness assumption: m = rho0 * dx^2
        return self.rest_density * (self.particle_spacing**2)

    @property
    def gravity(self) -> np.ndarray:
        return np.array([self.gravity_x, self.gravity_y], dtype=float)


@dataclass
class SPHState:
    """State vectors for all particles."""

    positions: np.ndarray  # shape (N, 2)
    velocities: np.ndarray  # shape (N, 2)


def make_initial_block(cfg: SPHConfig) -> SPHState:
    """Create a rectangular fluid block with zero initial velocity."""
    points = []
    for iy in range(cfg.block_ny):
        for ix in range(cfg.block_nx):
            x = cfg.block_origin_x + ix * cfg.particle_spacing
            y = cfg.block_origin_y + iy * cfg.particle_spacing
            points.append((x, y))

    positions = np.array(points, dtype=float)
    velocities = np.zeros_like(positions)
    return SPHState(positions=positions, velocities=velocities)


def cubic_spline_kernel_2d(r: np.ndarray, h: float) -> np.ndarray:
    """2D cubic spline kernel W(r,h) with compact support r < 2h."""
    q = r / h
    sigma = 10.0 / (7.0 * math.pi * h * h)

    w = np.zeros_like(r)
    mask1 = (q >= 0.0) & (q < 1.0)
    mask2 = (q >= 1.0) & (q < 2.0)

    q1 = q[mask1]
    q2 = q[mask2]
    w[mask1] = sigma * (1.0 - 1.5 * q1 * q1 + 0.75 * q1 * q1 * q1)
    w[mask2] = sigma * (0.25 * (2.0 - q2) ** 3)
    return w


def cubic_spline_dWdr_2d(r: np.ndarray, h: float) -> np.ndarray:
    """Radial derivative dW/dr for the 2D cubic spline kernel."""
    q = r / h
    sigma = 10.0 / (7.0 * math.pi * h * h)

    dWdr = np.zeros_like(r)
    mask1 = (q >= 0.0) & (q < 1.0)
    mask2 = (q >= 1.0) & (q < 2.0)

    q1 = q[mask1]
    q2 = q[mask2]

    # d/dq of piecewise kernel, then chain rule dq/dr = 1/h.
    dWdr[mask1] = sigma * (-3.0 * q1 + 2.25 * q1 * q1) / h
    dWdr[mask2] = sigma * (-0.75 * (2.0 - q2) ** 2) / h
    return dWdr


def pairwise_geometry(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return pairwise displacement dx_ij, squared distance and distance."""
    dx = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(dx * dx, axis=2)
    r = np.sqrt(r2)
    return dx, r2, r


def compute_density_pressure(
    positions: np.ndarray,
    cfg: SPHConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Density summation and Tait-like linearized pressure."""
    _, _, r = pairwise_geometry(positions)
    w = cubic_spline_kernel_2d(r, cfg.smoothing_length)

    rho = cfg.mass * np.sum(w, axis=1)
    pressure = cfg.stiffness * np.maximum(rho - cfg.rest_density, 0.0)
    return rho, pressure


def compute_acceleration(
    positions: np.ndarray,
    velocities: np.ndarray,
    density: np.ndarray,
    pressure: np.ndarray,
    cfg: SPHConfig,
) -> np.ndarray:
    """Compute acceleration from pressure + Monaghan artificial viscosity + gravity."""
    dx, r2, r = pairwise_geometry(positions)
    n_particles = positions.shape[0]

    dWdr = cubic_spline_dWdr_2d(r, cfg.smoothing_length)
    grad = np.zeros_like(dx)
    mask_nonzero = r > 1e-12
    grad[mask_nonzero] = (dWdr[mask_nonzero] / r[mask_nonzero])[:, np.newaxis] * dx[mask_nonzero]

    rho_i2 = density[:, np.newaxis] * density[:, np.newaxis]
    rho_j2 = density[np.newaxis, :] * density[np.newaxis, :]
    pressure_term = pressure[:, np.newaxis] / (rho_i2 + 1e-12) + pressure[np.newaxis, :] / (rho_j2 + 1e-12)

    v_ij = velocities[:, np.newaxis, :] - velocities[np.newaxis, :, :]
    vr = np.sum(v_ij * dx, axis=2)
    rho_bar = 0.5 * (density[:, np.newaxis] + density[np.newaxis, :])

    mu_ij = cfg.smoothing_length * vr / (r2 + cfg.viscosity_eps * cfg.smoothing_length * cfg.smoothing_length)
    pi_ij = np.zeros((n_particles, n_particles), dtype=float)
    mask_visc = (vr < 0.0) & (r < 2.0 * cfg.smoothing_length) & mask_nonzero
    pi_ij[mask_visc] = (
        -cfg.alpha_viscosity * cfg.sound_speed * mu_ij[mask_visc] / (rho_bar[mask_visc] + 1e-12)
    )

    coeff = cfg.mass * (pressure_term + pi_ij)
    np.fill_diagonal(coeff, 0.0)

    acc = -np.sum(coeff[:, :, np.newaxis] * grad, axis=1)
    acc += cfg.gravity[np.newaxis, :]
    return acc


def apply_boundary_conditions(state: SPHState, cfg: SPHConfig) -> None:
    """Reflect particles at domain walls with damping."""
    x_min = cfg.boundary_padding
    y_min = cfg.boundary_padding
    x_max = cfg.domain_width - cfg.boundary_padding
    y_max = cfg.domain_height - cfg.boundary_padding

    x = state.positions[:, 0]
    y = state.positions[:, 1]
    vx = state.velocities[:, 0]
    vy = state.velocities[:, 1]

    hit_left = x < x_min
    hit_right = x > x_max
    hit_bottom = y < y_min
    hit_top = y > y_max

    x[hit_left] = x_min
    x[hit_right] = x_max
    y[hit_bottom] = y_min
    y[hit_top] = y_max

    vx[hit_left | hit_right] *= -cfg.boundary_restitution
    vy[hit_bottom | hit_top] *= -cfg.boundary_restitution

    vy[hit_left | hit_right] *= cfg.tangential_damping
    vx[hit_bottom | hit_top] *= cfg.tangential_damping


def run_step(state: SPHState, cfg: SPHConfig) -> tuple[np.ndarray, np.ndarray]:
    """Advance one explicit time step and return latest (density, pressure)."""
    density, pressure = compute_density_pressure(state.positions, cfg)
    acc = compute_acceleration(state.positions, state.velocities, density, pressure, cfg)

    state.velocities += cfg.dt * acc
    state.positions += cfg.dt * state.velocities
    apply_boundary_conditions(state, cfg)

    density, pressure = compute_density_pressure(state.positions, cfg)
    return density, pressure


def summarize(state: SPHState, density: np.ndarray, step: int, cfg: SPHConfig) -> str:
    """Return one-line diagnostics for logs."""
    speed = np.linalg.norm(state.velocities, axis=1)
    kinetic = 0.5 * cfg.mass * float(np.sum(speed * speed))
    center = np.mean(state.positions, axis=0)

    return (
        f"step={step:4d}  "
        f"rho_mean={float(np.mean(density)):8.2f}  "
        f"rho_std={float(np.std(density)):7.2f}  "
        f"vmax={float(np.max(speed)):7.3f}  "
        f"E_k={kinetic:10.4f}  "
        f"center=({center[0]:.3f},{center[1]:.3f})"
    )


def main() -> None:
    cfg = SPHConfig()
    state = make_initial_block(cfg)

    print("=" * 96)
    print("SPH MVP (2D WCSPH, cubic spline kernel, artificial viscosity)")
    print("=" * 96)
    print(
        f"particles={state.positions.shape[0]}, dt={cfg.dt}, steps={cfg.steps}, "
        f"h={cfg.smoothing_length}, mass={cfg.mass:.4f}"
    )
    print(
        f"domain=({cfg.domain_width} x {cfg.domain_height}), "
        f"rest_density={cfg.rest_density}, stiffness={cfg.stiffness}"
    )

    density = np.full(state.positions.shape[0], cfg.rest_density, dtype=float)
    for step in range(1, cfg.steps + 1):
        density, _ = run_step(state, cfg)
        if step % cfg.log_interval == 0 or step == 1 or step == cfg.steps:
            print(summarize(state, density, step, cfg))

    print("-" * 96)
    print("Final particle sample (first 8):")
    for i in range(min(8, state.positions.shape[0])):
        px, py = state.positions[i]
        vx, vy = state.velocities[i]
        print(f"#{i:03d}: pos=({px:.4f}, {py:.4f}), vel=({vx:.4f}, {vy:.4f}), rho={density[i]:.2f}")


if __name__ == "__main__":
    main()

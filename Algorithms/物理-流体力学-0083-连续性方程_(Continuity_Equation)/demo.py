"""Minimal runnable MVP for the continuity equation.

We solve the 1D conservative form:
    d(rho)/dt + d(rho*u)/dx = 0
on a periodic domain using a first-order upwind finite-volume update.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ContinuityConfig:
    """Configuration for the 1D continuity-equation simulation."""

    domain_length: float = 1.0
    num_cells: int = 400
    velocity: float = 0.8
    cfl: float = 0.35
    final_time: float = 0.6


@dataclass(frozen=True)
class ContinuityResult:
    """Outputs and diagnostics of the simulation."""

    x: np.ndarray
    rho_initial: np.ndarray
    rho_numerical: np.ndarray
    rho_exact: np.ndarray
    dx: float
    dt: float
    num_steps: int
    velocity: float
    mass_initial: float
    mass_final: float
    relative_mass_drift: float
    l2_relative_error: float
    linf_error: float
    residual_inf: float


def validate_config(cfg: ContinuityConfig) -> None:
    if cfg.domain_length <= 0.0:
        raise ValueError("domain_length must be positive")
    if cfg.num_cells < 16:
        raise ValueError("num_cells must be >= 16")
    if cfg.cfl <= 0.0:
        raise ValueError("cfl must be positive")
    if cfg.final_time <= 0.0:
        raise ValueError("final_time must be positive")


def make_grid(domain_length: float, num_cells: int) -> tuple[np.ndarray, float]:
    dx = domain_length / float(num_cells)
    x = (np.arange(num_cells, dtype=float) + 0.5) * dx
    return x, dx


def initial_density(x: np.ndarray, domain_length: float) -> np.ndarray:
    """Smooth positive profile with a known periodic analytic advection solution."""
    theta = 2.0 * np.pi * x / domain_length
    return 1.0 + 0.2 * np.sin(theta) + 0.1 * np.cos(2.0 * theta)


def upwind_face_flux(rho: np.ndarray, velocity: float) -> np.ndarray:
    """Compute interface flux F_{i+1/2} = (rho*u)_{i+1/2} for periodic cells."""
    if velocity >= 0.0:
        # Upwind value at i+1/2 comes from cell i.
        return velocity * rho
    # Upwind value at i+1/2 comes from cell i+1.
    return velocity * np.roll(rho, -1)


def advance_one_step(
    rho: np.ndarray,
    velocity: float,
    dt: float,
    dx: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One conservative finite-volume step.

    Returns
    - rho_next
    - flux_right: F_{i+1/2}
    - flux_left: F_{i-1/2}
    """
    flux_right = upwind_face_flux(rho, velocity)
    flux_left = np.roll(flux_right, 1)
    rho_next = rho - (dt / dx) * (flux_right - flux_left)
    return rho_next, flux_right, flux_left


def exact_density(
    x: np.ndarray,
    time: float,
    velocity: float,
    domain_length: float,
) -> np.ndarray:
    x_backtracked = np.mod(x - velocity * time, domain_length)
    return initial_density(x_backtracked, domain_length)


def solve_continuity(cfg: ContinuityConfig) -> ContinuityResult:
    validate_config(cfg)
    x, dx = make_grid(cfg.domain_length, cfg.num_cells)

    speed = abs(cfg.velocity)
    if speed < 1e-14:
        num_steps = 1
        dt = cfg.final_time
    else:
        dt_cfl = cfg.cfl * dx / speed
        num_steps = max(1, int(np.ceil(cfg.final_time / dt_cfl)))
        dt = cfg.final_time / float(num_steps)

    rho = initial_density(x, cfg.domain_length)
    rho_initial = rho.copy()
    mass_initial = float(np.sum(rho_initial) * dx)

    residual_inf = 0.0

    for _ in range(num_steps):
        rho_next, flux_right, flux_left = advance_one_step(rho, cfg.velocity, dt, dx)

        # Discrete residual of continuity equation for this step.
        residual = (rho_next - rho) / dt + (flux_right - flux_left) / dx
        residual_inf = max(residual_inf, float(np.max(np.abs(residual))))

        rho = rho_next

    rho_exact = exact_density(x, cfg.final_time, cfg.velocity, cfg.domain_length)

    mass_final = float(np.sum(rho) * dx)
    relative_mass_drift = abs(mass_final - mass_initial) / abs(mass_initial)
    l2_relative_error = float(np.linalg.norm(rho - rho_exact) / np.linalg.norm(rho_exact))
    linf_error = float(np.max(np.abs(rho - rho_exact)))

    return ContinuityResult(
        x=x,
        rho_initial=rho_initial,
        rho_numerical=rho,
        rho_exact=rho_exact,
        dx=dx,
        dt=dt,
        num_steps=num_steps,
        velocity=cfg.velocity,
        mass_initial=mass_initial,
        mass_final=mass_final,
        relative_mass_drift=relative_mass_drift,
        l2_relative_error=l2_relative_error,
        linf_error=linf_error,
        residual_inf=residual_inf,
    )


def run_checks(result: ContinuityResult) -> None:
    # Conservative finite-volume update on periodic mesh should preserve mass closely.
    if result.relative_mass_drift > 1e-12:
        raise AssertionError(f"Mass drift too large: {result.relative_mass_drift:.3e}")

    # Upwind scheme is diffusive, so we keep an honest but strict enough threshold.
    if result.l2_relative_error > 0.10:
        raise AssertionError(f"L2 relative error too large: {result.l2_relative_error:.3e}")

    if np.min(result.rho_numerical) <= 0.0:
        raise AssertionError("Density became non-positive, which is non-physical here.")

    if result.residual_inf > 5e-12:
        raise AssertionError(f"Discrete residual too large: {result.residual_inf:.3e}")


def main() -> None:
    cfg = ContinuityConfig()
    result = solve_continuity(cfg)
    run_checks(result)

    print("=== Continuity Equation MVP (1D, periodic, upwind FV) ===")
    print(f"cells={cfg.num_cells}, steps={result.num_steps}, dx={result.dx:.6f}, dt={result.dt:.6f}")
    print(f"velocity={result.velocity:.6f}, final_time={cfg.final_time:.6f}")
    print(f"mass_initial={result.mass_initial:.12f}")
    print(f"mass_final  ={result.mass_final:.12f}")
    print(f"relative_mass_drift={result.relative_mass_drift:.3e}")
    print(f"l2_relative_error  ={result.l2_relative_error:.3e}")
    print(f"linf_error         ={result.linf_error:.3e}")
    print(f"discrete_residual_inf={result.residual_inf:.3e}")
    print(f"rho_initial_range=[{np.min(result.rho_initial):.6f}, {np.max(result.rho_initial):.6f}]")
    print(f"rho_final_range  =[{np.min(result.rho_numerical):.6f}, {np.max(result.rho_numerical):.6f}]")
    print("All checks passed.")


if __name__ == "__main__":
    main()

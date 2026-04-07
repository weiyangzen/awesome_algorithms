"""Minimal runnable MVP for MD integrators on a 2D Lennard-Jones system.

This script compares three time integrators under NVE dynamics:
- Explicit Euler
- Symplectic Euler (kick-drift)
- Velocity Verlet

It reports long-time relative energy drift and final-state error against
an internal high-resolution Velocity-Verlet reference trajectory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


EPS = 1e-12


@dataclass(frozen=True)
class MDConfig:
    n_particles: int = 16
    dim: int = 2
    mass: float = 1.0
    density: float = 0.78
    temperature: float = 0.9
    sigma: float = 1.0
    epsilon: float = 1.0
    cutoff: float = 2.5
    dt: float = 0.004
    steps: int = 2200
    reference_substeps: int = 5
    init_jitter_scale: float = 0.04
    seed: int = 20260407


@dataclass(frozen=True)
class IntegratorResult:
    name: str
    energies: np.ndarray
    final_positions: np.ndarray
    final_velocities: np.ndarray


def minimum_image(delta: np.ndarray, box_length: float) -> np.ndarray:
    """Apply periodic minimum-image convention."""
    return delta - box_length * np.rint(delta / box_length)


def init_lattice_positions(n_particles: int, dim: int, box_length: float) -> np.ndarray:
    """Initialize particles on a square lattice in [0, L)."""
    if dim != 2:
        raise ValueError("This MVP currently supports dim=2 only.")

    side = int(math.ceil(math.sqrt(n_particles)))
    spacing = box_length / side
    coords: list[list[float]] = []

    for ix in range(side):
        for iy in range(side):
            if len(coords) >= n_particles:
                break
            coords.append([(ix + 0.5) * spacing, (iy + 0.5) * spacing])
        if len(coords) >= n_particles:
            break

    return np.asarray(coords, dtype=float)


def initialize_state(cfg: MDConfig) -> tuple[float, np.ndarray, np.ndarray]:
    """Build deterministic initial coordinates/velocities for NVE MD."""
    rng = np.random.default_rng(cfg.seed)
    volume = cfg.n_particles / cfg.density
    box_length = volume ** (1.0 / cfg.dim)

    coords = init_lattice_positions(cfg.n_particles, cfg.dim, box_length)

    jitter = cfg.init_jitter_scale * (box_length / math.sqrt(cfg.n_particles))
    coords = (coords + rng.normal(0.0, jitter, size=coords.shape)) % box_length

    velocities = rng.normal(0.0, 1.0, size=(cfg.n_particles, cfg.dim))
    velocities -= velocities.mean(axis=0, keepdims=True)

    dof = cfg.dim * (cfg.n_particles - 1)
    target_kinetic = 0.5 * dof * cfg.temperature
    current_kinetic = 0.5 * cfg.mass * np.sum(velocities * velocities)
    scale = math.sqrt(max(target_kinetic, EPS) / max(current_kinetic, EPS))
    velocities *= scale

    return box_length, coords, velocities


def compute_forces_and_potential(
    coords: np.ndarray,
    box_length: float,
    sigma: float,
    epsilon: float,
    cutoff: float,
) -> tuple[np.ndarray, float]:
    """Compute LJ forces and shifted-cutoff pair potential in O(N^2)."""
    n = coords.shape[0]
    forces = np.zeros_like(coords)
    potential = 0.0

    cutoff2 = cutoff * cutoff
    inv_rc2 = (sigma * sigma) / cutoff2
    inv_rc6 = inv_rc2 * inv_rc2 * inv_rc2
    inv_rc12 = inv_rc6 * inv_rc6
    shift = 4.0 * epsilon * (inv_rc12 - inv_rc6)

    for i in range(n - 1):
        for j in range(i + 1, n):
            d = minimum_image(coords[i] - coords[j], box_length)
            r2 = float(np.dot(d, d))

            if r2 >= cutoff2:
                continue
            if r2 < 1e-10:
                raise ValueError("Particles are too close; force becomes singular.")

            inv_r2 = (sigma * sigma) / r2
            inv_r6 = inv_r2 * inv_r2 * inv_r2
            inv_r12 = inv_r6 * inv_r6

            pair_u = 4.0 * epsilon * (inv_r12 - inv_r6) - shift
            potential += pair_u

            prefactor = 24.0 * epsilon * (2.0 * inv_r12 - inv_r6) / r2
            fij = prefactor * d
            forces[i] += fij
            forces[j] -= fij

    return forces, float(potential)


def total_energy(mass: float, velocities: np.ndarray, potential: float) -> float:
    kinetic = 0.5 * mass * float(np.sum(velocities * velocities))
    return kinetic + potential


def integrate_euler(
    cfg: MDConfig,
    box_length: float,
    coords0: np.ndarray,
    vel0: np.ndarray,
) -> IntegratorResult:
    """Explicit Euler: x_{n+1}=x_n+dt v_n, v_{n+1}=v_n+dt a(x_n)."""
    coords = coords0.copy()
    velocities = vel0.copy()

    _, pot0 = compute_forces_and_potential(coords, box_length, cfg.sigma, cfg.epsilon, cfg.cutoff)
    energies = np.zeros(cfg.steps + 1, dtype=float)
    energies[0] = total_energy(cfg.mass, velocities, pot0)

    for step in range(1, cfg.steps + 1):
        forces, _ = compute_forces_and_potential(coords, box_length, cfg.sigma, cfg.epsilon, cfg.cutoff)
        acc = forces / cfg.mass

        coords = (coords + cfg.dt * velocities) % box_length
        velocities = velocities + cfg.dt * acc

        _, pot = compute_forces_and_potential(coords, box_length, cfg.sigma, cfg.epsilon, cfg.cutoff)
        energies[step] = total_energy(cfg.mass, velocities, pot)

    return IntegratorResult(
        name="Euler",
        energies=energies,
        final_positions=coords,
        final_velocities=velocities,
    )


def integrate_symplectic_euler(
    cfg: MDConfig,
    box_length: float,
    coords0: np.ndarray,
    vel0: np.ndarray,
) -> IntegratorResult:
    """Symplectic Euler (kick-drift): v update first, then x update."""
    coords = coords0.copy()
    velocities = vel0.copy()

    _, pot0 = compute_forces_and_potential(coords, box_length, cfg.sigma, cfg.epsilon, cfg.cutoff)
    energies = np.zeros(cfg.steps + 1, dtype=float)
    energies[0] = total_energy(cfg.mass, velocities, pot0)

    for step in range(1, cfg.steps + 1):
        forces, _ = compute_forces_and_potential(coords, box_length, cfg.sigma, cfg.epsilon, cfg.cutoff)
        acc = forces / cfg.mass

        velocities = velocities + cfg.dt * acc
        coords = (coords + cfg.dt * velocities) % box_length

        _, pot = compute_forces_and_potential(coords, box_length, cfg.sigma, cfg.epsilon, cfg.cutoff)
        energies[step] = total_energy(cfg.mass, velocities, pot)

    return IntegratorResult(
        name="SymplecticEuler",
        energies=energies,
        final_positions=coords,
        final_velocities=velocities,
    )


def integrate_velocity_verlet(
    cfg: MDConfig,
    box_length: float,
    coords0: np.ndarray,
    vel0: np.ndarray,
    *,
    dt_override: float | None = None,
    steps_override: int | None = None,
) -> IntegratorResult:
    """Velocity-Verlet (kick-drift-kick), standard in MD."""
    dt = cfg.dt if dt_override is None else dt_override
    steps = cfg.steps if steps_override is None else steps_override

    coords = coords0.copy()
    velocities = vel0.copy()

    forces, pot = compute_forces_and_potential(coords, box_length, cfg.sigma, cfg.epsilon, cfg.cutoff)
    energies = np.zeros(steps + 1, dtype=float)
    energies[0] = total_energy(cfg.mass, velocities, pot)

    for step in range(1, steps + 1):
        acc = forces / cfg.mass
        v_half = velocities + 0.5 * dt * acc
        coords = (coords + dt * v_half) % box_length

        forces_new, pot_new = compute_forces_and_potential(
            coords, box_length, cfg.sigma, cfg.epsilon, cfg.cutoff
        )
        acc_new = forces_new / cfg.mass
        velocities = v_half + 0.5 * dt * acc_new

        forces = forces_new
        pot = pot_new
        energies[step] = total_energy(cfg.mass, velocities, pot)

    return IntegratorResult(
        name="VelocityVerlet",
        energies=energies,
        final_positions=coords,
        final_velocities=velocities,
    )


def max_relative_energy_drift(energies: np.ndarray) -> float:
    """max_t |E(t)-E(0)| / max(|E(0)|, eps)."""
    e0 = energies[0]
    return float(np.max(np.abs(energies - e0)) / max(abs(e0), EPS))


def final_state_rms_errors(
    box_length: float,
    ref_pos: np.ndarray,
    ref_vel: np.ndarray,
    test_pos: np.ndarray,
    test_vel: np.ndarray,
) -> tuple[float, float]:
    """RMS errors against reference final state (with PBC-aware position diff)."""
    delta_pos = minimum_image(test_pos - ref_pos, box_length)
    delta_vel = test_vel - ref_vel

    pos_rms = math.sqrt(float(np.mean(np.sum(delta_pos * delta_pos, axis=1))))
    vel_rms = math.sqrt(float(np.mean(np.sum(delta_vel * delta_vel, axis=1))))
    return pos_rms, vel_rms


def summarize_results(
    cfg: MDConfig,
    box_length: float,
    results: list[IntegratorResult],
    ref_result: IntegratorResult,
) -> pd.DataFrame:
    """Create a comparable table of integrator quality metrics."""
    meta = {
        "Euler": {"order": 1, "symplectic": "No", "time_reversible": "No"},
        "SymplecticEuler": {"order": 1, "symplectic": "Yes", "time_reversible": "No"},
        "VelocityVerlet": {"order": 2, "symplectic": "Yes", "time_reversible": "Yes"},
    }

    rows: list[dict[str, float | int | str]] = []
    for r in results:
        pos_err, vel_err = final_state_rms_errors(
            box_length,
            ref_result.final_positions,
            ref_result.final_velocities,
            r.final_positions,
            r.final_velocities,
        )
        rows.append(
            {
                "integrator": r.name,
                "order": meta[r.name]["order"],
                "symplectic": meta[r.name]["symplectic"],
                "time_reversible": meta[r.name]["time_reversible"],
                "dt": cfg.dt,
                "steps": cfg.steps,
                "max_rel_energy_drift": max_relative_energy_drift(r.energies),
                "final_rms_pos_error": pos_err,
                "final_rms_vel_error": vel_err,
            }
        )

    table = pd.DataFrame(rows).sort_values("max_rel_energy_drift", ascending=True)
    return table


def main() -> None:
    cfg = MDConfig()
    box_length, coords0, vel0 = initialize_state(cfg)

    euler_result = integrate_euler(cfg, box_length, coords0, vel0)
    symp_result = integrate_symplectic_euler(cfg, box_length, coords0, vel0)
    verlet_result = integrate_velocity_verlet(cfg, box_length, coords0, vel0)

    ref_result = integrate_velocity_verlet(
        cfg,
        box_length,
        coords0,
        vel0,
        dt_override=cfg.dt / cfg.reference_substeps,
        steps_override=cfg.steps * cfg.reference_substeps,
    )

    table = summarize_results(
        cfg,
        box_length,
        [euler_result, symp_result, verlet_result],
        ref_result,
    )

    print("=== MD Integrators MVP (PHYS-0323) ===")
    print(
        f"N={cfg.n_particles}, dim={cfg.dim}, density={cfg.density}, "
        f"temperature={cfg.temperature}, dt={cfg.dt}, steps={cfg.steps}"
    )
    print(f"Reference: VelocityVerlet with dt={cfg.dt / cfg.reference_substeps}")
    with pd.option_context("display.precision", 6, "display.width", 160):
        print(table.to_string(index=False))

    row_by_name = {row["integrator"]: row for _, row in table.iterrows()}
    euler_drift = float(row_by_name["Euler"]["max_rel_energy_drift"])
    symp_drift = float(row_by_name["SymplecticEuler"]["max_rel_energy_drift"])
    verlet_drift = float(row_by_name["VelocityVerlet"]["max_rel_energy_drift"])

    euler_pos = float(row_by_name["Euler"]["final_rms_pos_error"])
    verlet_pos = float(row_by_name["VelocityVerlet"]["final_rms_pos_error"])

    checks = {
        "VelocityVerlet drift < Euler drift": verlet_drift < euler_drift,
        "SymplecticEuler drift < Euler drift": symp_drift < euler_drift,
        "VelocityVerlet pos error < Euler pos error": verlet_pos < euler_pos,
        "VelocityVerlet drift < 0.20": verlet_drift < 0.20,
    }

    print("\nThreshold checks:")
    for name, ok in checks.items():
        print(f"- {name}: {'OK' if ok else 'FAIL'}")

    if all(checks.values()):
        print("\nValidation: PASS")
    else:
        print("\nValidation: FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

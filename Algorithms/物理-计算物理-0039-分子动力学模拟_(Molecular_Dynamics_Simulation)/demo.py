"""Molecular Dynamics MVP: 2D Lennard-Jones fluid with Velocity-Verlet."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MDConfig:
    """Simulation settings in reduced Lennard-Jones units."""

    n_particles: int = 36
    dimension: int = 2
    density: float = 0.75
    temperature: float = 1.0
    mass: float = 1.0
    epsilon: float = 1.0
    sigma: float = 1.0
    cutoff: float = 2.5
    dt: float = 0.0035
    n_steps: int = 1600
    thermalization_steps: int = 300
    sample_interval: int = 20
    thermostat_interval: int = 10
    seed: int = 20260407


def _validate_config(cfg: MDConfig) -> None:
    if cfg.n_particles < 4:
        raise ValueError("n_particles must be >= 4")
    if cfg.dimension != 2:
        raise ValueError("This MVP is intentionally implemented for dimension=2 only")
    if cfg.density <= 0.0:
        raise ValueError("density must be positive")
    if cfg.temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if cfg.mass <= 0.0:
        raise ValueError("mass must be positive")
    if cfg.cutoff <= 0.0 or cfg.dt <= 0.0:
        raise ValueError("cutoff and dt must be positive")
    if cfg.n_steps <= cfg.thermalization_steps:
        raise ValueError("n_steps must be larger than thermalization_steps")
    if cfg.sample_interval <= 0:
        raise ValueError("sample_interval must be positive")
    if cfg.thermostat_interval <= 0:
        raise ValueError("thermostat_interval must be positive")


def _build_lattice_positions(
    n_particles: int, box_length: float, rng: np.random.Generator
) -> np.ndarray:
    side = math.ceil(math.sqrt(n_particles))
    spacing = box_length / side
    positions = np.zeros((n_particles, 2), dtype=np.float64)

    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= n_particles:
                break
            positions[idx, 0] = (i + 0.5) * spacing
            positions[idx, 1] = (j + 0.5) * spacing
            idx += 1
        if idx >= n_particles:
            break

    jitter = 0.05 * spacing * (rng.random((n_particles, 2)) - 0.5)
    positions = (positions + jitter) % box_length
    return positions


def _kinetic_energy(velocities: np.ndarray, mass: float) -> float:
    return 0.5 * mass * float(np.sum(velocities * velocities))


def _instantaneous_temperature(velocities: np.ndarray, mass: float) -> float:
    n_particles, dim = velocities.shape
    dof = dim * n_particles - dim
    if dof <= 0:
        dof = dim * n_particles
    kinetic = _kinetic_energy(velocities, mass)
    return (2.0 * kinetic) / dof


def _initialize_velocities(
    n_particles: int,
    temperature: float,
    mass: float,
    rng: np.random.Generator,
) -> np.ndarray:
    velocity_std = math.sqrt(temperature / mass)
    velocities = rng.normal(0.0, velocity_std, size=(n_particles, 2))

    # Remove center-of-mass drift.
    velocities -= np.mean(velocities, axis=0, keepdims=True)

    temp_now = _instantaneous_temperature(velocities, mass)
    if temp_now <= 1e-14:
        raise RuntimeError("initialized temperature is too small")
    velocities *= math.sqrt(temperature / temp_now)
    return velocities


def _forces_and_potential(
    positions: np.ndarray,
    box_length: float,
    epsilon: float,
    sigma: float,
    cutoff: float,
) -> tuple[np.ndarray, float]:
    n_particles = positions.shape[0]
    forces = np.zeros_like(positions)
    potential = 0.0

    cutoff2 = cutoff * cutoff
    sigma2 = sigma * sigma
    sigma_over_rc = sigma / cutoff
    sigma_over_rc_6 = sigma_over_rc**6
    shift = 4.0 * epsilon * (sigma_over_rc_6**2 - sigma_over_rc_6)

    for i in range(n_particles - 1):
        for j in range(i + 1, n_particles):
            delta = positions[i] - positions[j]
            delta -= box_length * np.round(delta / box_length)
            r2 = float(np.dot(delta, delta))
            if r2 < 1e-12:
                raise RuntimeError("Particles overlapped too closely (r^2 < 1e-12)")
            if r2 >= cutoff2:
                continue

            inv_r2 = 1.0 / r2
            sr2 = sigma2 * inv_r2
            sr6 = sr2 * sr2 * sr2
            sr12 = sr6 * sr6

            pair_potential = 4.0 * epsilon * (sr12 - sr6) - shift
            potential += pair_potential

            prefactor = 24.0 * epsilon * inv_r2 * (2.0 * sr12 - sr6)
            fij = prefactor * delta
            forces[i] += fij
            forces[j] -= fij

    return forces, potential


def _velocity_verlet_step(
    positions: np.ndarray,
    velocities: np.ndarray,
    forces: np.ndarray,
    box_length: float,
    cfg: MDConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    dt = cfg.dt
    mass = cfg.mass

    positions = positions + velocities * dt + 0.5 * (forces / mass) * (dt * dt)
    positions %= box_length

    new_forces, potential = _forces_and_potential(
        positions=positions,
        box_length=box_length,
        epsilon=cfg.epsilon,
        sigma=cfg.sigma,
        cutoff=cfg.cutoff,
    )
    velocities = velocities + 0.5 * (forces + new_forces) * (dt / mass)
    return positions, velocities, new_forces, potential


def _rescale_to_temperature(
    velocities: np.ndarray, target_temperature: float, mass: float
) -> np.ndarray:
    temperature = _instantaneous_temperature(velocities, mass)
    if temperature <= 1e-14:
        return velocities
    scale = math.sqrt(target_temperature / temperature)
    return velocities * scale


def run_md_simulation(cfg: MDConfig) -> tuple[pd.DataFrame, dict[str, float]]:
    _validate_config(cfg)
    rng = np.random.default_rng(cfg.seed)

    box_length = math.sqrt(cfg.n_particles / cfg.density)
    positions = _build_lattice_positions(cfg.n_particles, box_length, rng)
    velocities = _initialize_velocities(cfg.n_particles, cfg.temperature, cfg.mass, rng)
    forces, potential = _forces_and_potential(
        positions=positions,
        box_length=box_length,
        epsilon=cfg.epsilon,
        sigma=cfg.sigma,
        cutoff=cfg.cutoff,
    )

    records: list[dict[str, float]] = []
    for step in range(1, cfg.n_steps + 1):
        positions, velocities, forces, potential = _velocity_verlet_step(
            positions=positions,
            velocities=velocities,
            forces=forces,
            box_length=box_length,
            cfg=cfg,
        )

        if step <= cfg.thermalization_steps and step % cfg.thermostat_interval == 0:
            velocities = _rescale_to_temperature(
                velocities=velocities,
                target_temperature=cfg.temperature,
                mass=cfg.mass,
            )

        kinetic = _kinetic_energy(velocities, cfg.mass)
        temperature = _instantaneous_temperature(velocities, cfg.mass)
        total_energy = kinetic + potential

        if step > cfg.thermalization_steps and step % cfg.sample_interval == 0:
            records.append(
                {
                    "step": float(step),
                    "potential_energy": potential,
                    "kinetic_energy": kinetic,
                    "total_energy": total_energy,
                    "temperature": temperature,
                }
            )

    if not records:
        raise RuntimeError("No samples collected. Check thermalization/sample settings.")

    df = pd.DataFrame.from_records(records)
    first_total = float(df["total_energy"].iloc[0])
    last_total = float(df["total_energy"].iloc[-1])
    relative_drift = (last_total - first_total) / max(abs(first_total), 1e-12)

    summary = {
        "n_samples": float(len(df)),
        "mean_temperature": float(df["temperature"].mean()),
        "std_temperature": float(df["temperature"].std(ddof=0)),
        "mean_total_energy": float(df["total_energy"].mean()),
        "std_total_energy": float(df["total_energy"].std(ddof=0)),
        "relative_energy_drift": relative_drift,
    }
    return df, summary


def main() -> None:
    cfg = MDConfig()
    samples, summary = run_md_simulation(cfg)

    print("=== Molecular Dynamics (2D Lennard-Jones, Velocity-Verlet) ===")
    print(
        f"N={cfg.n_particles}, density={cfg.density:.3f}, T_target={cfg.temperature:.3f}, "
        f"dt={cfg.dt:.4f}, steps={cfg.n_steps}, seed={cfg.seed}"
    )
    print()
    print("Sample preview (first 8 rows):")
    print(samples.head(8).to_string(index=False, float_format=lambda x: f"{x: .6f}"))
    print()
    print("Summary:")
    for key in [
        "n_samples",
        "mean_temperature",
        "std_temperature",
        "mean_total_energy",
        "std_total_energy",
        "relative_energy_drift",
    ]:
        value = summary[key]
        print(f"  {key}: {value:.6f}")

    # Lightweight non-interactive sanity checks.
    assert summary["n_samples"] >= 20.0
    assert 0.70 <= summary["mean_temperature"] <= 1.30
    assert abs(summary["relative_energy_drift"]) < 0.15
    print("\nSelf-check passed.")


if __name__ == "__main__":
    main()

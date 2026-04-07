"""Minimal MVP for the microcanonical (NVE) ensemble.

The demo simulates a 1D periodic harmonic chain with velocity-Verlet
integration (a symplectic scheme). The Hamiltonian is conserved up to
small numerical error, which is the key computational signature of NVE.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NVEConfig:
    n_particles: int = 32
    mass: float = 1.0
    spring_k: float = 1.0
    target_total_energy: float = 32.0
    dt: float = 0.02
    n_steps: int = 8000
    sample_every: int = 160
    seed: int = 20260407


def kinetic_energy(velocities: np.ndarray, mass: float) -> float:
    return float(0.5 * mass * np.dot(velocities, velocities))


def potential_energy(positions: np.ndarray, spring_k: float) -> float:
    displacement = np.roll(positions, -1) - positions
    return float(0.5 * spring_k * np.dot(displacement, displacement))


def spring_forces(positions: np.ndarray, spring_k: float) -> np.ndarray:
    return spring_k * (np.roll(positions, -1) + np.roll(positions, 1) - 2.0 * positions)


def initialize_state(config: NVEConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    positions = rng.normal(0.0, 1.0, size=config.n_particles)
    positions -= float(np.mean(positions))

    target_potential = 0.4 * config.target_total_energy
    raw_potential = potential_energy(positions, config.spring_k)
    if raw_potential <= 1e-14:
        positions[0] += 1e-3
        raw_potential = potential_energy(positions, config.spring_k)
    positions *= np.sqrt(target_potential / raw_potential)

    velocities = rng.normal(0.0, 1.0, size=config.n_particles)
    velocities -= float(np.mean(velocities))
    target_kinetic = config.target_total_energy - target_potential
    raw_kinetic = kinetic_energy(velocities, config.mass)
    velocities *= np.sqrt(target_kinetic / raw_kinetic)

    return positions, velocities


def velocity_verlet_step(
    positions: np.ndarray,
    velocities: np.ndarray,
    forces: np.ndarray,
    config: NVEConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dt = config.dt
    inv_mass = 1.0 / config.mass

    new_positions = positions + velocities * dt + 0.5 * forces * inv_mass * dt * dt
    new_forces = spring_forces(new_positions, config.spring_k)
    new_velocities = velocities + 0.5 * (forces + new_forces) * inv_mass * dt
    return new_positions, new_velocities, new_forces


def microcanonical_temperature(kinetic: float, effective_dof: int) -> float:
    return float(2.0 * kinetic / effective_dof)


def run_simulation(config: NVEConfig) -> tuple[pd.DataFrame, dict[str, float]]:
    rng = np.random.default_rng(config.seed)
    positions, velocities = initialize_state(config, rng)
    forces = spring_forces(positions, config.spring_k)

    effective_dof = max(1, config.n_particles - 1)
    records: list[dict[str, float]] = []
    velocity_samples: list[np.ndarray] = []

    for step in range(config.n_steps + 1):
        if step % config.sample_every == 0 or step == config.n_steps:
            kinetic = kinetic_energy(velocities, config.mass)
            potential = potential_energy(positions, config.spring_k)
            total = kinetic + potential
            temperature = microcanonical_temperature(kinetic, effective_dof)
            records.append(
                {
                    "step": float(step),
                    "kinetic": kinetic,
                    "potential": potential,
                    "total_energy": total,
                    "temperature_est": temperature,
                }
            )
            velocity_samples.append(velocities.copy())

        if step == config.n_steps:
            break

        positions, velocities, forces = velocity_verlet_step(positions, velocities, forces, config)

    df = pd.DataFrame(records)
    initial_energy = float(df["total_energy"].iloc[0])
    relative_drift = (df["total_energy"] - initial_energy) / abs(initial_energy)

    mean_kinetic = float(df["kinetic"].mean())
    mean_potential = float(df["potential"].mean())
    equipartition_ratio = mean_kinetic / mean_potential if mean_potential > 0 else float("nan")

    temperature_est = float(df["temperature_est"].mean())
    temperature_theory = initial_energy / effective_dof

    stacked_v = np.concatenate(velocity_samples)
    v_centered = stacked_v - float(np.mean(stacked_v))
    v_var = float(np.mean(v_centered * v_centered))
    if v_var > 0.0:
        v_excess_kurtosis = float(np.mean(v_centered**4) / (v_var * v_var) - 3.0)
    else:
        v_excess_kurtosis = float("nan")

    summary = {
        "initial_energy": initial_energy,
        "mean_total_energy": float(df["total_energy"].mean()),
        "max_abs_relative_drift": float(np.max(np.abs(relative_drift))),
        "std_relative_drift": float(np.std(relative_drift)),
        "mean_kinetic": mean_kinetic,
        "mean_potential": mean_potential,
        "equipartition_ratio_K_over_U": float(equipartition_ratio),
        "temperature_estimate": temperature_est,
        "temperature_theory": float(temperature_theory),
        "temperature_relative_error": float((temperature_est - temperature_theory) / temperature_theory),
        "velocity_mean": float(np.mean(stacked_v)),
        "velocity_variance": v_var,
        "velocity_excess_kurtosis": v_excess_kurtosis,
    }
    return df, summary


def main() -> None:
    config = NVEConfig()
    df, summary = run_simulation(config)

    print("=== Microcanonical (NVE) Ensemble MVP: Harmonic Chain MD ===")
    print(
        "config:",
        f"N={config.n_particles}, m={config.mass}, k={config.spring_k}, E_target={config.target_total_energy},",
        f"dt={config.dt}, n_steps={config.n_steps}, sample_every={config.sample_every}, seed={config.seed}",
    )
    print()
    print(df.to_string(index=False, float_format=lambda x: f"{x:10.6f}"))
    print()

    for key in (
        "initial_energy",
        "mean_total_energy",
        "max_abs_relative_drift",
        "std_relative_drift",
        "mean_kinetic",
        "mean_potential",
        "equipartition_ratio_K_over_U",
        "temperature_estimate",
        "temperature_theory",
        "temperature_relative_error",
        "velocity_mean",
        "velocity_variance",
        "velocity_excess_kurtosis",
    ):
        print(f"{key:>30s}: {summary[key]: .6e}")

    passed = (
        summary["max_abs_relative_drift"] < 5e-3
        and 0.8 < summary["equipartition_ratio_K_over_U"] < 1.2
        and abs(summary["temperature_relative_error"]) < 0.2
    )
    print(f"\nNVE sanity check: {'PASS' if passed else 'CHECK_MANUALLY'}")


if __name__ == "__main__":
    main()

"""Nose-Hoover thermostat MVP on an ensemble of 1D harmonic oscillators.

Run:
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for a minimal Nose-Hoover simulation."""

    n_oscillators: int = 256
    n_steps: int = 40_000
    dt: float = 0.002
    sample_every: int = 20
    burn_in_steps: int = 12_000
    target_temperature: float = 1.0
    thermostat_mass: float = 5.0
    mass: float = 1.0
    k_spring: float = 1.0
    kb: float = 1.0
    seed: int = 42


def harmonic_force(position: np.ndarray, k_spring: float) -> np.ndarray:
    """Force for V(q)=0.5*k*q^2."""

    return -k_spring * position


def kinetic_energy(momentum: np.ndarray, mass: float) -> float:
    """Total kinetic energy for independent 1D oscillators."""

    return 0.5 * float(np.sum((momentum * momentum) / mass))


def instantaneous_temperature(momentum: np.ndarray, mass: float, kb: float, dof: int) -> float:
    """Instantaneous temperature from equipartition, T=2K/(dof*kb)."""

    return 2.0 * kinetic_energy(momentum, mass) / (dof * kb)


def simulate(config: SimulationConfig) -> dict[str, float | np.ndarray]:
    """Run Nose-Hoover dynamics and return summary statistics and traces."""

    if config.n_oscillators <= 0:
        raise ValueError("n_oscillators must be positive")
    if config.n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if config.dt <= 0.0:
        raise ValueError("dt must be positive")
    if config.sample_every <= 0:
        raise ValueError("sample_every must be positive")
    if config.thermostat_mass <= 0.0:
        raise ValueError("thermostat_mass must be positive")
    if config.burn_in_steps < 0 or config.burn_in_steps >= config.n_steps:
        raise ValueError("burn_in_steps must be in [0, n_steps)")

    rng = np.random.default_rng(config.seed)
    dof = config.n_oscillators

    # Initialize from the target canonical scales.
    position = rng.normal(
        loc=0.0,
        scale=np.sqrt(config.kb * config.target_temperature / config.k_spring),
        size=dof,
    )
    momentum = rng.normal(
        loc=0.0,
        scale=np.sqrt(config.mass * config.kb * config.target_temperature),
        size=dof,
    )

    xi = 0.0  # Thermostat friction variable.
    force = harmonic_force(position, config.k_spring)

    n_records = config.n_steps // config.sample_every
    temperatures = np.empty(n_records, dtype=float)
    x2_means = np.empty(n_records, dtype=float)
    mechanical_energies = np.empty(n_records, dtype=float)

    record_idx = 0
    for step in range(config.n_steps):
        # Symmetric splitting: xi(1/2) -> p(friction 1/2) -> p(force 1/2) -> q -> p(force 1/2) -> p(friction 1/2) -> xi(1/2)
        k_now = kinetic_energy(momentum, config.mass)
        xi += (
            0.5
            * config.dt
            * (2.0 * k_now - dof * config.kb * config.target_temperature)
            / config.thermostat_mass
        )

        momentum *= np.exp(-0.5 * xi * config.dt)
        momentum += 0.5 * config.dt * force
        position += config.dt * momentum / config.mass
        force = harmonic_force(position, config.k_spring)
        momentum += 0.5 * config.dt * force
        momentum *= np.exp(-0.5 * xi * config.dt)

        k_now = kinetic_energy(momentum, config.mass)
        xi += (
            0.5
            * config.dt
            * (2.0 * k_now - dof * config.kb * config.target_temperature)
            / config.thermostat_mass
        )

        if (step + 1) % config.sample_every == 0:
            temperatures[record_idx] = instantaneous_temperature(momentum, config.mass, config.kb, dof)
            x2_means[record_idx] = float(np.mean(position * position))
            mechanical_energies[record_idx] = k_now + 0.5 * config.k_spring * float(np.sum(position * position))
            record_idx += 1

    burn_idx = config.burn_in_steps // config.sample_every
    prod_temperatures = temperatures[burn_idx:]
    prod_x2_means = x2_means[burn_idx:]

    expected_x2 = config.kb * config.target_temperature / config.k_spring
    mean_temp = float(np.mean(prod_temperatures))
    temp_std = float(np.std(prod_temperatures))
    mean_x2 = float(np.mean(prod_x2_means))

    return {
        "target_temperature": config.target_temperature,
        "mean_temperature": mean_temp,
        "temperature_std": temp_std,
        "temperature_rel_error": abs(mean_temp - config.target_temperature) / config.target_temperature,
        "expected_x2": expected_x2,
        "mean_x2": mean_x2,
        "x2_rel_error": abs(mean_x2 - expected_x2) / expected_x2,
        "mean_mechanical_energy": float(np.mean(mechanical_energies[burn_idx:])),
        "temperatures": temperatures,
        "x2_means": x2_means,
    }


def main() -> None:
    config = SimulationConfig()
    result = simulate(config)

    print("Nose-Hoover Thermostat MVP (1D harmonic ensemble)")
    print(f"oscillators            : {config.n_oscillators}")
    print(f"steps                  : {config.n_steps}")
    print(f"dt                     : {config.dt}")
    print(f"target temperature     : {result['target_temperature']:.6f}")
    print(f"mean temperature       : {result['mean_temperature']:.6f}")
    print(f"temperature std        : {result['temperature_std']:.6f}")
    print(f"temperature rel. error : {result['temperature_rel_error']:.6%}")
    print(f"E[x^2] theory          : {result['expected_x2']:.6f}")
    print(f"E[x^2] simulation      : {result['mean_x2']:.6f}")
    print(f"x^2 rel. error         : {result['x2_rel_error']:.6%}")
    print(f"mean mechanical energy : {result['mean_mechanical_energy']:.6f}")


if __name__ == "__main__":
    main()

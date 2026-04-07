"""Minimal runnable MVP for the Andersen thermostat.

Model:
- 1D harmonic oscillator with mass m and spring constant k.
- Deterministic drift uses velocity Verlet integration.
- Thermostat step: with probability nu * dt, velocity is resampled
  from Maxwell-Boltzmann distribution N(0, k_B T / m).

Goal:
Demonstrate that long-time samples approach canonical statistics:
    <v^2> = k_B T / m
    <x^2> = k_B T / k
for the harmonic potential U(x) = 0.5 k x^2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AndersenConfig:
    mass: float = 1.0
    spring_k: float = 2.0
    target_temperature: float = 1.5
    boltzmann_k: float = 1.0
    collision_frequency: float = 2.0  # nu
    dt: float = 0.002
    n_steps: int = 140_000
    burn_in: int = 20_000
    seed: int = 20260407

    def validate(self) -> None:
        if self.mass <= 0.0:
            raise ValueError(f"mass must be positive, got {self.mass}")
        if self.spring_k <= 0.0:
            raise ValueError(f"spring_k must be positive, got {self.spring_k}")
        if self.target_temperature <= 0.0:
            raise ValueError(
                f"target_temperature must be positive, got {self.target_temperature}"
            )
        if self.boltzmann_k <= 0.0:
            raise ValueError(f"boltzmann_k must be positive, got {self.boltzmann_k}")
        if self.collision_frequency < 0.0:
            raise ValueError(
                "collision_frequency must be non-negative, "
                f"got {self.collision_frequency}"
            )
        if self.dt <= 0.0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        if self.n_steps < 10:
            raise ValueError(f"n_steps is too small, got {self.n_steps}")
        if not (0 <= self.burn_in < self.n_steps):
            raise ValueError(
                f"burn_in must satisfy 0 <= burn_in < n_steps, got {self.burn_in}"
            )

        collision_probability = self.collision_frequency * self.dt
        if collision_probability > 1.0:
            raise ValueError(
                "collision_frequency * dt must be <= 1 for Bernoulli collision model, "
                f"got {collision_probability}"
            )


@dataclass(frozen=True)
class SimulationResult:
    positions: np.ndarray
    velocities: np.ndarray
    kinetic_energy: np.ndarray
    potential_energy: np.ndarray
    collision_events: np.ndarray
    summary: pd.DataFrame


def acceleration(x: float, cfg: AndersenConfig) -> float:
    return -(cfg.spring_k / cfg.mass) * x


def maxwell_sigma(cfg: AndersenConfig) -> float:
    return np.sqrt(cfg.boltzmann_k * cfg.target_temperature / cfg.mass)


def verlet_andersen_step(
    x: float,
    v: float,
    cfg: AndersenConfig,
    rng: np.random.Generator,
    sigma_v: float,
) -> tuple[float, float, bool]:
    """One integration step: velocity Verlet + Andersen collision."""
    dt = cfg.dt

    a_old = acceleration(x, cfg)
    v_half = v + 0.5 * dt * a_old
    x_new = x + dt * v_half
    a_new = acceleration(x_new, cfg)
    v_new = v_half + 0.5 * dt * a_new

    collided = bool(rng.random() < cfg.collision_frequency * dt)
    if collided:
        v_new = float(rng.normal(loc=0.0, scale=sigma_v))

    return x_new, v_new, collided


def run_simulation(cfg: AndersenConfig) -> SimulationResult:
    cfg.validate()

    rng = np.random.default_rng(cfg.seed)
    sigma_v = maxwell_sigma(cfg)

    x = 4.0 * np.sqrt(cfg.boltzmann_k * cfg.target_temperature / cfg.spring_k)
    v = 0.0

    positions = np.empty(cfg.n_steps, dtype=np.float64)
    velocities = np.empty(cfg.n_steps, dtype=np.float64)
    kinetic_energy = np.empty(cfg.n_steps, dtype=np.float64)
    potential_energy = np.empty(cfg.n_steps, dtype=np.float64)
    collision_events = np.empty(cfg.n_steps, dtype=np.bool_)

    for i in range(cfg.n_steps):
        x, v, collided = verlet_andersen_step(x, v, cfg, rng, sigma_v)
        positions[i] = x
        velocities[i] = v
        kinetic_energy[i] = 0.5 * cfg.mass * v * v
        potential_energy[i] = 0.5 * cfg.spring_k * x * x
        collision_events[i] = collided

    idx = slice(cfg.burn_in, None)
    x_eq = positions[idx]
    v_eq = velocities[idx]

    sample_var_x = float(np.mean(x_eq * x_eq))
    sample_var_v = float(np.mean(v_eq * v_eq))
    target_var_x = cfg.boltzmann_k * cfg.target_temperature / cfg.spring_k
    target_var_v = cfg.boltzmann_k * cfg.target_temperature / cfg.mass

    sample_temperature = cfg.mass * sample_var_v / cfg.boltzmann_k
    empirical_collision_frequency = float(np.mean(collision_events)) / cfg.dt

    summary = pd.DataFrame(
        {
            "metric": [
                "target_temperature",
                "sample_temperature",
                "target_var_x",
                "sample_var_x",
                "target_var_v",
                "sample_var_v",
                "target_collision_frequency",
                "empirical_collision_frequency",
                "mean_kinetic_energy_eq",
                "mean_potential_energy_eq",
            ],
            "value": [
                cfg.target_temperature,
                sample_temperature,
                target_var_x,
                sample_var_x,
                target_var_v,
                sample_var_v,
                cfg.collision_frequency,
                empirical_collision_frequency,
                float(np.mean(kinetic_energy[idx])),
                float(np.mean(potential_energy[idx])),
            ],
        }
    )

    return SimulationResult(
        positions=positions,
        velocities=velocities,
        kinetic_energy=kinetic_energy,
        potential_energy=potential_energy,
        collision_events=collision_events,
        summary=summary,
    )


def value_from_summary(summary: pd.DataFrame, metric: str) -> float:
    row = summary.loc[summary["metric"] == metric, "value"]
    if row.empty:
        raise KeyError(f"Metric {metric} not found in summary")
    return float(row.iloc[0])


def relative_error(observed: float, target: float) -> float:
    if target == 0.0:
        return abs(observed - target)
    return abs(observed - target) / abs(target)


def main() -> None:
    cfg = AndersenConfig()
    result = run_simulation(cfg)

    summary = result.summary
    temp_target = value_from_summary(summary, "target_temperature")
    temp_sample = value_from_summary(summary, "sample_temperature")
    var_x_target = value_from_summary(summary, "target_var_x")
    var_x_sample = value_from_summary(summary, "sample_var_x")
    var_v_target = value_from_summary(summary, "target_var_v")
    var_v_sample = value_from_summary(summary, "sample_var_v")
    nu_target = value_from_summary(summary, "target_collision_frequency")
    nu_sample = value_from_summary(summary, "empirical_collision_frequency")

    err_temp = relative_error(temp_sample, temp_target)
    err_var_x = relative_error(var_x_sample, var_x_target)
    err_var_v = relative_error(var_v_sample, var_v_target)
    err_nu = relative_error(nu_sample, nu_target)

    print("Andersen Thermostat MVP")
    print(
        f"mass={cfg.mass:.3f}, spring_k={cfg.spring_k:.3f}, "
        f"target_temperature={cfg.target_temperature:.3f}"
    )
    print(
        f"dt={cfg.dt:.4f}, n_steps={cfg.n_steps}, burn_in={cfg.burn_in}, "
        f"collision_frequency={cfg.collision_frequency:.3f}"
    )
    print("--- equilibrium checks ---")
    print(f"sample_temperature={temp_sample:.6f} (target={temp_target:.6f}, rel_err={err_temp:.3%})")
    print(f"sample_var_x={var_x_sample:.6f} (target={var_x_target:.6f}, rel_err={err_var_x:.3%})")
    print(f"sample_var_v={var_v_sample:.6f} (target={var_v_target:.6f}, rel_err={err_var_v:.3%})")
    print(f"empirical_nu={nu_sample:.6f} (target={nu_target:.6f}, rel_err={err_nu:.3%})")
    print("--- energy means (post burn-in) ---")
    print(
        "mean_kinetic_energy="
        f"{value_from_summary(summary, 'mean_kinetic_energy_eq'):.6f}, "
        "mean_potential_energy="
        f"{value_from_summary(summary, 'mean_potential_energy_eq'):.6f}"
    )

    assert err_temp < 0.06, f"Temperature mismatch too large: {err_temp:.3%}"
    assert err_var_x < 0.08, f"Position variance mismatch too large: {err_var_x:.3%}"
    assert err_var_v < 0.06, f"Velocity variance mismatch too large: {err_var_v:.3%}"
    assert err_nu < 0.08, f"Collision frequency mismatch too large: {err_nu:.3%}"

    print("All checks passed.")


if __name__ == "__main__":
    main()

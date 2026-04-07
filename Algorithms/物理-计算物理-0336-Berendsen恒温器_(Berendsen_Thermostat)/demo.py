"""Minimal runnable MVP for Berendsen thermostat (PHYS-0329).

This demo simulates independent 3D harmonic oscillators with velocity-Verlet
integration, then applies Berendsen velocity rescaling to weakly couple the
system to a target heat bath temperature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class SimulationConfig:
    """Configuration for the molecular dynamics toy system."""

    n_particles: int = 64
    dim: int = 3
    mass: float = 1.0
    spring_k: float = 1.2
    dt: float = 0.002
    steps: int = 2500
    k_boltz: float = 1.0
    target_temperature: float = 1.0
    tau_temperature: float = 0.08


@dataclass
class SimulationHistory:
    """Time series produced by one simulation run."""

    temperature: np.ndarray
    kinetic: np.ndarray
    potential: np.ndarray
    total_energy: np.ndarray
    lambda_scale: np.ndarray


def sample_initial_state(
    cfg: SimulationConfig,
    init_temperature: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create initial positions and velocities.

    Positions are sampled from a normal distribution. Velocities are sampled
    from Maxwell-like Gaussian then rescaled to the specified initial
    temperature and zero net momentum.
    """
    rng = np.random.default_rng(seed)

    positions = rng.normal(loc=0.0, scale=0.8, size=(cfg.n_particles, cfg.dim))
    velocities = rng.normal(
        loc=0.0,
        scale=np.sqrt(cfg.k_boltz * init_temperature / cfg.mass),
        size=(cfg.n_particles, cfg.dim),
    )

    # Remove center-of-mass drift to avoid counting bulk translation as heat.
    velocities -= velocities.mean(axis=0, keepdims=True)

    current_t = instantaneous_temperature(velocities, cfg)
    scale = np.sqrt(init_temperature / max(current_t, 1e-12))
    velocities *= scale
    return positions, velocities


def compute_forces(positions: np.ndarray, cfg: SimulationConfig) -> np.ndarray:
    """Harmonic restoring force: F = -k x for each particle coordinate."""
    return -cfg.spring_k * positions


def kinetic_energy(velocities: np.ndarray, cfg: SimulationConfig) -> float:
    """Compute kinetic energy K = 0.5 m sum(v^2)."""
    return 0.5 * cfg.mass * float(np.sum(velocities * velocities))


def potential_energy(positions: np.ndarray, cfg: SimulationConfig) -> float:
    """Compute harmonic potential U = 0.5 k sum(x^2)."""
    return 0.5 * cfg.spring_k * float(np.sum(positions * positions))


def instantaneous_temperature(velocities: np.ndarray, cfg: SimulationConfig) -> float:
    """Estimate instantaneous temperature from kinetic energy.

    We use dof = N * dim - dim because center-of-mass momentum is constrained.
    """
    dof = cfg.n_particles * cfg.dim - cfg.dim
    ke = kinetic_energy(velocities, cfg)
    return 2.0 * ke / (dof * cfg.k_boltz)


def berendsen_scale_factor(
    current_temperature: float,
    target_temperature: float,
    dt: float,
    tau_temperature: float,
) -> float:
    """Compute Berendsen velocity scaling factor.

    lambda^2 = 1 + dt/tau * (T0/T - 1)
    """
    t_now = max(current_temperature, 1e-12)
    lambda_sq = 1.0 + (dt / tau_temperature) * (target_temperature / t_now - 1.0)
    lambda_sq = max(lambda_sq, 1e-12)
    return float(np.sqrt(lambda_sq))


def run_simulation(
    cfg: SimulationConfig,
    positions0: np.ndarray,
    velocities0: np.ndarray,
    use_thermostat: bool,
) -> SimulationHistory:
    """Run MD simulation with or without Berendsen thermostat."""
    positions = positions0.copy()
    velocities = velocities0.copy()
    forces = compute_forces(positions, cfg)

    temp_hist = np.empty(cfg.steps, dtype=np.float64)
    ke_hist = np.empty(cfg.steps, dtype=np.float64)
    pe_hist = np.empty(cfg.steps, dtype=np.float64)
    e_hist = np.empty(cfg.steps, dtype=np.float64)
    lam_hist = np.ones(cfg.steps, dtype=np.float64)

    inv_mass = 1.0 / cfg.mass

    for step in range(cfg.steps):
        # Velocity-Verlet integration before thermostat coupling.
        velocities_half = velocities + 0.5 * cfg.dt * forces * inv_mass
        positions = positions + cfg.dt * velocities_half
        new_forces = compute_forces(positions, cfg)
        velocities = velocities_half + 0.5 * cfg.dt * new_forces * inv_mass

        temp_before = instantaneous_temperature(velocities, cfg)
        lam = 1.0
        if use_thermostat:
            lam = berendsen_scale_factor(
                current_temperature=temp_before,
                target_temperature=cfg.target_temperature,
                dt=cfg.dt,
                tau_temperature=cfg.tau_temperature,
            )
            velocities *= lam

        forces = new_forces

        ke = kinetic_energy(velocities, cfg)
        pe = potential_energy(positions, cfg)
        temp_hist[step] = instantaneous_temperature(velocities, cfg)
        ke_hist[step] = ke
        pe_hist[step] = pe
        e_hist[step] = ke + pe
        lam_hist[step] = lam

    return SimulationHistory(
        temperature=temp_hist,
        kinetic=ke_hist,
        potential=pe_hist,
        total_energy=e_hist,
        lambda_scale=lam_hist,
    )


def summarize_history(name: str, history: SimulationHistory, cfg: SimulationConfig) -> Dict[str, float]:
    """Compute summary statistics for a run."""
    tail = 500
    t_mean_tail = float(np.mean(history.temperature[-tail:]))
    t_std_tail = float(np.std(history.temperature[-tail:]))
    e_drift = float(history.total_energy[-1] - history.total_energy[0])
    lam_mean = float(np.mean(history.lambda_scale))

    summary = {
        "final_temperature": float(history.temperature[-1]),
        "tail_mean_temperature": t_mean_tail,
        "tail_std_temperature": t_std_tail,
        "energy_drift": e_drift,
        "mean_lambda": lam_mean,
    }

    print(f"[{name}] final T = {summary['final_temperature']:.4f}")
    print(
        f"[{name}] tail mean/std T = "
        f"{summary['tail_mean_temperature']:.4f} / {summary['tail_std_temperature']:.4f}"
    )
    print(f"[{name}] energy drift (E_end-E_start) = {summary['energy_drift']:.4f}")
    print(f"[{name}] mean lambda = {summary['mean_lambda']:.6f}")
    return summary


def main() -> None:
    print("Berendsen Thermostat MVP (PHYS-0329)")
    print("=" * 72)

    cfg = SimulationConfig(
        n_particles=64,
        dim=3,
        mass=1.0,
        spring_k=1.2,
        dt=0.002,
        steps=2500,
        target_temperature=1.0,
        tau_temperature=0.08,
    )
    init_temperature = 4.0

    positions0, velocities0 = sample_initial_state(cfg, init_temperature=init_temperature, seed=336)
    init_t = instantaneous_temperature(velocities0, cfg)
    print(f"initial temperature after calibration: {init_t:.4f}")
    print(f"target temperature: {cfg.target_temperature:.4f}")
    print(f"dt/tau = {cfg.dt / cfg.tau_temperature:.4f}")
    print("-" * 72)

    history_nve = run_simulation(cfg, positions0, velocities0, use_thermostat=False)
    summary_nve = summarize_history("NVE(no thermostat)", history_nve, cfg)

    print("-" * 72)

    history_ber = run_simulation(cfg, positions0, velocities0, use_thermostat=True)
    summary_ber = summarize_history("Berendsen", history_ber, cfg)

    print("-" * 72)
    dist_ber = abs(summary_ber["tail_mean_temperature"] - cfg.target_temperature)
    dist_nve = abs(summary_nve["tail_mean_temperature"] - cfg.target_temperature)

    print(f"distance to target (Berendsen): {dist_ber:.4f}")
    print(f"distance to target (NVE): {dist_nve:.4f}")

    if not (dist_ber < 0.12):
        raise RuntimeError("Berendsen thermostat failed to regulate temperature near target.")

    if not (dist_ber < dist_nve):
        raise RuntimeError(
            "Berendsen run is not closer to target temperature than no-thermostat baseline."
        )

    if not (summary_ber["tail_std_temperature"] < summary_nve["tail_std_temperature"]):
        raise RuntimeError("Expected Berendsen thermostat to suppress temperature fluctuations.")

    print("All checks passed.")


if __name__ == "__main__":
    main()

"""Minimal runnable MVP for Newton's law of universal gravitation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.constants import G
from sklearn.linear_model import LinearRegression


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for a two-body Earth-Moon simulation."""

    dt_seconds: float = 1800.0
    num_steps: int = 2400


def gravitational_force_vector(
    m1: float,
    m2: float,
    pos1: np.ndarray,
    pos2: np.ndarray,
) -> np.ndarray:
    """Force on body-1 due to body-2 from Newton's inverse-square law."""
    r_vec = pos2 - pos1
    dist = float(np.linalg.norm(r_vec))
    if dist <= 0.0:
        raise ValueError("Body distance must be positive.")
    return G * m1 * m2 * r_vec / (dist**3)


def accelerations(positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Compute accelerations for all bodies using pairwise gravity."""
    n_bodies = positions.shape[0]
    acc = np.zeros_like(positions, dtype=np.float64)

    for i in range(n_bodies):
        for j in range(i + 1, n_bodies):
            r_vec = positions[j] - positions[i]
            dist = float(np.linalg.norm(r_vec))
            if dist <= 0.0:
                raise ValueError("Encountered overlapping bodies.")
            inv_dist3 = 1.0 / (dist**3)
            pair = G * r_vec * inv_dist3
            acc[i] += pair * masses[j]
            acc[j] -= pair * masses[i]
    return acc


def total_energy(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
) -> float:
    """Total mechanical energy (kinetic + potential) of an N-body state."""
    kinetic = 0.5 * np.sum(masses[:, None] * velocities * velocities)
    potential = 0.0
    n_bodies = positions.shape[0]
    for i in range(n_bodies):
        for j in range(i + 1, n_bodies):
            dist = float(np.linalg.norm(positions[j] - positions[i]))
            potential -= G * masses[i] * masses[j] / dist
    return float(kinetic + potential)


def total_angular_momentum(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
) -> np.ndarray:
    """Angular momentum vector sum(m_i * r_i x v_i)."""
    return np.sum(np.cross(positions, masses[:, None] * velocities), axis=0)


def velocity_verlet(
    masses: np.ndarray,
    positions0: np.ndarray,
    velocities0: np.ndarray,
    cfg: SimulationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Integrate trajectories with the explicit Velocity-Verlet scheme."""
    if cfg.dt_seconds <= 0.0:
        raise ValueError("dt_seconds must be positive.")
    if cfg.num_steps <= 0:
        raise ValueError("num_steps must be positive.")

    n_bodies = masses.size
    times = np.arange(cfg.num_steps + 1, dtype=np.float64) * cfg.dt_seconds
    positions = np.zeros((cfg.num_steps + 1, n_bodies, 3), dtype=np.float64)
    velocities = np.zeros_like(positions)
    energies = np.zeros(cfg.num_steps + 1, dtype=np.float64)
    ang_mom_norm = np.zeros(cfg.num_steps + 1, dtype=np.float64)

    positions[0] = positions0
    velocities[0] = velocities0

    acc = accelerations(positions[0], masses)
    energies[0] = total_energy(positions[0], velocities[0], masses)
    ang_mom_norm[0] = float(
        np.linalg.norm(total_angular_momentum(positions[0], velocities[0], masses))
    )

    for k in range(cfg.num_steps):
        pos_next = (
            positions[k] + velocities[k] * cfg.dt_seconds + 0.5 * acc * cfg.dt_seconds**2
        )
        acc_next = accelerations(pos_next, masses)
        vel_next = velocities[k] + 0.5 * (acc + acc_next) * cfg.dt_seconds

        positions[k + 1] = pos_next
        velocities[k + 1] = vel_next
        energies[k + 1] = total_energy(pos_next, vel_next, masses)
        ang_mom_norm[k + 1] = float(
            np.linalg.norm(total_angular_momentum(pos_next, vel_next, masses))
        )
        acc = acc_next

    return times, positions, velocities, energies, ang_mom_norm


def estimate_orbital_period(relative_positions: np.ndarray, times: np.ndarray) -> float:
    """Estimate orbital period from unwrapped polar angle slope."""
    angles = np.unwrap(np.arctan2(relative_positions[:, 1], relative_positions[:, 0]))
    omega = (angles[-1] - angles[0]) / (times[-1] - times[0])
    if omega <= 0.0:
        raise RuntimeError("Estimated angular speed must be positive.")
    return 2.0 * math.pi / omega


def inverse_square_regression(m1: float, m2: float) -> tuple[float, float, float]:
    """Fit log10(F)=a*log10(r)+b to verify inverse-square slope."""
    r_values = np.linspace(2.0e8, 1.2e9, 24, dtype=np.float64)
    f_values = G * m1 * m2 / (r_values**2)

    x = np.log10(r_values).reshape(-1, 1)
    y = np.log10(f_values)

    model = LinearRegression()
    model.fit(x, y)
    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    r2 = float(model.score(x, y))
    return slope, intercept, r2


def torch_force_consistency(
    m1: float,
    m2: float,
    pos1: np.ndarray,
    pos2: np.ndarray,
) -> float:
    """Cross-check one force vector computed by NumPy vs PyTorch."""
    force_np = gravitational_force_vector(m1, m2, pos1, pos2)

    t_pos1 = torch.tensor(pos1, dtype=torch.float64)
    t_pos2 = torch.tensor(pos2, dtype=torch.float64)
    t_r = t_pos2 - t_pos1
    t_dist = torch.linalg.norm(t_r)
    force_t = G * m1 * m2 * t_r / (t_dist**3)

    return float(np.linalg.norm(force_np - force_t.numpy()))


def main() -> None:
    cfg = SimulationConfig()

    # Earth-Moon two-body system in center-of-mass coordinates.
    m_earth = 5.9722e24
    m_moon = 7.34767309e22
    masses = np.array([m_earth, m_moon], dtype=np.float64)

    distance0 = 384_400_000.0
    total_mass = masses.sum()
    reduced_speed = math.sqrt(G * total_mass / distance0)

    positions0 = np.array(
        [
            [-distance0 * m_moon / total_mass, 0.0, 0.0],
            [distance0 * m_earth / total_mass, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    velocities0 = np.array(
        [
            [0.0, -reduced_speed * m_moon / total_mass, 0.0],
            [0.0, reduced_speed * m_earth / total_mass, 0.0],
        ],
        dtype=np.float64,
    )

    times, positions, velocities, energies, ang_mom_norm = velocity_verlet(
        masses, positions0, velocities0, cfg
    )

    rel = positions[:, 1] - positions[:, 0]
    rel_dist = np.linalg.norm(rel, axis=1)
    force_magnitude = G * m_earth * m_moon / (rel_dist**2)

    period_numeric = estimate_orbital_period(rel, times)
    period_theory = 2.0 * math.pi * math.sqrt(distance0**3 / (G * total_mass))

    energy_drift = float(np.max(np.abs(energies - energies[0])) / abs(energies[0]))
    ang_mom_drift = float(
        np.max(np.abs(ang_mom_norm - ang_mom_norm[0])) / abs(ang_mom_norm[0])
    )

    slope, intercept, r2 = inverse_square_regression(m_earth, m_moon)
    torch_err = torch_force_consistency(m_earth, m_moon, positions0[0], positions0[1])

    period_rel_error = abs(period_numeric - period_theory) / period_theory

    summary = pd.DataFrame(
        [
            ("initial_distance_km", distance0 / 1_000.0, "km"),
            ("mean_distance_km", float(np.mean(rel_dist) / 1_000.0), "km"),
            ("initial_force_N", float(force_magnitude[0]), "N"),
            ("mean_force_N", float(np.mean(force_magnitude)), "N"),
            ("period_theory_days", period_theory / 86_400.0, "days"),
            ("period_numeric_days", period_numeric / 86_400.0, "days"),
            ("period_relative_error", period_rel_error, "ratio"),
            ("energy_rel_drift", energy_drift, "ratio"),
            ("angular_momentum_rel_drift", ang_mom_drift, "ratio"),
            ("inverse_square_slope", slope, "expected -2"),
            ("inverse_square_intercept", intercept, "log10 scale"),
            ("inverse_square_r2", r2, "[0,1]"),
            ("torch_numpy_force_diff", torch_err, "N"),
        ],
        columns=["metric", "value", "unit"],
    )

    sample_idx = np.linspace(0, cfg.num_steps, 6, dtype=int)
    trajectory_view = pd.DataFrame(
        {
            "t_days": times[sample_idx] / 86_400.0,
            "distance_km": rel_dist[sample_idx] / 1_000.0,
            "force_N": force_magnitude[sample_idx],
            "moon_x_km": positions[sample_idx, 1, 0] / 1_000.0,
            "moon_y_km": positions[sample_idx, 1, 1] / 1_000.0,
        }
    )

    print("=== Newton Universal Gravitation: Earth-Moon MVP ===")
    print(summary.to_string(index=False, justify="left", float_format=lambda x: f"{x:.8e}"))
    print("\n=== Trajectory Snapshot (Moon in COM frame) ===")
    print(trajectory_view.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    if period_rel_error > 0.05:
        raise RuntimeError("Numerical period deviates too much from Kepler prediction.")
    if energy_drift > 1e-3:
        raise RuntimeError("Energy drift is too large for this MVP setup.")
    if ang_mom_drift > 1e-8:
        raise RuntimeError("Angular momentum drift is too large.")
    if abs(slope + 2.0) > 1e-10:
        raise RuntimeError("Inverse-square regression slope is inconsistent with theory.")
    if torch_err > 1e-6:
        raise RuntimeError("Torch and NumPy force results diverge unexpectedly.")


if __name__ == "__main__":
    main()

"""Minimal runnable MVP for Dissipative Particle Dynamics (PHYS-0336).

This script demonstrates a compact 2D DPD simulation with:
- pairwise conservative + dissipative + random forces
- fluctuation-dissipation relation: sigma^2 = 2 * gamma * kB * T
- periodic boundary conditions (minimum-image convention)
- non-interactive run with printed diagnostics and assertions
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DPDConfig:
    n_particles: int = 64
    dim: int = 2
    box_length: float = 10.0

    dt: float = 0.004
    steps: int = 3000
    report_every: int = 300

    rc: float = 1.0
    conservative_a: float = 5.0
    gamma: float = 6.0

    mass: float = 1.0
    kbt: float = 1.0

    seed: int = 7

    def validate(self) -> None:
        if self.n_particles < 8:
            raise ValueError("n_particles must be >= 8")
        if self.dim != 2:
            raise ValueError("This MVP currently supports dim=2 only")
        if self.box_length <= 0.0:
            raise ValueError("box_length must be positive")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.report_every <= 0:
            raise ValueError("report_every must be positive")
        if self.rc <= 0.0:
            raise ValueError("rc must be positive")
        if self.conservative_a < 0.0:
            raise ValueError("conservative_a must be non-negative")
        if self.gamma <= 0.0:
            raise ValueError("gamma must be positive")
        if self.mass <= 0.0:
            raise ValueError("mass must be positive")
        if self.kbt <= 0.0:
            raise ValueError("kbt must be positive")


@dataclass
class DPDState:
    positions: np.ndarray
    velocities: np.ndarray


def minimum_image(displacement: np.ndarray, box_length: float) -> np.ndarray:
    """Apply minimum-image convention under periodic boundary conditions."""
    return displacement - box_length * np.round(displacement / box_length)


def initialize_state(cfg: DPDConfig, rng: np.random.Generator) -> DPDState:
    """Randomly initialize positions and Maxwell-like velocities."""
    positions = rng.uniform(0.0, cfg.box_length, size=(cfg.n_particles, cfg.dim))

    velocity_std = np.sqrt(cfg.kbt / cfg.mass)
    velocities = rng.normal(0.0, velocity_std, size=(cfg.n_particles, cfg.dim))

    # Remove center-of-mass drift so total momentum starts near zero.
    velocities -= np.mean(velocities, axis=0, keepdims=True)
    return DPDState(positions=positions, velocities=velocities)


def compute_pair_forces(
    positions: np.ndarray,
    velocities: np.ndarray,
    cfg: DPDConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, int]:
    """Compute DPD pair forces and conservative potential energy.

    For each pair i<j within cutoff rc:
    F_ij = (F_C + F_D + F_R) * e_ij
    where
    F_C = a * (1 - r/rc)
    F_D = -gamma * (1 - r/rc)^2 * ((v_ij · e_ij))
    F_R = sigma * (1 - r/rc) * theta_ij / sqrt(dt)
    sigma^2 = 2 * gamma * kBT
    """
    n = positions.shape[0]
    forces = np.zeros_like(positions)
    potential_energy = 0.0
    pair_count = 0

    sigma = np.sqrt(2.0 * cfg.gamma * cfg.kbt)

    for i in range(n - 1):
        for j in range(i + 1, n):
            rij = positions[j] - positions[i]
            rij = minimum_image(rij, cfg.box_length)

            r2 = float(np.dot(rij, rij))
            if r2 <= 1.0e-15:
                continue

            r = np.sqrt(r2)
            if r >= cfg.rc:
                continue

            pair_count += 1
            eij = rij / r
            vij = velocities[i] - velocities[j]
            vij_dot_eij = float(np.dot(vij, eij))

            wr = 1.0 - r / cfg.rc
            wd = wr * wr

            f_conservative = cfg.conservative_a * wr
            f_dissipative = -cfg.gamma * wd * vij_dot_eij
            theta_ij = float(rng.normal(0.0, 1.0))
            f_random = sigma * wr * theta_ij / np.sqrt(cfg.dt)

            fij = (f_conservative + f_dissipative + f_random) * eij

            forces[i] += fij
            forces[j] -= fij

            potential_energy += 0.5 * cfg.conservative_a * wr * wr

    return forces, potential_energy, pair_count


def kinetic_temperature(velocities: np.ndarray, mass: float, kbt: float) -> float:
    """Instantaneous kinetic temperature in reduced units."""
    n, dim = velocities.shape
    dof = max(dim * (n - 1), 1)  # remove COM dof approximately
    kinetic_energy = 0.5 * mass * float(np.sum(velocities * velocities))
    return (2.0 * kinetic_energy) / (dof * kbt)


def radial_distribution_function(
    positions: np.ndarray,
    box_length: float,
    r_max: float,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate 2D radial distribution function g(r) from one snapshot."""
    n = positions.shape[0]
    dr = r_max / n_bins
    edges = np.linspace(0.0, r_max, n_bins + 1)
    hist = np.zeros(n_bins, dtype=np.float64)

    for i in range(n - 1):
        for j in range(i + 1, n):
            rij = minimum_image(positions[j] - positions[i], box_length)
            r = float(np.linalg.norm(rij))
            if 0.0 < r < r_max:
                bin_idx = int(r / dr)
                if bin_idx >= n_bins:
                    bin_idx = n_bins - 1
                hist[bin_idx] += 2.0

    density = n / (box_length * box_length)
    radii = 0.5 * (edges[:-1] + edges[1:])

    shell_area = 2.0 * np.pi * radii * dr
    normalization = n * density * shell_area

    g_r = np.divide(hist, normalization, out=np.zeros_like(hist), where=normalization > 0)
    return radii, g_r


def run_simulation(cfg: DPDConfig) -> tuple[DPDState, pd.DataFrame]:
    """Run the DPD simulation with explicit Euler-Maruyama integration."""
    cfg.validate()
    rng = np.random.default_rng(cfg.seed)
    state = initialize_state(cfg, rng)

    history_rows: list[dict[str, float | int]] = []

    for step in range(1, cfg.steps + 1):
        forces, pe_conservative, pair_count = compute_pair_forces(
            positions=state.positions,
            velocities=state.velocities,
            cfg=cfg,
            rng=rng,
        )

        # Euler-Maruyama update for stochastic dynamics.
        state.velocities += (forces / cfg.mass) * cfg.dt
        state.positions = (state.positions + state.velocities * cfg.dt) % cfg.box_length

        if step % cfg.report_every == 0 or step == cfg.steps:
            temp = kinetic_temperature(state.velocities, cfg.mass, cfg.kbt)
            vcm = np.mean(state.velocities, axis=0)
            com_speed = float(np.linalg.norm(vcm))
            rms_speed = float(np.sqrt(np.mean(np.sum(state.velocities**2, axis=1))))

            history_rows.append(
                {
                    "step": step,
                    "temperature": float(temp),
                    "potential_energy": float(pe_conservative),
                    "pair_count": int(pair_count),
                    "rms_speed": rms_speed,
                    "com_speed": com_speed,
                }
            )

    history = pd.DataFrame(history_rows)
    return state, history


def main() -> None:
    cfg = DPDConfig()
    final_state, history = run_simulation(cfg)

    temp_final = float(history.iloc[-1]["temperature"])
    temp_mean = float(history["temperature"].mean())
    pair_mean = float(history["pair_count"].mean())
    com_speed_max = float(history["com_speed"].max())

    r_vals, g_r = radial_distribution_function(
        positions=final_state.positions,
        box_length=cfg.box_length,
        r_max=min(2.5, 0.5 * cfg.box_length),
        n_bins=24,
    )

    # A coarse local-structure proxy from g(r) near the cutoff range.
    near_cutoff = (r_vals > 0.7 * cfg.rc) & (r_vals < 1.2 * cfg.rc)
    g_peak_near_rc = float(np.max(g_r[near_cutoff])) if np.any(near_cutoff) else float(np.max(g_r))

    summary = pd.DataFrame(
        {
            "metric": [
                "n_particles",
                "box_length",
                "density",
                "steps",
                "dt",
                "rc",
                "a",
                "gamma",
                "sigma",
                "temperature_mean",
                "temperature_final",
                "pair_count_mean",
                "com_speed_max",
                "g_peak_near_rc",
            ],
            "value": [
                cfg.n_particles,
                cfg.box_length,
                cfg.n_particles / (cfg.box_length**2),
                cfg.steps,
                cfg.dt,
                cfg.rc,
                cfg.conservative_a,
                cfg.gamma,
                np.sqrt(2.0 * cfg.gamma * cfg.kbt),
                temp_mean,
                temp_final,
                pair_mean,
                com_speed_max,
                g_peak_near_rc,
            ],
        }
    )

    print("DPD 2D MVP")
    print(history.to_string(index=False))
    print("\nSummary")
    print(summary.to_string(index=False))

    # Basic validation gates for this lightweight MVP.
    if not np.all(np.isfinite(final_state.positions)):
        raise AssertionError("positions contain non-finite values")
    if not np.all(np.isfinite(final_state.velocities)):
        raise AssertionError("velocities contain non-finite values")

    if not (0.50 <= temp_mean <= 1.80):
        raise AssertionError(f"mean temperature out of range: {temp_mean:.4f}")
    if com_speed_max >= 0.25:
        raise AssertionError(f"center-of-mass drift too large: {com_speed_max:.4f}")
    if pair_mean <= 20.0:
        raise AssertionError(f"too few interacting pairs on average: {pair_mean:.2f}")

    print("\nValidation: PASS")


if __name__ == "__main__":
    main()

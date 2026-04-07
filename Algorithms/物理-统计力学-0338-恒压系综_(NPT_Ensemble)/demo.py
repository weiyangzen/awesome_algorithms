"""Minimal MVP for the NPT ensemble (constant N, P, T).

This script implements a 2D Lennard-Jones fluid sampled by
Metropolis Monte Carlo in the isothermal-isobaric ensemble:
- Particle displacement moves (fixed volume)
- Isotropic volume-change moves (all coordinates scaled)

It runs two pressures and verifies the expected trend:
higher pressure -> lower average volume.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NPTConfig:
    n_particles: int = 36
    dim: int = 2
    temperature: float = 1.1
    pressure: float = 0.8
    init_density: float = 0.70
    sigma: float = 1.0
    epsilon: float = 1.0
    cutoff: float = 2.5
    move_step: float = 0.18
    volume_step: float = 1.4
    min_volume: float = 20.0
    burn_in_sweeps: int = 350
    sample_sweeps: int = 900
    sample_interval: int = 10
    seed: int = 20260407


@dataclass(frozen=True)
class RunSummary:
    pressure: float
    mean_volume: float
    mean_density: float
    mean_energy_per_particle: float
    mean_enthalpy_per_particle: float
    particle_acceptance: float
    volume_acceptance: float
    n_samples: int


def minimum_image(delta: np.ndarray, box_length: float) -> np.ndarray:
    """Apply periodic boundary condition via minimum-image convention."""
    return delta - box_length * np.rint(delta / box_length)


def lj_pair_energy(
    r2: float,
    sigma: float,
    epsilon: float,
    cutoff: float,
) -> float:
    """Shifted Lennard-Jones potential for a single pair."""
    cutoff2 = cutoff * cutoff
    if r2 >= cutoff2:
        return 0.0

    inv_r2 = (sigma * sigma) / r2
    inv_r6 = inv_r2 * inv_r2 * inv_r2
    inv_r12 = inv_r6 * inv_r6
    raw = 4.0 * epsilon * (inv_r12 - inv_r6)

    inv_rc2 = (sigma * sigma) / cutoff2
    inv_rc6 = inv_rc2 * inv_rc2 * inv_rc2
    inv_rc12 = inv_rc6 * inv_rc6
    shift = 4.0 * epsilon * (inv_rc12 - inv_rc6)
    return raw - shift


def total_energy(
    coords: np.ndarray,
    box_length: float,
    sigma: float,
    epsilon: float,
    cutoff: float,
) -> float:
    """Total pair potential energy (O(N^2))."""
    n = coords.shape[0]
    e_total = 0.0
    for i in range(n - 1):
        for j in range(i + 1, n):
            d = minimum_image(coords[i] - coords[j], box_length)
            r2 = float(np.dot(d, d))
            e_total += lj_pair_energy(r2, sigma, epsilon, cutoff)
    return e_total


def local_energy_of_particle(
    idx: int,
    trial_pos: np.ndarray,
    coords: np.ndarray,
    box_length: float,
    sigma: float,
    epsilon: float,
    cutoff: float,
) -> float:
    """Energy contribution between one particle and all others."""
    n = coords.shape[0]
    e_local = 0.0
    for j in range(n):
        if j == idx:
            continue
        d = minimum_image(trial_pos - coords[j], box_length)
        r2 = float(np.dot(d, d))
        e_local += lj_pair_energy(r2, sigma, epsilon, cutoff)
    return e_local


def init_lattice_positions(n_particles: int, dim: int, box_length: float) -> np.ndarray:
    """Create a simple square lattice as initial coordinates in [0, L)."""
    if dim != 2:
        raise ValueError("This MVP currently supports dim=2 only.")

    side = int(math.ceil(math.sqrt(n_particles)))
    spacing = box_length / side
    positions: list[list[float]] = []

    for ix in range(side):
        for iy in range(side):
            if len(positions) >= n_particles:
                break
            positions.append([(ix + 0.5) * spacing, (iy + 0.5) * spacing])
        if len(positions) >= n_particles:
            break

    return np.array(positions, dtype=float)


def attempt_particle_move(
    coords: np.ndarray,
    box_length: float,
    cfg: NPTConfig,
    beta: float,
    rng: np.random.Generator,
) -> bool:
    """Metropolis displacement move under fixed volume."""
    i = int(rng.integers(0, cfg.n_particles))
    old_pos = coords[i].copy()

    old_local = local_energy_of_particle(
        i, old_pos, coords, box_length, cfg.sigma, cfg.epsilon, cfg.cutoff
    )

    disp = rng.uniform(-cfg.move_step, cfg.move_step, size=cfg.dim)
    trial_pos = (old_pos + disp) % box_length

    new_local = local_energy_of_particle(
        i, trial_pos, coords, box_length, cfg.sigma, cfg.epsilon, cfg.cutoff
    )

    delta_u = new_local - old_local
    log_accept = -beta * delta_u

    if math.log(rng.random()) < min(0.0, log_accept):
        coords[i] = trial_pos
        return True
    return False


def attempt_volume_move(
    coords: np.ndarray,
    box_length: float,
    current_energy: float,
    cfg: NPTConfig,
    beta: float,
    rng: np.random.Generator,
) -> tuple[bool, float, float, np.ndarray]:
    """Metropolis isotropic volume move with coordinate scaling."""
    old_volume = box_length**cfg.dim
    trial_volume = old_volume + rng.uniform(-cfg.volume_step, cfg.volume_step)

    if trial_volume <= cfg.min_volume:
        return False, box_length, current_energy, coords

    scale = (trial_volume / old_volume) ** (1.0 / cfg.dim)
    new_box = box_length * scale
    trial_coords = (coords * scale) % new_box

    trial_energy = total_energy(trial_coords, new_box, cfg.sigma, cfg.epsilon, cfg.cutoff)
    delta_u = trial_energy - current_energy
    delta_v = trial_volume - old_volume

    # NPT acceptance in reduced units (k_B = 1):
    # log A = -beta * (ΔU + PΔV) + N * ln(V'/V)
    log_accept = (
        -beta * (delta_u + cfg.pressure * delta_v)
        + cfg.n_particles * math.log(trial_volume / old_volume)
    )

    if math.log(rng.random()) < min(0.0, log_accept):
        return True, new_box, trial_energy, trial_coords
    return False, box_length, current_energy, coords


def run_npt_simulation(cfg: NPTConfig) -> tuple[pd.DataFrame, RunSummary]:
    """Run NPT Monte Carlo and return time-series samples + summary."""
    rng = np.random.default_rng(cfg.seed)
    beta = 1.0 / cfg.temperature

    init_volume = cfg.n_particles / cfg.init_density
    box_length = init_volume ** (1.0 / cfg.dim)

    coords = init_lattice_positions(cfg.n_particles, cfg.dim, box_length)
    current_energy = total_energy(coords, box_length, cfg.sigma, cfg.epsilon, cfg.cutoff)

    particle_accept = 0
    particle_trials = 0
    volume_accept = 0
    volume_trials = 0

    records: list[dict[str, float]] = []
    total_sweeps = cfg.burn_in_sweeps + cfg.sample_sweeps

    for sweep in range(total_sweeps):
        for _ in range(cfg.n_particles):
            particle_trials += 1
            if attempt_particle_move(coords, box_length, cfg, beta, rng):
                particle_accept += 1

        volume_trials += 1
        accepted, box_length, current_energy, coords = attempt_volume_move(
            coords=coords,
            box_length=box_length,
            current_energy=current_energy,
            cfg=cfg,
            beta=beta,
            rng=rng,
        )
        if accepted:
            volume_accept += 1

        if sweep >= cfg.burn_in_sweeps and (sweep - cfg.burn_in_sweeps) % cfg.sample_interval == 0:
            volume = box_length**cfg.dim
            density = cfg.n_particles / volume
            enthalpy = current_energy + cfg.pressure * volume
            records.append(
                {
                    "sweep": float(sweep),
                    "volume": float(volume),
                    "density": float(density),
                    "energy_per_particle": float(current_energy / cfg.n_particles),
                    "enthalpy_per_particle": float(enthalpy / cfg.n_particles),
                }
            )

    df = pd.DataFrame(records)

    summary = RunSummary(
        pressure=cfg.pressure,
        mean_volume=float(df["volume"].mean()),
        mean_density=float(df["density"].mean()),
        mean_energy_per_particle=float(df["energy_per_particle"].mean()),
        mean_enthalpy_per_particle=float(df["enthalpy_per_particle"].mean()),
        particle_acceptance=particle_accept / particle_trials,
        volume_acceptance=volume_accept / volume_trials,
        n_samples=len(df),
    )
    return df, summary


def print_summary_table(summaries: list[RunSummary]) -> None:
    rows = [
        {
            "pressure": s.pressure,
            "mean_volume": s.mean_volume,
            "mean_density": s.mean_density,
            "mean_energy_per_particle": s.mean_energy_per_particle,
            "mean_enthalpy_per_particle": s.mean_enthalpy_per_particle,
            "particle_acceptance": s.particle_acceptance,
            "volume_acceptance": s.volume_acceptance,
            "n_samples": s.n_samples,
        }
        for s in summaries
    ]
    table = pd.DataFrame(rows).sort_values("pressure")
    with pd.option_context("display.precision", 4, "display.width", 140):
        print(table.to_string(index=False))


def main() -> None:
    print("NPT ensemble MVP: 2D Lennard-Jones Metropolis Monte Carlo")

    pressures = [0.5, 1.4]
    summaries: list[RunSummary] = []

    for idx, p in enumerate(pressures):
        cfg = NPTConfig(pressure=p, seed=20260407 + idx * 97)
        _, summary = run_npt_simulation(cfg)
        summaries.append(summary)

    print_summary_table(summaries)

    low_p = min(summaries, key=lambda x: x.pressure)
    high_p = max(summaries, key=lambda x: x.pressure)

    # Basic sanity checks to keep this MVP self-validating.
    assert low_p.n_samples > 20 and high_p.n_samples > 20
    assert 0.02 < low_p.particle_acceptance < 0.95
    assert 0.02 < high_p.particle_acceptance < 0.95
    assert 0.02 < low_p.volume_acceptance < 0.95
    assert 0.02 < high_p.volume_acceptance < 0.95
    assert high_p.mean_volume < low_p.mean_volume
    assert high_p.mean_density > low_p.mean_density

    print("Checks passed: higher pressure produced lower mean volume in NPT sampling.")


if __name__ == "__main__":
    main()

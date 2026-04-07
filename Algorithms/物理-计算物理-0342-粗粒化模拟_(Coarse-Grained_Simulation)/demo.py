"""Coarse-Grained Simulation MVP.

This script demonstrates a compact coarse-grained polymer workflow:
1) build a synthetic fine-grained chain,
2) map atoms to coarse beads,
3) run overdamped Langevin dynamics with effective interactions,
4) report observables and validation metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CGConfig:
    seed: int = 42
    dim: int = 3

    n_beads: int = 12
    atoms_per_bead: int = 4

    temperature: float = 1.0
    diffusion: float = 0.2
    dt: float = 5.0e-4
    n_steps: int = 6000

    bond_k: float = 200.0
    bond_r0: float = 1.0

    wca_epsilon: float = 0.5
    wca_sigma: float = 0.8

    sample_interval: int = 100

    bond_mean_tol: float = 0.08
    min_msd_growth: float = 0.01
    max_abs_position: float = 100.0


def build_fine_grained_chain(cfg: CGConfig, rng: np.random.Generator) -> np.ndarray:
    n_atoms = cfg.n_beads * cfg.atoms_per_bead
    atom_spacing = cfg.bond_r0 / cfg.atoms_per_bead

    coords = np.zeros((n_atoms, cfg.dim), dtype=float)
    x = np.arange(n_atoms, dtype=float) * atom_spacing
    coords[:, 0] = x

    thermal_jitter = 0.03
    coords += rng.normal(scale=thermal_jitter, size=coords.shape)

    coords -= coords.mean(axis=0, keepdims=True)
    return coords


def map_atoms_to_beads(atom_coords: np.ndarray, atoms_per_bead: int) -> np.ndarray:
    if atom_coords.shape[0] % atoms_per_bead != 0:
        raise ValueError("atom count must be divisible by atoms_per_bead")

    n_beads = atom_coords.shape[0] // atoms_per_bead
    reshaped = atom_coords.reshape(n_beads, atoms_per_bead, atom_coords.shape[1])
    return reshaped.mean(axis=1)


def bond_forces_and_energy(positions: np.ndarray, k_bond: float, r0: float) -> tuple[np.ndarray, float]:
    forces = np.zeros_like(positions)
    energy = 0.0

    for i in range(positions.shape[0] - 1):
        dr = positions[i + 1] - positions[i]
        dist = float(np.linalg.norm(dr))
        if dist < 1e-12:
            continue

        extension = dist - r0
        f_vec = k_bond * extension * (dr / dist)

        forces[i] += f_vec
        forces[i + 1] -= f_vec
        energy += 0.5 * k_bond * extension * extension

    return forces, float(energy)


def wca_forces_and_energy(positions: np.ndarray, epsilon: float, sigma: float) -> tuple[np.ndarray, float]:
    forces = np.zeros_like(positions)
    energy = 0.0

    r_cut = (2.0 ** (1.0 / 6.0)) * sigma
    sigma6 = sigma**6

    n = positions.shape[0]
    for i in range(n - 1):
        for j in range(i + 1, n):
            dr = positions[j] - positions[i]
            r = float(np.linalg.norm(dr))
            if r >= r_cut or r < 1e-12:
                continue

            inv_r = 1.0 / r
            sr6 = sigma6 * inv_r**6
            sr12 = sr6 * sr6

            u = 4.0 * epsilon * (sr12 - sr6) + epsilon
            energy += u

            coeff = 24.0 * epsilon * (2.0 * sr12 - sr6) * (inv_r**2)
            f_on_i = -coeff * dr
            forces[i] += f_on_i
            forces[j] -= f_on_i

    return forces, float(energy)


def total_forces_and_energy(positions: np.ndarray, cfg: CGConfig) -> tuple[np.ndarray, float]:
    f_bond, e_bond = bond_forces_and_energy(positions, cfg.bond_k, cfg.bond_r0)
    f_wca, e_wca = wca_forces_and_energy(positions, cfg.wca_epsilon, cfg.wca_sigma)
    return f_bond + f_wca, e_bond + e_wca


def observables(positions: np.ndarray, com0: np.ndarray) -> dict[str, float]:
    com = positions.mean(axis=0)
    centered = positions - com

    rg = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    ree = float(np.linalg.norm(positions[-1] - positions[0]))

    bonds = positions[1:] - positions[:-1]
    bond_lengths = np.linalg.norm(bonds, axis=1)
    mean_bond = float(np.mean(bond_lengths))

    msd_com = float(np.sum((com - com0) ** 2))

    return {
        "rg": rg,
        "ree": ree,
        "mean_bond": mean_bond,
        "msd_com": msd_com,
    }


def run_simulation(cfg: CGConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)

    atom_coords = build_fine_grained_chain(cfg, rng)
    positions = map_atoms_to_beads(atom_coords, cfg.atoms_per_bead)

    com0 = positions.mean(axis=0).copy()
    mobility = cfg.diffusion / cfg.temperature
    noise_scale = np.sqrt(2.0 * cfg.diffusion * cfg.dt)

    rows: list[dict[str, float]] = []

    for step in range(cfg.n_steps + 1):
        forces, potential = total_forces_and_energy(positions, cfg)

        obs = observables(positions, com0)
        if step % cfg.sample_interval == 0 or step == cfg.n_steps:
            rows.append(
                {
                    "step": float(step),
                    "time": step * cfg.dt,
                    "potential": potential,
                    "rg": obs["rg"],
                    "ree": obs["ree"],
                    "mean_bond": obs["mean_bond"],
                    "msd_com": obs["msd_com"],
                }
            )

        if step == cfg.n_steps:
            break

        random_kick = rng.normal(size=positions.shape)
        positions = positions + mobility * forces * cfg.dt + noise_scale * random_kick

    return pd.DataFrame(rows)


def validate(table: pd.DataFrame, cfg: CGConfig) -> tuple[bool, dict[str, float]]:
    finite = bool(np.isfinite(table.to_numpy(dtype=float)).all())

    mean_bond_avg = float(table["mean_bond"].mean())
    bond_error = abs(mean_bond_avg - cfg.bond_r0)

    msd_growth = float(table["msd_com"].iloc[-1] - table["msd_com"].iloc[0])
    max_abs_position_proxy = float(max(table["rg"].max(), table["ree"].max()))

    passed = (
        finite
        and bond_error <= cfg.bond_mean_tol
        and msd_growth >= cfg.min_msd_growth
        and max_abs_position_proxy <= cfg.max_abs_position
    )

    metrics = {
        "finite": float(finite),
        "mean_bond_avg": mean_bond_avg,
        "bond_error": bond_error,
        "msd_growth": msd_growth,
        "max_abs_position_proxy": max_abs_position_proxy,
    }
    return passed, metrics


def main() -> None:
    cfg = CGConfig()
    table = run_simulation(cfg)
    passed, metrics = validate(table, cfg)

    sample_idx = np.linspace(0, len(table) - 1, min(10, len(table)), dtype=int)
    sample = table.iloc[sample_idx]

    print("=== Coarse-Grained Simulation MVP ===")
    print(
        "beads={0}, atoms_per_bead={1}, steps={2}, dt={3}, T={4}, D={5}".format(
            cfg.n_beads,
            cfg.atoms_per_bead,
            cfg.n_steps,
            cfg.dt,
            cfg.temperature,
            cfg.diffusion,
        )
    )
    print(
        "bond_k={0}, bond_r0={1}, wca_epsilon={2}, wca_sigma={3}".format(
            cfg.bond_k,
            cfg.bond_r0,
            cfg.wca_epsilon,
            cfg.wca_sigma,
        )
    )
    print()

    print("Sampled trajectory statistics:")
    print(sample.to_string(index=False, float_format=lambda x: f"{x:.6e}"))
    print()

    print("Validation metrics:")
    print(f"finite                   = {bool(metrics['finite'])}")
    print(f"mean_bond_avg            = {metrics['mean_bond_avg']:.6e}")
    print(f"bond_error               = {metrics['bond_error']:.6e} (tol={cfg.bond_mean_tol:.2e})")
    print(f"msd_growth               = {metrics['msd_growth']:.6e} (min={cfg.min_msd_growth:.2e})")
    print(
        "max_abs_position_proxy   = "
        f"{metrics['max_abs_position_proxy']:.6e} (max={cfg.max_abs_position:.2e})"
    )
    print(f"Validation: {'PASS' if passed else 'FAIL'}")

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

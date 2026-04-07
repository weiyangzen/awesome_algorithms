"""Parallel Tempering MVP on a 2D Ising model.

This script is intentionally compact and transparent:
- Local Metropolis updates at each temperature.
- Adjacent replica exchanges (Replica Exchange Monte Carlo).
- A baseline single-chain run at the coldest temperature.
- Deterministic diagnostics and assertions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PTConfig:
    """Configuration for the MVP experiment."""

    lattice_size: int = 10
    temperatures: tuple[float, ...] = (1.5, 1.8, 2.1, 2.4, 2.7, 3.0)
    sweeps: int = 1200
    burn_in: int = 400
    seed_pt: int = 42
    seed_baseline: int = 43
    coupling: float = 1.0


def total_energy(spins: np.ndarray, coupling: float = 1.0) -> float:
    """Total energy for a periodic 2D Ising lattice.

    E = -J * sum_{<ij>} s_i s_j, counting each bond once.
    """

    return float(-coupling * np.sum(spins * (np.roll(spins, 1, axis=0) + np.roll(spins, 1, axis=1))))


def metropolis_sweep(
    spins: np.ndarray,
    beta: float,
    energy: float,
    rng: np.random.Generator,
    coupling: float = 1.0,
) -> tuple[float, float]:
    """One Metropolis sweep: L*L single-spin flip proposals."""

    l = spins.shape[0]
    accepted = 0

    for _ in range(l * l):
        i = int(rng.integers(0, l))
        j = int(rng.integers(0, l))

        s = spins[i, j]
        nn = (
            spins[(i - 1) % l, j]
            + spins[(i + 1) % l, j]
            + spins[i, (j - 1) % l]
            + spins[i, (j + 1) % l]
        )
        delta_e = 2.0 * coupling * s * nn

        if delta_e <= 0.0 or rng.random() < np.exp(-beta * delta_e):
            spins[i, j] = -s
            energy += delta_e
            accepted += 1

    return energy, accepted / (l * l)


def lag1_autocorr(series: np.ndarray) -> float:
    """Lag-1 autocorrelation for a 1D numeric series."""

    x = np.asarray(series, dtype=float)
    x = x - x.mean()
    denom = float(np.dot(x[:-1], x[:-1]))
    if denom < 1e-14:
        return 0.0
    return float(np.dot(x[:-1], x[1:]) / denom)


def run_parallel_tempering(cfg: PTConfig) -> dict[str, np.ndarray | float | int]:
    """Run replica exchange Monte Carlo across a temperature ladder."""

    rng = np.random.default_rng(cfg.seed_pt)
    betas = 1.0 / np.asarray(cfg.temperatures, dtype=float)
    n_rep = len(betas)

    l = cfg.lattice_size
    n_sites = l * l

    replicas = [rng.choice([-1, 1], size=(l, l)).astype(np.int8) for _ in range(n_rep)]
    energies = np.array([total_energy(rep, cfg.coupling) for rep in replicas], dtype=float)

    # replica_ids[position] tells which logical replica currently occupies this temperature slot
    replica_ids = list(range(n_rep))

    swap_attempts = np.zeros(n_rep - 1, dtype=int)
    swap_accepted = np.zeros(n_rep - 1, dtype=int)
    local_accept_acc = np.zeros(n_rep, dtype=float)

    cold_energy_trace: list[float] = []
    cold_abs_mag_trace: list[float] = []
    visit_counts = np.zeros((n_rep, n_rep), dtype=int)

    for sweep in range(cfg.sweeps):
        # 1) Local updates at each temperature.
        for k in range(n_rep):
            energies[k], acc = metropolis_sweep(
                replicas[k], float(betas[k]), float(energies[k]), rng, cfg.coupling
            )
            local_accept_acc[k] += acc

        # 2) Periodic consistency check against exact recomputation.
        if sweep % 100 == 0:
            for k in range(n_rep):
                recomputed = total_energy(replicas[k], cfg.coupling)
                if not np.isclose(recomputed, energies[k], atol=1e-9):
                    raise AssertionError(
                        f"Energy drift at replica {k}: tracked={energies[k]}, recomputed={recomputed}"
                    )
                energies[k] = recomputed

        # 3) Replica exchange on adjacent temperature pairs (odd-even scheme).
        start = sweep % 2
        for i in range(start, n_rep - 1, 2):
            j = i + 1
            swap_attempts[i] += 1

            # Detailed-balance acceptance ratio for swapping neighboring temperatures.
            log_accept_ratio = (betas[i] - betas[j]) * (energies[i] - energies[j])
            if np.log(rng.random()) < min(0.0, float(log_accept_ratio)):
                replicas[i], replicas[j] = replicas[j], replicas[i]
                energies[i], energies[j] = energies[j], energies[i]
                replica_ids[i], replica_ids[j] = replica_ids[j], replica_ids[i]
                swap_accepted[i] += 1

        # 4) Collect post-burn-in observables at the coldest slot.
        if sweep >= cfg.burn_in:
            cold_energy_trace.append(float(energies[0]) / n_sites)
            cold_abs_mag_trace.append(float(abs(replicas[0].mean())))
            for slot, rid in enumerate(replica_ids):
                visit_counts[rid, slot] += 1

    return {
        "cold_energy_trace": np.asarray(cold_energy_trace),
        "cold_abs_mag_trace": np.asarray(cold_abs_mag_trace),
        "swap_attempts": swap_attempts,
        "swap_accepted": swap_accepted,
        "swap_rates": swap_accepted / np.maximum(1, swap_attempts),
        "local_acceptance": local_accept_acc / cfg.sweeps,
        "id0_visited_temps": int(np.count_nonzero(visit_counts[0] > 0)),
    }


def run_baseline_single_chain(cfg: PTConfig) -> dict[str, np.ndarray | float]:
    """Single-chain Metropolis run at the coldest temperature."""

    rng = np.random.default_rng(cfg.seed_baseline)
    beta = 1.0 / float(cfg.temperatures[0])

    l = cfg.lattice_size
    n_sites = l * l

    spins = rng.choice([-1, 1], size=(l, l)).astype(np.int8)
    energy = total_energy(spins, cfg.coupling)

    cold_energy_trace: list[float] = []
    cold_abs_mag_trace: list[float] = []
    local_accept_acc = 0.0

    for sweep in range(cfg.sweeps):
        energy, acc = metropolis_sweep(spins, beta, energy, rng, cfg.coupling)
        local_accept_acc += acc

        if sweep % 100 == 0:
            recomputed = total_energy(spins, cfg.coupling)
            if not np.isclose(recomputed, energy, atol=1e-9):
                raise AssertionError(
                    f"Baseline energy drift: tracked={energy}, recomputed={recomputed}"
                )
            energy = recomputed

        if sweep >= cfg.burn_in:
            cold_energy_trace.append(float(energy) / n_sites)
            cold_abs_mag_trace.append(float(abs(spins.mean())))

    return {
        "cold_energy_trace": np.asarray(cold_energy_trace),
        "cold_abs_mag_trace": np.asarray(cold_abs_mag_trace),
        "local_acceptance": local_accept_acc / cfg.sweeps,
    }


def main() -> None:
    cfg = PTConfig()

    pt = run_parallel_tempering(cfg)
    baseline = run_baseline_single_chain(cfg)

    pt_lag1 = lag1_autocorr(pt["cold_abs_mag_trace"])
    base_lag1 = lag1_autocorr(baseline["cold_abs_mag_trace"])

    print("=== Parallel Tempering MVP (2D Ising) ===")
    print(
        f"Lattice: {cfg.lattice_size}x{cfg.lattice_size}, "
        f"sweeps={cfg.sweeps}, burn_in={cfg.burn_in}"
    )
    print(f"Temperatures: {cfg.temperatures}")

    print("-- PT diagnostics --")
    print(f"Swap rates (adjacent pairs): {np.array2string(pt['swap_rates'], precision=3)}")
    print(
        "Local acceptance by temperature: "
        f"{np.array2string(pt['local_acceptance'], precision=3)}"
    )
    print(f"Replica-0 visited slots: {pt['id0_visited_temps']}/{len(cfg.temperatures)}")
    print(f"Cold mean energy/spin (PT): {pt['cold_energy_trace'].mean():.4f}")
    print(f"Cold |m| lag-1 autocorr (PT): {pt_lag1:.4f}")

    print("-- Baseline diagnostics (single low-temperature chain) --")
    print(f"Baseline local acceptance: {baseline['local_acceptance']:.3f}")
    print(f"Cold mean energy/spin (baseline): {baseline['cold_energy_trace'].mean():.4f}")
    print(f"Cold |m| lag-1 autocorr (baseline): {base_lag1:.4f}")

    # Deterministic MVP checks (with fixed seeds/config).
    assert np.all(pt["swap_accepted"] > 0), "Some adjacent replicas never exchanged."
    assert pt["id0_visited_temps"] >= 2, "Replica diffusion across temperature ladder failed."
    assert pt_lag1 < base_lag1, "PT should decorrelate low-temperature observables in this setup."

    print("All checks passed.")


if __name__ == "__main__":
    main()

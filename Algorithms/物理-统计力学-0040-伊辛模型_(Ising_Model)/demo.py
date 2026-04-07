"""2D Ising model MVP using Metropolis Monte Carlo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class IsingStats:
    """Thermodynamic observables estimated from Monte Carlo samples."""

    temperature: float
    energy_per_spin: float
    abs_magnetization_per_spin: float
    heat_capacity_per_spin: float
    susceptibility_per_spin: float


def init_spins(lattice_size: int, rng: np.random.Generator) -> np.ndarray:
    """Initialize spins randomly from {-1, +1}."""
    return rng.choice(np.array([-1, 1], dtype=np.int8), size=(lattice_size, lattice_size))


def local_delta_energy(spins: np.ndarray, i: int, j: int) -> int:
    """Energy difference for flipping spin at (i, j), with periodic boundary."""
    l = spins.shape[0]
    s = int(spins[i, j])
    nb_sum = (
        int(spins[(i - 1) % l, j])
        + int(spins[(i + 1) % l, j])
        + int(spins[i, (j - 1) % l])
        + int(spins[i, (j + 1) % l])
    )
    return 2 * s * nb_sum


def metropolis_sweep(spins: np.ndarray, temperature: float, rng: np.random.Generator) -> None:
    """Perform one full Monte Carlo sweep: N = L^2 random single-spin proposals."""
    l = spins.shape[0]
    n_sites = l * l
    inv_t = 1.0 / temperature

    for _ in range(n_sites):
        i = int(rng.integers(0, l))
        j = int(rng.integers(0, l))
        d_e = local_delta_energy(spins, i, j)
        if d_e <= 0 or rng.random() < np.exp(-d_e * inv_t):
            spins[i, j] = -spins[i, j]


def total_energy(spins: np.ndarray) -> float:
    """Compute total energy H = -sum_{<ij>} s_i s_j for J=1, h=0."""
    # Count each bond once: right and down neighbors only.
    horizontal = spins * np.roll(spins, shift=-1, axis=1)
    vertical = spins * np.roll(spins, shift=-1, axis=0)
    return -float(np.sum(horizontal + vertical))


def run_ising_at_temperature(
    lattice_size: int,
    temperature: float,
    burn_in_sweeps: int,
    sample_sweeps: int,
    rng_seed: int,
) -> IsingStats:
    """Run Metropolis MCMC and estimate observables at one temperature."""
    rng = np.random.default_rng(rng_seed)
    spins = init_spins(lattice_size, rng)
    n_sites = lattice_size * lattice_size

    for _ in range(burn_in_sweeps):
        metropolis_sweep(spins, temperature, rng)

    energies: list[float] = []
    magnetizations: list[float] = []

    for _ in range(sample_sweeps):
        metropolis_sweep(spins, temperature, rng)
        energies.append(total_energy(spins))
        magnetizations.append(float(np.sum(spins)))

    e = np.asarray(energies, dtype=np.float64)
    m = np.asarray(magnetizations, dtype=np.float64)

    e_mean = float(np.mean(e))
    e2_mean = float(np.mean(e * e))
    m_mean = float(np.mean(m))
    m2_mean = float(np.mean(m * m))

    energy_per_spin = e_mean / n_sites
    abs_magnetization_per_spin = float(np.mean(np.abs(m))) / n_sites
    heat_capacity_per_spin = (e2_mean - e_mean * e_mean) / (n_sites * temperature * temperature)
    susceptibility_per_spin = (m2_mean - m_mean * m_mean) / (n_sites * temperature)

    return IsingStats(
        temperature=temperature,
        energy_per_spin=energy_per_spin,
        abs_magnetization_per_spin=abs_magnetization_per_spin,
        heat_capacity_per_spin=heat_capacity_per_spin,
        susceptibility_per_spin=susceptibility_per_spin,
    )


def main() -> None:
    lattice_size = 24
    burn_in_sweeps = 120
    sample_sweeps = 360
    temperatures = np.linspace(1.5, 3.5, 9)
    base_seed = 20260407

    stats: list[IsingStats] = []
    for idx, t in enumerate(temperatures):
        stats.append(
            run_ising_at_temperature(
                lattice_size=lattice_size,
                temperature=float(t),
                burn_in_sweeps=burn_in_sweeps,
                sample_sweeps=sample_sweeps,
                rng_seed=base_seed + idx,
            )
        )

    tc_exact = 2.0 / np.log(1.0 + np.sqrt(2.0))

    print("2D Ising Model (Metropolis Monte Carlo)")
    print(
        f"L={lattice_size}, burn_in={burn_in_sweeps}, samples={sample_sweeps}, "
        f"T_c(exact, infinite lattice)≈{tc_exact:.6f}"
    )
    print("-" * 78)
    print(f"{'T':>6} {'e=<H>/N':>12} {'|m|=<|M|>/N':>14} {'C/N':>12} {'chi/N':>12}")
    print("-" * 78)
    for s in stats:
        print(
            f"{s.temperature:6.2f} "
            f"{s.energy_per_spin:12.6f} "
            f"{s.abs_magnetization_per_spin:14.6f} "
            f"{s.heat_capacity_per_spin:12.6f} "
            f"{s.susceptibility_per_spin:12.6f}"
        )


if __name__ == "__main__":
    main()

"""Frustrated magnetism MVP.

This script compares antiferromagnetic Ising models on:
1) square lattice (unfrustrated), and
2) triangular lattice (geometrically frustrated).

A simple Metropolis simulated annealing is used to approach low-energy states.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IsingLattice:
    """Antiferromagnetic Ising model on a periodic 2D lattice."""

    name: str
    L: int
    J: float
    bonds_i: np.ndarray
    bonds_j: np.ndarray
    neighbors: np.ndarray
    degrees: np.ndarray

    @property
    def n_sites(self) -> int:
        return self.L * self.L

    @property
    def n_bonds(self) -> int:
        return int(self.bonds_i.size)

    def energy(self, spins: np.ndarray) -> float:
        """Hamiltonian H = J * sum_{<ij>} s_i s_j, J>0 for AF interaction."""
        return float(self.J * np.sum(spins[self.bonds_i] * spins[self.bonds_j]))

    def frustrated_ratio(self, spins: np.ndarray) -> float:
        """Fraction of bonds with s_i s_j = +1 (unsatisfied for AF coupling)."""
        return float(np.mean((spins[self.bonds_i] * spins[self.bonds_j]) > 0))


def _idx(x: int, y: int, L: int) -> int:
    return x * L + y


def _build_neighbor_table(n_sites: int, bonds: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    degrees = np.zeros(n_sites, dtype=np.int32)
    for i, j in bonds:
        degrees[i] += 1
        degrees[j] += 1

    max_deg = int(np.max(degrees))
    neighbors = -np.ones((n_sites, max_deg), dtype=np.int32)
    fill_ptr = np.zeros(n_sites, dtype=np.int32)

    for i, j in bonds:
        p = fill_ptr[i]
        neighbors[i, p] = j
        fill_ptr[i] += 1

        q = fill_ptr[j]
        neighbors[j, q] = i
        fill_ptr[j] += 1

    return neighbors, degrees


def build_square_af_ising(L: int, J: float = 1.0) -> IsingLattice:
    """Square lattice with periodic boundary and nearest-neighbor AF coupling."""
    bonds: list[tuple[int, int]] = []
    for x in range(L):
        for y in range(L):
            i = _idx(x, y, L)
            bonds.append((i, _idx((x + 1) % L, y, L)))
            bonds.append((i, _idx(x, (y + 1) % L, L)))

    neighbors, degrees = _build_neighbor_table(L * L, bonds)
    bond_arr = np.asarray(bonds, dtype=np.int32)
    return IsingLattice(
        name="Square (Unfrustrated AF)",
        L=L,
        J=J,
        bonds_i=bond_arr[:, 0],
        bonds_j=bond_arr[:, 1],
        neighbors=neighbors,
        degrees=degrees,
    )


def build_triangular_af_ising(L: int, J: float = 1.0) -> IsingLattice:
    """Triangular lattice with periodic boundary and nearest-neighbor AF coupling."""
    bonds: list[tuple[int, int]] = []
    for x in range(L):
        for y in range(L):
            i = _idx(x, y, L)
            bonds.append((i, _idx((x + 1) % L, y, L)))
            bonds.append((i, _idx(x, (y + 1) % L, L)))
            bonds.append((i, _idx((x + 1) % L, (y - 1) % L, L)))

    neighbors, degrees = _build_neighbor_table(L * L, bonds)
    bond_arr = np.asarray(bonds, dtype=np.int32)
    return IsingLattice(
        name="Triangular (Frustrated AF)",
        L=L,
        J=J,
        bonds_i=bond_arr[:, 0],
        bonds_j=bond_arr[:, 1],
        neighbors=neighbors,
        degrees=degrees,
    )


def metropolis_anneal(
    model: IsingLattice,
    seed: int,
    t_high: float = 4.0,
    t_low: float = 0.12,
    n_temps: int = 40,
    sweeps_per_temp: int = 14,
) -> dict[str, np.ndarray | float]:
    """Simulated annealing with Metropolis updates."""
    rng = np.random.default_rng(seed)
    spins = rng.choice(np.array([-1, 1], dtype=np.int8), size=model.n_sites)

    temps = np.geomspace(t_high, t_low, num=n_temps)
    energies = np.zeros(n_temps, dtype=np.float64)
    frustrated = np.zeros(n_temps, dtype=np.float64)

    for t_idx, temp in enumerate(temps):
        for _ in range(sweeps_per_temp):
            for _ in range(model.n_sites):
                k = int(rng.integers(0, model.n_sites))
                deg = int(model.degrees[k])
                nn = model.neighbors[k, :deg]
                local_sum = int(np.sum(spins[nn]))
                delta_e = -2.0 * model.J * float(spins[k]) * float(local_sum)

                if delta_e <= 0.0 or rng.random() < np.exp(-delta_e / temp):
                    spins[k] = np.int8(-spins[k])

        energies[t_idx] = model.energy(spins)
        frustrated[t_idx] = model.frustrated_ratio(spins)

    return {
        "spins": spins,
        "temps": temps,
        "energies": energies,
        "frustrated": frustrated,
        "final_energy": float(energies[-1]),
        "final_frustrated": float(frustrated[-1]),
    }


def summarize(model: IsingLattice, result: dict[str, np.ndarray | float]) -> dict[str, float]:
    e_total = float(result["final_energy"])
    f_ratio = float(result["final_frustrated"])
    e_per_site = e_total / model.n_sites
    e_per_bond = e_total / model.n_bonds

    return {
        "n_sites": float(model.n_sites),
        "n_bonds": float(model.n_bonds),
        "energy_total": e_total,
        "energy_per_site": e_per_site,
        "energy_per_bond": e_per_bond,
        "frustrated_ratio": f_ratio,
    }


def main() -> None:
    L = 18
    J = 1.0

    sq_model = build_square_af_ising(L=L, J=J)
    tri_model = build_triangular_af_ising(L=L, J=J)

    # Use independent, reproducible seeds from one parent RNG.
    seed_gen = np.random.default_rng(20260407)
    sq_seed = int(seed_gen.integers(1, 1_000_000_000))
    tri_seed = int(seed_gen.integers(1, 1_000_000_000))

    sq_res = metropolis_anneal(sq_model, seed=sq_seed)
    tri_res = metropolis_anneal(tri_model, seed=tri_seed)

    sq = summarize(sq_model, sq_res)
    tri = summarize(tri_model, tri_res)

    print("=== Frustrated Magnetism MVP: AF Ising + Metropolis Annealing ===")
    print(f"Lattice size: L={L}, sites={int(sq['n_sites'])}, coupling J={J}")
    print("")

    print(f"[{sq_model.name}]")
    print(f"  bonds                : {int(sq['n_bonds'])}")
    print(f"  final energy (total) : {sq['energy_total']:.4f}")
    print(f"  final energy / site  : {sq['energy_per_site']:.4f}")
    print(f"  final energy / bond  : {sq['energy_per_bond']:.4f}")
    print(f"  frustrated bond ratio: {sq['frustrated_ratio']:.4f}")
    print("")

    print(f"[{tri_model.name}]")
    print(f"  bonds                : {int(tri['n_bonds'])}")
    print(f"  final energy (total) : {tri['energy_total']:.4f}")
    print(f"  final energy / site  : {tri['energy_per_site']:.4f}")
    print(f"  final energy / bond  : {tri['energy_per_bond']:.4f}")
    print(f"  frustrated bond ratio: {tri['frustrated_ratio']:.4f}")
    print("")

    print("Reference lower bounds for AF Ising:")
    print("  square lattice      -> frustrated ratio theoretical minimum = 0")
    print("  triangular lattice  -> frustrated ratio theoretical minimum = 1/3")
    print("")

    gap = tri["frustrated_ratio"] - sq["frustrated_ratio"]
    print(f"Observed frustration gap (triangular - square): {gap:.4f}")


if __name__ == "__main__":
    main()

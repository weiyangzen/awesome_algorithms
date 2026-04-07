"""Minimal runnable MVP for Kagome antiferromagnet simulation.

Model: classical nearest-neighbor Heisenberg antiferromagnet on a periodic
Kagome lattice. The script uses simulated annealing + Metropolis updates and
prints diagnostics that expose geometric frustration behavior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


FloatArray = np.ndarray
IntArray = np.ndarray


@dataclass
class KagomeConfig:
    """Control parameters for the Kagome magnet MVP."""

    Lx: int = 4
    Ly: int = 4
    J: float = 1.0
    sweeps: int = 180
    burn_in: int = 70
    beta_start: float = 0.2
    beta_end: float = 6.5
    proposal_sigma: float = 0.35
    seed: int = 7
    neighbor_tol: float = 1e-4
    sf_grid: int = 4


def normalize_rows(vectors: FloatArray) -> FloatArray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return vectors / norms


def random_unit_vectors(n: int, rng: np.random.Generator) -> FloatArray:
    raw = rng.normal(size=(n, 3))
    return normalize_rows(raw)


def build_kagome_coordinates(Lx: int, Ly: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Build 2D coordinates for periodic Kagome supercell.

    Primitive vectors:
      a1 = (1, 0), a2 = (1/2, sqrt(3)/2)
    Basis:
      b0 = (0, 0), b1 = (1/2, 0), b2 = (1/4, sqrt(3)/4)
    """

    a1 = np.array([1.0, 0.0])
    a2 = np.array([0.5, math.sqrt(3.0) / 2.0])
    basis = np.array(
        [
            [0.0, 0.0],
            [0.5, 0.0],
            [0.25, math.sqrt(3.0) / 4.0],
        ],
        dtype=float,
    )

    coords: list[FloatArray] = []
    for ix in range(Lx):
        for iy in range(Ly):
            origin = ix * a1 + iy * a2
            for b in basis:
                coords.append(origin + b)

    super_a = Lx * a1
    super_b = Ly * a2
    return np.array(coords, dtype=float), super_a, super_b


def _min_periodic_distance(base_delta: FloatArray, super_a: FloatArray, super_b: FloatArray) -> float:
    best = float("inf")
    for pa in (-1, 0, 1):
        for pb in (-1, 0, 1):
            delta = base_delta + pa * super_a + pb * super_b
            dist = float(np.linalg.norm(delta))
            if dist < best:
                best = dist
    return best


def build_kagome_bonds(
    coords: FloatArray,
    super_a: FloatArray,
    super_b: FloatArray,
    d_nn: float = 0.5,
    tol: float = 1e-4,
) -> IntArray:
    """Infer nearest-neighbor bonds from periodic geometry."""

    n = coords.shape[0]
    bonds: list[tuple[int, int]] = []

    for i in range(n - 1):
        ri = coords[i]
        for j in range(i + 1, n):
            base_delta = coords[j] - ri
            dmin = _min_periodic_distance(base_delta, super_a, super_b)
            if abs(dmin - d_nn) < tol:
                bonds.append((i, j))

    if not bonds:
        raise RuntimeError("No bonds were detected; check geometry or tolerance.")

    return np.array(bonds, dtype=np.int64)


def build_neighbor_list(num_sites: int, bonds: IntArray) -> list[np.ndarray]:
    neighbors: list[list[int]] = [[] for _ in range(num_sites)]

    for i, j in bonds:
        neighbors[int(i)].append(int(j))
        neighbors[int(j)].append(int(i))

    return [np.array(sorted(items), dtype=np.int64) for items in neighbors]


def find_triangles(neighbors: list[np.ndarray]) -> IntArray:
    """Enumerate 3-cliques (elementary triangles for Kagome nearest-neighbor graph)."""

    n = len(neighbors)
    neighbor_sets = [set(arr.tolist()) for arr in neighbors]
    triangles: list[tuple[int, int, int]] = []

    for i in range(n):
        for j in neighbors[i]:
            jj = int(j)
            if jj <= i:
                continue
            common = neighbor_sets[i].intersection(neighbor_sets[jj])
            for k in common:
                if k > jj:
                    triangles.append((i, jj, int(k)))

    return np.array(triangles, dtype=np.int64)


def total_energy(spins: FloatArray, bonds: IntArray, J: float) -> float:
    pair_dot = np.einsum("ij,ij->i", spins[bonds[:, 0]], spins[bonds[:, 1]])
    return float(J * np.sum(pair_dot))


def metropolis_sweep(
    spins: FloatArray,
    neighbors: list[np.ndarray],
    beta: float,
    J: float,
    sigma: float,
    rng: np.random.Generator,
) -> float:
    """One Monte Carlo sweep: N attempted single-spin updates."""

    n = spins.shape[0]
    accepted = 0

    for _ in range(n):
        i = int(rng.integers(0, n))
        old_spin = spins[i].copy()

        trial_spin = old_spin + sigma * rng.normal(size=3)
        norm = float(np.linalg.norm(trial_spin))
        if norm < 1e-12:
            continue
        trial_spin /= norm

        local_sum = np.sum(spins[neighbors[i]], axis=0)
        delta_e = float(J * np.dot(trial_spin - old_spin, local_sum))

        if (delta_e <= 0.0) or (rng.random() < math.exp(-beta * delta_e)):
            spins[i] = trial_spin
            accepted += 1

    return accepted / n


def mean_neighbor_dot(spins: FloatArray, bonds: IntArray) -> float:
    dots = np.einsum("ij,ij->i", spins[bonds[:, 0]], spins[bonds[:, 1]])
    return float(np.mean(dots))


def mean_abs_scalar_chirality(spins: FloatArray, triangles: IntArray) -> float:
    if triangles.size == 0:
        return float("nan")

    s0 = spins[triangles[:, 0]]
    s1 = spins[triangles[:, 1]]
    s2 = spins[triangles[:, 2]]
    chi = np.einsum("ij,ij->i", s0, np.cross(s1, s2))
    return float(np.mean(np.abs(chi)))


def structure_factor_peak(
    spins: FloatArray,
    coords: FloatArray,
    Lx: int,
    Ly: int,
    hmax: int,
) -> tuple[float, tuple[int, int]]:
    """Compute coarse static structure-factor peak over integer (h,k) grid.

    q(h,k) = h * b1 / Lx + k * b2 / Ly, excluding (0,0).
    """

    a1 = np.array([1.0, 0.0])
    a2 = np.array([0.5, math.sqrt(3.0) / 2.0])
    a_mat = np.column_stack((a1, a2))
    b_mat = 2.0 * math.pi * np.linalg.inv(a_mat).T
    b1 = b_mat[:, 0]
    b2 = b_mat[:, 1]

    n = spins.shape[0]
    best_s = -float("inf")
    best_hk = (0, 0)

    for h in range(-hmax, hmax + 1):
        for k in range(-hmax, hmax + 1):
            if h == 0 and k == 0:
                continue
            q_vec = (h / Lx) * b1 + (k / Ly) * b2
            phase = np.exp(1j * (coords @ q_vec))
            amp = np.sum(spins * phase[:, None], axis=0)
            sq = float(np.real(np.vdot(amp, amp)) / n)
            if sq > best_s:
                best_s = sq
                best_hk = (h, k)

    return best_s, best_hk


def run_simulation(config: KagomeConfig) -> tuple[pd.DataFrame, dict[str, float | int | tuple[int, int]]]:
    coords, super_a, super_b = build_kagome_coordinates(config.Lx, config.Ly)
    bonds = build_kagome_bonds(
        coords=coords,
        super_a=super_a,
        super_b=super_b,
        d_nn=0.5,
        tol=config.neighbor_tol,
    )

    n = coords.shape[0]
    neighbors = build_neighbor_list(num_sites=n, bonds=bonds)
    triangles = find_triangles(neighbors)

    degrees = np.array([len(nn) for nn in neighbors], dtype=int)
    if not np.all(degrees == 4):
        raise RuntimeError(f"Kagome coordination check failed: degrees={np.unique(degrees)}")

    rng = np.random.default_rng(config.seed)
    spins = random_unit_vectors(n=n, rng=rng)

    betas = np.linspace(config.beta_start, config.beta_end, config.sweeps)
    records: list[dict[str, float | int]] = []

    for sweep_id, beta in enumerate(betas, start=1):
        accept_ratio = metropolis_sweep(
            spins=spins,
            neighbors=neighbors,
            beta=float(beta),
            J=config.J,
            sigma=config.proposal_sigma,
            rng=rng,
        )

        e_per_spin = total_energy(spins, bonds, config.J) / n
        mag = float(np.linalg.norm(np.mean(spins, axis=0)))
        nn_dot = mean_neighbor_dot(spins, bonds)
        abs_chi = mean_abs_scalar_chirality(spins, triangles)

        records.append(
            {
                "sweep": sweep_id,
                "beta": float(beta),
                "accept_ratio": accept_ratio,
                "energy_per_spin": e_per_spin,
                "magnetization": mag,
                "nn_dot": nn_dot,
                "abs_chirality": abs_chi,
            }
        )

    history = pd.DataFrame(records)

    window_start = max(config.burn_in, config.sweeps // 2)
    measured = history.iloc[window_start:].copy()

    sf_peak, sf_hk = structure_factor_peak(
        spins=spins,
        coords=coords,
        Lx=config.Lx,
        Ly=config.Ly,
        hmax=config.sf_grid,
    )

    summary: dict[str, float | int | tuple[int, int]] = {
        "num_sites": n,
        "num_bonds": int(bonds.shape[0]),
        "num_triangles": int(triangles.shape[0]),
        "avg_degree": float(np.mean(degrees)),
        "measured_energy_per_spin": float(measured["energy_per_spin"].mean()),
        "measured_magnetization": float(measured["magnetization"].mean()),
        "measured_nn_dot": float(measured["nn_dot"].mean()),
        "measured_abs_chirality": float(measured["abs_chirality"].mean()),
        "final_accept_ratio": float(history["accept_ratio"].iloc[-1]),
        "structure_factor_peak": float(sf_peak),
        "structure_factor_hk": sf_hk,
    }

    return history, summary


def main() -> None:
    config = KagomeConfig()
    history, summary = run_simulation(config)

    # Basic finite-value checks for MVP validation.
    required_scalars = [
        "measured_energy_per_spin",
        "measured_magnetization",
        "measured_nn_dot",
        "measured_abs_chirality",
        "final_accept_ratio",
        "structure_factor_peak",
    ]
    for key in required_scalars:
        if not np.isfinite(summary[key]):
            raise RuntimeError(f"Non-finite diagnostic detected: {key}={summary[key]}")

    print("Kagome Magnet MVP (classical AFM Heisenberg, simulated annealing)")
    print(
        f"Lx={config.Lx}, Ly={config.Ly}, N={summary['num_sites']}, "
        f"bonds={summary['num_bonds']}, triangles={summary['num_triangles']}"
    )
    print(f"avg_degree={summary['avg_degree']:.2f}, J={config.J:.2f}, seed={config.seed}")
    print("-")
    print(
        "Measured-window averages: "
        f"E/N={summary['measured_energy_per_spin']:.6f}, "
        f"|M|={summary['measured_magnetization']:.6f}, "
        f"<Si.Sj>={summary['measured_nn_dot']:.6f}, "
        f"<|chi|>={summary['measured_abs_chirality']:.6f}"
    )
    print(
        f"Final accept_ratio={summary['final_accept_ratio']:.4f}, "
        f"S(q)_peak={summary['structure_factor_peak']:.6f} at h,k={summary['structure_factor_hk']}"
    )

    print("\nLast 8 sweeps:")
    print(
        history.tail(8).to_string(
            index=False,
            float_format=lambda x: f"{x: .6f}",
        )
    )


if __name__ == "__main__":
    main()

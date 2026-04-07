"""Minimal runnable MVP for the pseudopotential method.

This demo solves a 1D periodic problem in a plane-wave basis:
    H_{G,G'}(k) = 0.5 * (k + G)^2 * delta_{G,G'} + V_{G-G'}
with a smooth local pseudopotential form factor
    V_q = -V0 * exp(-0.5 * (q * sigma)^2).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a 1D pseudopotential band-structure MVP."""

    lattice_constant: float = 1.0
    g_cut: int = 5
    n_kpoints: int = 121
    n_bands: int = 4
    v0: float = 0.35
    sigma: float = 0.35


def reciprocal_vectors(g_cut: int, lattice_constant: float) -> np.ndarray:
    """Return reciprocal lattice vectors G = 2*pi*n/a for n in [-g_cut, g_cut]."""
    n = np.arange(-g_cut, g_cut + 1, dtype=float)
    return (2.0 * np.pi / lattice_constant) * n


def form_factor(delta_g: np.ndarray, v0: float, sigma: float) -> np.ndarray:
    """Smooth local pseudopotential form factor V_q."""
    return -v0 * np.exp(-0.5 * (delta_g * sigma) ** 2)


def build_hamiltonian(
    k: float,
    gvecs: np.ndarray,
    v0: float,
    sigma: float,
    zero_average_term: bool = True,
) -> np.ndarray:
    """Build H(k) in plane-wave basis."""
    kinetic = 0.5 * (k + gvecs) ** 2
    delta = gvecs[:, None] - gvecs[None, :]
    potential = form_factor(delta, v0=v0, sigma=sigma)

    if zero_average_term:
        # Removing q=0 only shifts all eigenvalues and does not affect gaps.
        np.fill_diagonal(potential, 0.0)

    h = potential
    h[np.diag_indices_from(h)] += kinetic
    return 0.5 * (h + h.T)


def solve_bands(config: ModelConfig) -> tuple[np.ndarray, np.ndarray]:
    """Solve the lowest bands over the first Brillouin zone."""
    a = config.lattice_constant
    kpoints = np.linspace(-np.pi / a, np.pi / a, config.n_kpoints)
    gvecs = reciprocal_vectors(config.g_cut, a)

    bands = np.empty((config.n_kpoints, config.n_bands), dtype=float)
    for i, k in enumerate(kpoints):
        h = build_hamiltonian(k, gvecs, v0=config.v0, sigma=config.sigma)
        evals = eigh(h, eigvals_only=True, check_finite=False)
        bands[i, :] = evals[: config.n_bands]

    return kpoints, bands


def summarize_gap(kpoints: np.ndarray, bands: np.ndarray, lattice_constant: float) -> float:
    """Estimate first direct gap near +pi/a."""
    bz_edge = np.pi / lattice_constant
    edge_idx = int(np.argmin(np.abs(kpoints - bz_edge)))
    return float(bands[edge_idx, 1] - bands[edge_idx, 0])


def bands_to_dataframe(kpoints: np.ndarray, bands: np.ndarray, lattice_constant: float) -> pd.DataFrame:
    """Assemble tabular output for inspection and downstream plotting."""
    reduced_k = kpoints * lattice_constant / np.pi
    data: dict[str, np.ndarray] = {
        "k": kpoints,
        "k_over_pi": reduced_k,
    }
    for j in range(bands.shape[1]):
        data[f"band_{j + 1}"] = bands[:, j]
    return pd.DataFrame(data)


def main() -> None:
    config = ModelConfig()

    kpoints, bands = solve_bands(config)
    if np.any(np.diff(bands, axis=1) < -1e-10):
        raise RuntimeError("Eigenvalues are not sorted. Numerical issue detected.")

    gap_with_pseudo = summarize_gap(kpoints, bands, config.lattice_constant)

    free_config = ModelConfig(
        lattice_constant=config.lattice_constant,
        g_cut=config.g_cut,
        n_kpoints=config.n_kpoints,
        n_bands=config.n_bands,
        v0=0.0,
        sigma=config.sigma,
    )
    free_kpoints, free_bands = solve_bands(free_config)
    gap_free = summarize_gap(free_kpoints, free_bands, free_config.lattice_constant)

    gamma_idx = int(np.argmin(np.abs(kpoints)))
    gamma_energies = bands[gamma_idx, :]

    df = bands_to_dataframe(kpoints, bands, config.lattice_constant)
    output_csv = Path(__file__).with_name("bands.csv")
    df.to_csv(output_csv, index=False)

    print("Pseudopotential Method MVP (1D plane-wave)")
    print(f"basis size = {2 * config.g_cut + 1}, k-points = {config.n_kpoints}, bands = {config.n_bands}")
    print(f"Gamma-point energies (first {config.n_bands}): {np.array2string(gamma_energies, precision=6)}")
    print(f"Gap at BZ edge with pseudopotential (V0={config.v0:.3f}): {gap_with_pseudo:.6f}")
    print(f"Gap at BZ edge in free-electron limit (V0=0): {gap_free:.6f}")
    print(f"Saved band table to: {output_csv}")
    print("First five rows:")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()

"""Pauli exclusion principle MVP.

This demo builds a fermionic many-body state with a Slater determinant and
verifies two key consequences of Pauli exclusion:
1) exchanging two identical fermions flips the wavefunction sign;
2) forcing two fermions into the same one-particle state collapses amplitude.

It also computes finite-temperature Fermi-Dirac occupations by solving for the
chemical potential under a fixed particle number, and checks occupancy <= 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy import linalg, optimize, special
from sklearn.metrics import mean_absolute_error, r2_score


@dataclass(frozen=True)
class PauliConfig:
    """Configuration for the minimal Pauli exclusion verification."""

    n_particles: int = 4
    n_levels: int = 12
    well_length: float = 1.0
    temperature_low: float = 0.05
    temperature_high: float = 0.8


def level_energies(n_levels: int) -> np.ndarray:
    """Infinite square well energies in scaled units: E_n = n^2, n=1..M."""
    n = np.arange(1, n_levels + 1, dtype=float)
    return n**2


def orbital_value(n_quantum: int, x: np.ndarray, length: float) -> np.ndarray:
    """Return infinite-square-well orbital phi_n(x)=sqrt(2/L) sin(n pi x / L)."""
    return np.sqrt(2.0 / length) * np.sin(n_quantum * np.pi * x / length)


def slater_matrix(
    coordinates: np.ndarray,
    orbital_indices: np.ndarray,
    length: float,
) -> np.ndarray:
    """Construct matrix A_{ij} = phi_{n_i}(x_j) for Slater determinant."""
    if coordinates.size != orbital_indices.size:
        raise ValueError("coordinates and orbital_indices must have same length.")

    n_particles = coordinates.size
    matrix = np.empty((n_particles, n_particles), dtype=float)
    for i, n_quantum in enumerate(orbital_indices):
        matrix[i, :] = orbital_value(int(n_quantum), coordinates, length)
    return matrix


def slater_amplitude(
    coordinates: np.ndarray,
    orbital_indices: np.ndarray,
    length: float,
) -> float:
    """Return normalized fermionic amplitude det(A)/sqrt(N!)."""
    matrix = slater_matrix(coordinates, orbital_indices, length)
    normalization = np.sqrt(float(special.factorial(coordinates.size, exact=False)))
    return float(linalg.det(matrix) / normalization)


def ideal_zero_temperature_occupancy(n_levels: int, n_particles: int) -> np.ndarray:
    """Ground-state filling for spinless fermions: first N levels occupied."""
    if not (0 < n_particles <= n_levels):
        raise ValueError("Require 0 < n_particles <= n_levels.")
    occ = np.zeros(n_levels, dtype=float)
    occ[:n_particles] = 1.0
    return occ


def fermi_dirac_given_mu(energies: np.ndarray, mu: float, temperature: float) -> np.ndarray:
    """Evaluate occupations f(E)=1/(exp((E-mu)/T)+1) with numerically safe clip."""
    x = np.clip((energies - mu) / temperature, -80.0, 80.0)
    return 1.0 / (np.exp(x) + 1.0)


def solve_chemical_potential(energies: np.ndarray, n_particles: int, temperature: float) -> float:
    """Solve sum_i f(E_i,mu,T)=N for mu by bracketed root finding."""
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    if not (0 < n_particles < energies.size):
        raise ValueError("Require 0 < n_particles < number of levels.")

    def residual(mu: float) -> float:
        return float(fermi_dirac_given_mu(energies, mu, temperature).sum() - n_particles)

    lo = float(energies.min() - 10.0 - 40.0 * temperature)
    hi = float(energies.max() + 10.0 + 40.0 * temperature)

    expand_step = 10.0 + 40.0 * temperature
    for _ in range(100):
        if residual(lo) <= 0.0:
            break
        lo -= expand_step
    else:
        raise RuntimeError("Failed to find lower bracket for chemical potential.")

    for _ in range(100):
        if residual(hi) >= 0.0:
            break
        hi += expand_step
    else:
        raise RuntimeError("Failed to find upper bracket for chemical potential.")

    return float(optimize.brentq(residual, lo, hi, xtol=1e-12, rtol=1e-10, maxiter=300))


def finite_temperature_occupancy(
    energies: np.ndarray,
    n_particles: int,
    temperature: float,
) -> tuple[float, np.ndarray]:
    """Return chemical potential and occupations at given finite temperature."""
    mu = solve_chemical_potential(energies, n_particles=n_particles, temperature=temperature)
    occ = fermi_dirac_given_mu(energies, mu=mu, temperature=temperature)
    return mu, occ


def run_pauli_demo(config: PauliConfig) -> dict[str, object]:
    """Run all checks and return tabular + scalar diagnostics."""
    energies = level_energies(config.n_levels)
    orbital_indices = np.arange(1, config.n_particles + 1)
    coordinates = np.linspace(0.12, 0.88, config.n_particles)

    psi_ref = slater_amplitude(coordinates, orbital_indices, config.well_length)

    swapped = coordinates.copy()
    swapped[[0, 1]] = swapped[[1, 0]]
    psi_swapped = slater_amplitude(swapped, orbital_indices, config.well_length)
    antisymmetry_error = abs(psi_ref + psi_swapped)

    equal_coordinates = coordinates.copy()
    equal_coordinates[1] = equal_coordinates[0]
    equal_position_amplitude = abs(
        slater_amplitude(equal_coordinates, orbital_indices, config.well_length)
    )

    duplicate_orbitals = orbital_indices.copy()
    duplicate_orbitals[1] = duplicate_orbitals[0]
    duplicate_orbital_amplitude = abs(
        slater_amplitude(coordinates, duplicate_orbitals, config.well_length)
    )

    ideal_occ = ideal_zero_temperature_occupancy(config.n_levels, config.n_particles)
    mu_low, occ_low = finite_temperature_occupancy(
        energies,
        n_particles=config.n_particles,
        temperature=config.temperature_low,
    )
    mu_high, occ_high = finite_temperature_occupancy(
        energies,
        n_particles=config.n_particles,
        temperature=config.temperature_high,
    )

    mae_low = float(mean_absolute_error(ideal_occ, occ_low))
    r2_low = float(r2_score(ideal_occ, occ_low))

    occ_high_torch = torch.tensor(occ_high, dtype=torch.float64)
    torch_cap_violation = float(torch.relu(occ_high_torch - 1.0).max().item())
    torch_particle_error = float(abs(occ_high_torch.sum().item() - config.n_particles))

    df = pd.DataFrame(
        {
            "level_n": np.arange(1, config.n_levels + 1),
            "energy": energies,
            "occ_ideal_T0": ideal_occ,
            "occ_lowT": occ_low,
            "occ_highT": occ_high,
        }
    )
    df["pauli_cap_margin_highT"] = 1.0 - df["occ_highT"]

    return {
        "table": df,
        "psi_ref": psi_ref,
        "psi_swapped": psi_swapped,
        "antisymmetry_error": antisymmetry_error,
        "equal_position_amplitude": equal_position_amplitude,
        "duplicate_orbital_amplitude": duplicate_orbital_amplitude,
        "mu_low": mu_low,
        "mu_high": mu_high,
        "mae_low": mae_low,
        "r2_low": r2_low,
        "torch_cap_violation": torch_cap_violation,
        "torch_particle_error": torch_particle_error,
        "sum_occ_low": float(occ_low.sum()),
        "sum_occ_high": float(occ_high.sum()),
    }


def main() -> None:
    config = PauliConfig()
    result = run_pauli_demo(config)

    table = result["table"]
    assert isinstance(table, pd.DataFrame)

    print("Pauli Exclusion Principle Demo (spinless fermions)")
    print(
        f"N={config.n_particles}, levels={config.n_levels}, "
        f"T_low={config.temperature_low}, T_high={config.temperature_high}"
    )
    print("-" * 72)
    print(f"psi_ref                     = {result['psi_ref']:.8e}")
    print(f"psi_swapped                 = {result['psi_swapped']:.8e}")
    print(f"antisymmetry_error          = {result['antisymmetry_error']:.3e}")
    print(f"equal_position_amplitude    = {result['equal_position_amplitude']:.3e}")
    print(f"duplicate_orbital_amplitude = {result['duplicate_orbital_amplitude']:.3e}")
    print(f"mu_low                      = {result['mu_low']:.6f}")
    print(f"mu_high                     = {result['mu_high']:.6f}")
    print(f"lowT_MAE_vs_T0step          = {result['mae_low']:.6e}")
    print(f"lowT_R2_vs_T0step           = {result['r2_low']:.6f}")
    print(f"torch_cap_violation         = {result['torch_cap_violation']:.3e}")
    print(f"torch_particle_error        = {result['torch_particle_error']:.3e}")
    print(f"sum_occ_low                 = {result['sum_occ_low']:.10f}")
    print(f"sum_occ_high                = {result['sum_occ_high']:.10f}")
    print("-" * 72)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # Gate checks for non-interactive validation.
    assert abs(result["psi_ref"]) > 1e-10, "Reference amplitude unexpectedly collapsed."
    assert result["antisymmetry_error"] < 1e-10, "Exchange antisymmetry check failed."
    assert result["equal_position_amplitude"] < 1e-10, "Coincident coordinates should null ψ."
    assert (
        result["duplicate_orbital_amplitude"] < 1e-10
    ), "Duplicate one-particle state should null ψ."
    assert result["torch_cap_violation"] < 1e-12, "Occupancy exceeded Pauli cap of 1."
    assert result["torch_particle_error"] < 1e-9, "Particle-number conservation check failed."
    assert result["mae_low"] < 0.08, "Low-T occupancy should be close to T=0 step filling."
    assert result["r2_low"] > 0.9, "Low-T occupancy should align with T=0 pattern."


if __name__ == "__main__":
    main()

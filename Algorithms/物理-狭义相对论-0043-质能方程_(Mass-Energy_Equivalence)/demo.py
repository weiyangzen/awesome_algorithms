"""Minimal runnable MVP for Mass-Energy Equivalence.

This script demonstrates:
1) Rest energy: E0 = m c^2
2) Inverse conversion: m = E0 / c^2
3) Relativistic total/kinetic energy at a given beta.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# SI exact speed of light (m/s), defined by the SI.
C_LIGHT = 299_792_458.0
JOULE_PER_KWH = 3.6e6
JOULE_PER_TNT_TON = 4.184e9


def validate_mass_array(mass_kg: np.ndarray) -> np.ndarray:
    """Validate and return a 1D finite non-negative mass array."""
    mass_kg = np.asarray(mass_kg, dtype=np.float64)
    if mass_kg.ndim != 1:
        raise ValueError("mass_kg must be a 1D array.")
    if not np.all(np.isfinite(mass_kg)):
        raise ValueError("mass_kg contains non-finite values.")
    if np.any(mass_kg < 0.0):
        raise ValueError("mass_kg must be non-negative.")
    return mass_kg


def lorentz_gamma(beta: float) -> float:
    """Compute gamma = 1/sqrt(1-beta^2) for |beta| < 1."""
    if not np.isfinite(beta):
        raise ValueError("beta must be finite.")
    if abs(beta) >= 1.0:
        raise ValueError("|beta| must be < 1.")
    return float(1.0 / np.sqrt(1.0 - beta * beta))


def rest_energy_joule(mass_kg: np.ndarray) -> np.ndarray:
    """Compute rest energy E0 = m c^2 (J)."""
    mass_kg = validate_mass_array(mass_kg)
    return mass_kg * (C_LIGHT**2)


def mass_from_energy_joule(energy_joule: np.ndarray) -> np.ndarray:
    """Compute equivalent mass m = E/c^2 (kg)."""
    energy_joule = np.asarray(energy_joule, dtype=np.float64)
    if energy_joule.ndim != 1:
        raise ValueError("energy_joule must be a 1D array.")
    if not np.all(np.isfinite(energy_joule)):
        raise ValueError("energy_joule contains non-finite values.")
    if np.any(energy_joule < 0.0):
        raise ValueError("energy_joule must be non-negative.")
    return energy_joule / (C_LIGHT**2)


def total_energy_joule(mass_kg: np.ndarray, beta: float) -> np.ndarray:
    """Compute relativistic total energy E = gamma m c^2 (J)."""
    gamma = lorentz_gamma(beta)
    return gamma * rest_energy_joule(mass_kg)


def kinetic_energy_joule(mass_kg: np.ndarray, beta: float) -> np.ndarray:
    """Compute kinetic energy K = (gamma-1) m c^2 (J)."""
    return total_energy_joule(mass_kg, beta) - rest_energy_joule(mass_kg)


def build_summary_table(mass_kg: np.ndarray, beta: float) -> pd.DataFrame:
    """Build a compact summary table for mass-energy conversions."""
    e0 = rest_energy_joule(mass_kg)
    e_total = total_energy_joule(mass_kg, beta)
    e_kin = e_total - e0

    return pd.DataFrame(
        {
            "mass_kg": mass_kg,
            "rest_energy_j": e0,
            "rest_energy_kwh": e0 / JOULE_PER_KWH,
            "rest_energy_tnt_ton": e0 / JOULE_PER_TNT_TON,
            f"total_energy_j_at_beta_{beta:.2f}": e_total,
            f"kinetic_energy_j_at_beta_{beta:.2f}": e_kin,
        }
    )


def main() -> None:
    masses_kg = np.array([1e-6, 1e-3, 1.0, 70.0], dtype=np.float64)
    beta = 0.8

    summary = build_summary_table(masses_kg, beta)

    # Non-interactive validation checks.
    e0 = rest_energy_joule(masses_kg)
    masses_back = mass_from_energy_joule(e0)
    e0_beta0 = total_energy_joule(masses_kg, beta=0.0)
    ek = kinetic_energy_joule(masses_kg, beta)

    assert np.allclose(masses_back, masses_kg, rtol=1e-12, atol=0.0), "m <-> E conversion failed."
    assert np.allclose(rest_energy_joule(2.0 * masses_kg), 2.0 * e0, rtol=1e-12, atol=0.0), "Linearity failed."
    assert np.allclose(e0_beta0, e0, rtol=1e-12, atol=0.0), "E_total should equal E0 at beta=0."
    assert np.all(ek >= 0.0), "Kinetic energy must be non-negative."

    one_kg_rest = float(rest_energy_joule(np.array([1.0]))[0])

    print("Mass-Energy Equivalence MVP (SI units)")
    print(f"c = {C_LIGHT:.1f} m/s")
    print(f"beta = {beta:.2f}, gamma = {lorentz_gamma(beta):.6f}")
    print(f"Rest energy of 1 kg: {one_kg_rest:.6e} J")
    print()
    print(summary.to_string(index=False, justify='right', float_format=lambda x: f"{x:.6e}"))


if __name__ == "__main__":
    main()

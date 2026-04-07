"""Minimal runnable MVP for Bekenstein-Hawking entropy.

This script builds a transparent computational chain for a Schwarzschild black hole:
M -> r_s -> A -> S_BH, and validates two key relations:
1) S_BH from horizon area matches the closed-form M^2 scaling.
2) First law consistency: dS/dM = c^2 / T_H.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import constants


@dataclass(frozen=True)
class EntropyConfig:
    """Configuration for Bekenstein-Hawking entropy MVP."""

    min_mass_kg: float = 1.0
    n_summary_points: int = 8
    n_validation_points: int = 320


def schwarzschild_radius_m(mass_kg: float | np.ndarray) -> np.ndarray:
    """Schwarzschild radius r_s = 2GM/c^2."""
    mass = np.maximum(np.asarray(mass_kg, dtype=float), 0.0)
    return 2.0 * constants.G * mass / constants.c**2


def horizon_area_m2(mass_kg: float | np.ndarray) -> np.ndarray:
    """Event-horizon area A = 4*pi*r_s^2."""
    r_s = schwarzschild_radius_m(mass_kg)
    return 4.0 * np.pi * r_s**2


def planck_area_m2() -> float:
    """Planck area l_p^2 = hbar*G/c^3."""
    return constants.hbar * constants.G / constants.c**3


def bekenstein_hawking_entropy_j_per_k(mass_kg: float | np.ndarray) -> np.ndarray:
    """Bekenstein-Hawking entropy S = k_B*c^3*A / (4*G*hbar)."""
    area = horizon_area_m2(mass_kg)
    return constants.k * constants.c**3 * area / (4.0 * constants.G * constants.hbar)


def entropy_closed_form_j_per_k(mass_kg: float | np.ndarray) -> np.ndarray:
    """Equivalent closed form for Schwarzschild BH: S = 4*pi*k_B*G*M^2 / (hbar*c)."""
    mass = np.maximum(np.asarray(mass_kg, dtype=float), 0.0)
    coeff = 4.0 * np.pi * constants.k * constants.G / (constants.hbar * constants.c)
    return coeff * mass**2


def hawking_temperature_kelvin(mass_kg: float | np.ndarray) -> np.ndarray:
    """Hawking temperature for a Schwarzschild black hole."""
    mass = np.maximum(np.asarray(mass_kg, dtype=float), 1e-30)
    return constants.hbar * constants.c**3 / (
        8.0 * np.pi * constants.G * mass * constants.k
    )


def entropy_in_kb_units(mass_kg: float | np.ndarray) -> np.ndarray:
    """Dimensionless entropy S / k_B."""
    return bekenstein_hawking_entropy_j_per_k(mass_kg) / constants.k


def entropy_in_bits(mass_kg: float | np.ndarray) -> np.ndarray:
    """Information content in bits: S / (k_B ln 2)."""
    return entropy_in_kb_units(mass_kg) / np.log(2.0)


def finite_difference_derivative(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """First derivative dy/dx on a monotonic, non-uniform grid."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    if x_arr.ndim != 1 or y_arr.ndim != 1 or x_arr.size != y_arr.size:
        raise ValueError("x and y must be 1D arrays with the same length")
    if x_arr.size < 3:
        raise ValueError("need at least 3 points for finite differences")

    dydx = np.empty_like(y_arr)
    dydx[0] = (y_arr[1] - y_arr[0]) / (x_arr[1] - x_arr[0])
    dydx[-1] = (y_arr[-1] - y_arr[-2]) / (x_arr[-1] - x_arr[-2])
    dydx[1:-1] = (y_arr[2:] - y_arr[:-2]) / (x_arr[2:] - x_arr[:-2])
    return dydx


def fit_loglog_power_law(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit log(y) = alpha + p*log(x), return (p, exp(alpha))."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    if np.any(x_arr <= 0.0) or np.any(y_arr <= 0.0):
        raise ValueError("power-law fit requires positive x and y")

    lx = np.log(x_arr)
    ly = np.log(y_arr)
    design = np.column_stack([np.ones_like(lx), lx])
    coef, _, _, _ = np.linalg.lstsq(design, ly, rcond=None)
    alpha = float(coef[0])
    p = float(coef[1])
    return p, float(np.exp(alpha))


def build_summary_table(masses_kg: np.ndarray) -> pd.DataFrame:
    area = horizon_area_m2(masses_kg)
    entropy_area = bekenstein_hawking_entropy_j_per_k(masses_kg)
    entropy_closed = entropy_closed_form_j_per_k(masses_kg)
    temp = hawking_temperature_kelvin(masses_kg)

    return pd.DataFrame(
        {
            "M_kg": masses_kg,
            "r_s_m": schwarzschild_radius_m(masses_kg),
            "A_m2": area,
            "T_H_K": temp,
            "S_BH_J_per_K": entropy_area,
            "S_over_kB": entropy_area / constants.k,
            "S_bits": entropy_in_bits(masses_kg),
            "rel_diff_area_vs_closed": np.abs(entropy_area - entropy_closed)
            / np.maximum(np.abs(entropy_closed), 1e-300),
        }
    )


def validate_first_law(mass_grid_kg: np.ndarray) -> pd.DataFrame:
    """Validate dS/dM = c^2 / T_H numerically."""
    masses = np.asarray(mass_grid_kg, dtype=float)
    entropy = bekenstein_hawking_entropy_j_per_k(masses)
    temperature = hawking_temperature_kelvin(masses)

    dsdm_num = finite_difference_derivative(masses, entropy)
    dsdm_rhs = constants.c**2 / temperature
    rel_error = np.abs(dsdm_num - dsdm_rhs) / np.maximum(np.abs(dsdm_rhs), 1e-300)

    return pd.DataFrame(
        {
            "M_kg": masses,
            "dS_dM_numeric": dsdm_num,
            "c2_over_T": dsdm_rhs,
            "rel_error": rel_error,
        }
    )


def main() -> None:
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.float_format", lambda x: f"{x:.6e}")

    cfg = EntropyConfig(min_mass_kg=1.0, n_summary_points=8, n_validation_points=320)

    summary_masses = np.geomspace(1.0e5, 1.0e10, cfg.n_summary_points)
    summary_masses = np.maximum(summary_masses, cfg.min_mass_kg)

    validation_masses = np.geomspace(1.0e5, 1.0e10, cfg.n_validation_points)
    validation_masses = np.maximum(validation_masses, cfg.min_mass_kg)

    print("=== Bekenstein-Hawking Entropy MVP (Schwarzschild) ===")
    print(f"Planck area l_p^2 = {planck_area_m2():.6e} m^2")
    print()

    summary = build_summary_table(summary_masses)
    print("--- Summary Table ---")
    print(summary.to_string(index=False))
    print()

    first_law_df = validate_first_law(validation_masses)
    max_first_law_error = float(first_law_df["rel_error"].iloc[1:-1].max())
    mean_first_law_error = float(first_law_df["rel_error"].iloc[1:-1].mean())

    entropy_values = bekenstein_hawking_entropy_j_per_k(validation_masses)
    power, prefactor = fit_loglog_power_law(validation_masses, entropy_values)

    print("--- Validation Metrics ---")
    print(f"max rel diff (area-form vs closed-form): {summary['rel_diff_area_vs_closed'].max():.3e}")
    print(f"first-law max rel error (interior points): {max_first_law_error:.3e}")
    print(f"first-law mean rel error (interior points): {mean_first_law_error:.3e}")
    print(f"log-log fit: S ~ prefactor * M^p, p = {power:.8f}, prefactor = {prefactor:.6e}")

    monotonic_ok = bool(np.all(np.diff(entropy_values) > 0.0))
    print(f"entropy monotonic in mass: {monotonic_ok}")

    assert monotonic_ok, "Entropy should increase monotonically with mass"
    assert float(summary["rel_diff_area_vs_closed"].max()) < 1e-12, "Two entropy formulas disagree"
    assert max_first_law_error < 5e-3, "First-law consistency check failed"
    assert abs(power - 2.0) < 1e-10, "Entropy-mass scaling exponent should be 2"


if __name__ == "__main__":
    main()

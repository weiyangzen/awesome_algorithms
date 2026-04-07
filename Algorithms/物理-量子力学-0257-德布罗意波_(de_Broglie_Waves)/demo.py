"""Minimal runnable MVP for de Broglie Waves.

This script demonstrates:
1) Non-relativistic de Broglie wavelength computation from kinetic energy.
2) Regression verification of lambda = h * (1/p).
3) Relativistic correction impact for accelerated electrons.
4) Numerical consistency between NumPy and PyTorch implementations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy import constants, stats
from sklearn.linear_model import LinearRegression


@dataclass(frozen=True)
class ParticleBatch:
    """Particle and energy setup for a batch wavelength computation."""

    particle: str
    mass_kg: float
    kinetic_energy_ev: np.ndarray


def momentum_nonrelativistic(mass_kg: float, kinetic_energy_j: np.ndarray) -> np.ndarray:
    """Return p = sqrt(2 m K) for non-relativistic particles."""
    if mass_kg <= 0.0:
        raise ValueError("mass_kg must be positive")
    if np.any(kinetic_energy_j <= 0.0):
        raise ValueError("kinetic energies must be positive")
    return np.sqrt(2.0 * mass_kg * kinetic_energy_j)


def de_broglie_wavelength(momentum_kg_m_s: np.ndarray) -> np.ndarray:
    """Return lambda = h / p."""
    if np.any(momentum_kg_m_s <= 0.0):
        raise ValueError("momentum values must be positive")
    return constants.h / momentum_kg_m_s


def compute_de_broglie_table(batch: ParticleBatch) -> pd.DataFrame:
    """Build a table for de Broglie quantities under non-relativistic approximation."""
    kinetic_energy_j = batch.kinetic_energy_ev * constants.e
    momentum = momentum_nonrelativistic(batch.mass_kg, kinetic_energy_j)
    wavelength = de_broglie_wavelength(momentum)
    velocity = momentum / batch.mass_kg
    wave_number = 2.0 * np.pi / wavelength

    # For free non-relativistic matter waves: v_phase = v_group / 2, v_group = v.
    v_group = velocity
    v_phase = 0.5 * velocity

    df = pd.DataFrame(
        {
            "particle": batch.particle,
            "E_eV": batch.kinetic_energy_ev,
            "K_J": kinetic_energy_j,
            "p_kg_m_s": momentum,
            "lambda_m": wavelength,
            "lambda_pm": wavelength * 1e12,
            "k_rad_m": wave_number,
            "v_group_m_s": v_group,
            "v_phase_m_s": v_phase,
        }
    )
    return df


def fit_lambda_inverse_p(momentum: np.ndarray, wavelength: np.ndarray) -> dict[str, float]:
    """Fit lambda = a * (1/p) + b and compare with theoretical a = h, b = 0."""
    x = (1.0 / momentum).reshape(-1, 1)
    y = wavelength

    model = LinearRegression(fit_intercept=True)
    model.fit(x, y)

    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    correlation = float(stats.pearsonr(x[:, 0], y).statistic)

    slope_rel_err = abs(slope - constants.h) / constants.h

    return {
        "slope": slope,
        "intercept": intercept,
        "correlation": correlation,
        "slope_rel_err": slope_rel_err,
    }


def electron_wavelength_vs_voltage(voltage_v: np.ndarray) -> pd.DataFrame:
    """Compare non-relativistic and relativistic electron de Broglie wavelengths."""
    if np.any(voltage_v <= 0.0):
        raise ValueError("voltage values must be positive")

    m = constants.m_e
    c = constants.c
    k_j = constants.e * voltage_v

    # Non-relativistic momentum and wavelength.
    p_nonrel = np.sqrt(2.0 * m * k_j)
    lambda_nonrel = constants.h / p_nonrel

    # Relativistic momentum from E^2 = (pc)^2 + (mc^2)^2.
    e0 = m * c * c
    e_total = e0 + k_j
    p_rel = np.sqrt(e_total * e_total - e0 * e0) / c
    lambda_rel = constants.h / p_rel

    rel_error = np.abs(lambda_nonrel - lambda_rel) / lambda_rel

    return pd.DataFrame(
        {
            "V_volt": voltage_v,
            "lambda_nonrel_pm": lambda_nonrel * 1e12,
            "lambda_rel_pm": lambda_rel * 1e12,
            "relative_error": rel_error,
        }
    )


def torch_numpy_consistency_check(kinetic_energy_ev: np.ndarray, wavelength_np: np.ndarray) -> float:
    """Compute max absolute difference between NumPy and PyTorch wavelength pipelines."""
    e_t = torch.tensor(kinetic_energy_ev, dtype=torch.float64)
    k_j_t = e_t * constants.e
    p_t = torch.sqrt(2.0 * constants.m_e * k_j_t)
    wavelength_t = constants.h / p_t

    wavelength_np_t = torch.tensor(wavelength_np, dtype=torch.float64)
    max_abs_diff = float(torch.max(torch.abs(wavelength_t - wavelength_np_t)).item())
    return max_abs_diff


def main() -> None:
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", lambda x: f"{x:.6e}")

    energies_ev = np.array([30.0, 50.0, 100.0, 150.0, 300.0, 1000.0], dtype=np.float64)
    batch = ParticleBatch(
        particle="electron",
        mass_kg=constants.m_e,
        kinetic_energy_ev=energies_ev,
    )

    print("=== Demo A: Non-relativistic de Broglie table ===")
    table = compute_de_broglie_table(batch)
    print(table.to_string(index=False))

    fit_report = fit_lambda_inverse_p(
        momentum=table["p_kg_m_s"].to_numpy(dtype=np.float64),
        wavelength=table["lambda_m"].to_numpy(dtype=np.float64),
    )

    print("\n=== Demo B: Regression check for lambda = h * (1/p) ===")
    for key, value in fit_report.items():
        print(f"{key:>16s}: {value:.12e}")

    torch_diff = torch_numpy_consistency_check(
        kinetic_energy_ev=energies_ev,
        wavelength_np=table["lambda_m"].to_numpy(dtype=np.float64),
    )
    print("\n=== Demo C: NumPy vs PyTorch consistency ===")
    print(f"{'max_abs_diff_m':>16s}: {torch_diff:.12e}")

    voltages = np.array([50.0, 500.0, 5_000.0, 50_000.0, 100_000.0], dtype=np.float64)
    rel_table = electron_wavelength_vs_voltage(voltages)
    print("\n=== Demo D: Relativistic correction vs acceleration voltage ===")
    print(rel_table.to_string(index=False))

    low_v_error = float(rel_table.iloc[0]["relative_error"])
    high_v_error = float(rel_table.iloc[-1]["relative_error"])

    assert fit_report["slope_rel_err"] < 1e-12, (
        f"Linear-fit slope should recover h; got relative error {fit_report['slope_rel_err']:.3e}"
    )
    assert abs(fit_report["intercept"]) < 1e-20, (
        f"Intercept should be near zero; got {fit_report['intercept']:.3e}"
    )
    assert fit_report["correlation"] > 0.999999999999, (
        f"Correlation should be near 1; got {fit_report['correlation']:.12f}"
    )
    assert torch_diff < 1e-22, f"Torch/NumPy mismatch too large: {torch_diff:.3e}"

    # Relativistic correction should be tiny at low voltage and non-negligible at 100 kV.
    assert low_v_error < 1e-3, f"Low-voltage non-relativistic error too large: {low_v_error:.3e}"
    assert high_v_error > 3e-2, f"High-voltage relativistic correction unexpectedly small: {high_v_error:.3e}"

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()

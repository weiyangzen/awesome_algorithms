"""Rutherford scattering formula MVP.

This script computes differential cross-sections, synthesizes noisy observations,
and fits an effective target charge with SciPy curve fitting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


# Physical constants (SI)
ELEMENTARY_CHARGE_C = 1.602_176_634e-19
EPSILON_0_F_PER_M = 8.854_187_812_8e-12
PI = np.pi


def mev_to_joule(energy_mev: float) -> float:
    """Convert kinetic energy from MeV to Joule."""
    return energy_mev * 1.0e6 * ELEMENTARY_CHARGE_C


def rutherford_dsigma_domega(
    theta_rad: np.ndarray | float,
    z_projectile: float,
    z_target: float,
    kinetic_energy_joule: float,
) -> np.ndarray:
    """Differential cross section dσ/dΩ for Rutherford scattering.

    Formula:
    dσ/dΩ = [ Z1*Z2*e^2 / (16π*ε0*E) ]^2 * csc^4(θ/2)

    Returns SI units: m^2 / sr.
    """
    theta = np.asarray(theta_rad, dtype=float)
    half_theta = 0.5 * theta

    k = z_projectile * z_target * ELEMENTARY_CHARGE_C**2 / (4.0 * PI * EPSILON_0_F_PER_M)
    prefactor = (k / (4.0 * kinetic_energy_joule)) ** 2
    sin_half = np.sin(half_theta)
    return prefactor / np.power(sin_half, 4)


def _fit_model(
    theta_rad: np.ndarray,
    z_target: float,
    *,
    z_projectile: float,
    kinetic_energy_joule: float,
) -> np.ndarray:
    """Wrapper for curve_fit: keep projectile/energy fixed, fit z_target."""
    return rutherford_dsigma_domega(
        theta_rad=theta_rad,
        z_projectile=z_projectile,
        z_target=z_target,
        kinetic_energy_joule=kinetic_energy_joule,
    )


def main() -> None:
    # Scenario: alpha particle (Z1=2) scattering by gold nucleus (Z2=79).
    z_projectile = 2.0
    z_target_true = 79.0
    energy_mev = 5.0
    energy_joule = mev_to_joule(energy_mev)

    # Avoid very small angles to keep finite values in a practical numeric demo.
    theta_deg = np.linspace(20.0, 160.0, 40)
    theta_rad = np.deg2rad(theta_deg)

    dsigma_true = rutherford_dsigma_domega(
        theta_rad=theta_rad,
        z_projectile=z_projectile,
        z_target=z_target_true,
        kinetic_energy_joule=energy_joule,
    )

    # Synthetic measurements: multiplicative Gaussian noise (deterministic seed).
    rng = np.random.default_rng(seed=20260407)
    rel_noise = 0.08
    dsigma_obs = dsigma_true * (1.0 + rng.normal(0.0, rel_noise, size=theta_rad.size))
    dsigma_obs = np.clip(dsigma_obs, a_min=1e-40, a_max=None)

    # Fit target charge from observations.
    scale = 1.0e26
    fit_fn = lambda ang, zt: scale * _fit_model(  # noqa: E731
        ang, zt, z_projectile=z_projectile, kinetic_energy_joule=energy_joule
    )
    popt, pcov = curve_fit(
        fit_fn,
        theta_rad,
        dsigma_obs * scale,
        p0=[60.0],
        bounds=(1.0, 120.0),
        maxfev=20_000,
    )
    z_target_fit = float(popt[0])
    z_target_std = float(np.sqrt(np.diag(pcov))[0]) if pcov.size else float("nan")

    dsigma_fit = fit_fn(theta_rad, z_target_fit) / scale
    rel_rmse = float(np.sqrt(np.mean(((dsigma_fit - dsigma_obs) / dsigma_obs) ** 2)))

    table = pd.DataFrame(
        {
            "theta_deg": theta_deg,
            "dsigma_obs_m2_per_sr": dsigma_obs,
            "dsigma_fit_m2_per_sr": dsigma_fit,
            "relative_error_fit_vs_obs": (dsigma_fit - dsigma_obs) / dsigma_obs,
        }
    )

    print("=== Rutherford Scattering Formula MVP ===")
    print(f"Projectile charge Z1: {z_projectile:.0f}")
    print(f"True target charge Z2: {z_target_true:.0f}")
    print(f"Fitted target charge Z2: {z_target_fit:.3f} ± {z_target_std:.3f}")
    print(f"Kinetic energy: {energy_mev:.2f} MeV")
    print(f"Angle range: {theta_deg.min():.1f}° to {theta_deg.max():.1f}°")
    print(f"Relative RMSE (fit vs synthetic obs): {rel_rmse:.4f}")
    print()
    print("Sample rows:")
    print(table.head(8).to_string(index=False, justify='center', float_format=lambda x: f"{x:.4e}"))


if __name__ == "__main__":
    main()

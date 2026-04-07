"""Dipole radiation MVP.

Model:
- Oscillating electric dipole: p(t) = p0 * cos(omega t) * z_hat
- Radiation pattern (time-averaged, far field):
    dP/dOmega = (omega^4 p0^2 / (32 pi^2 eps0 c^3)) * sin^2(theta)
- Total radiated power:
    P = omega^4 p0^2 / (12 pi eps0 c^3)

The script verifies these relations numerically and prints key metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS0 = 8.854_187_812_8e-12  # vacuum permittivity (F/m)
MU0 = 1.256_637_062_12e-6  # vacuum permeability (H/m)
C0 = 1.0 / np.sqrt(EPS0 * MU0)  # speed of light in vacuum (m/s)


@dataclass(frozen=True)
class DipoleConfig:
    """Configuration for harmonic electric dipole radiation."""

    p0_coulomb_meter: float = 1.0e-9  # dipole moment amplitude p0 (C*m)
    frequency_hz: float = 10.0e6  # oscillation frequency f (Hz)
    observation_radius_m: float = 25.0  # far-field observation radius (m)
    n_theta: int = 720  # polar discretization for sphere integration
    n_phi: int = 720  # azimuthal discretization for sphere integration
    n_time: int = 720  # samples over one period for time averaging


def angular_power_density(theta: np.ndarray, cfg: DipoleConfig) -> np.ndarray:
    """Return analytic time-averaged dP/dOmega(theta) for the dipole."""
    omega = 2.0 * np.pi * cfg.frequency_hz
    prefactor = omega**4 * cfg.p0_coulomb_meter**2 / (32.0 * np.pi**2 * EPS0 * C0**3)
    return prefactor * np.sin(theta) ** 2


def total_power_theory(cfg: DipoleConfig) -> float:
    """Closed-form total power radiated by a harmonic electric dipole."""
    omega = 2.0 * np.pi * cfg.frequency_hz
    return omega**4 * cfg.p0_coulomb_meter**2 / (12.0 * np.pi * EPS0 * C0**3)


def far_field_e_theta(theta: np.ndarray, t: np.ndarray, cfg: DipoleConfig) -> np.ndarray:
    """Far-zone theta-component electric field E_theta(theta, t).

    Formula:
        E_theta = (mu0 p0 omega^2 / (4 pi r)) * sin(theta) * cos(omega (t - r/c))
    """
    omega = 2.0 * np.pi * cfg.frequency_hz
    amp = MU0 * cfg.p0_coulomb_meter * omega**2 / (4.0 * np.pi * cfg.observation_radius_m)
    phase = omega * (t - cfg.observation_radius_m / C0)
    return amp * np.sin(theta)[:, None] * np.cos(phase)[None, :]


def integrate_total_power_from_domega(dpdomega_theta: np.ndarray, cfg: DipoleConfig) -> float:
    """Integrate total power on a theta-phi grid from dP/dOmega(theta)."""
    dtheta = np.pi / cfg.n_theta
    dphi = 2.0 * np.pi / cfg.n_phi
    theta = (np.arange(cfg.n_theta) + 0.5) * dtheta

    # Broadcast to 2D sphere grid: dpdomega(theta, phi) = dpdomega(theta)
    dpdomega_2d = np.repeat(dpdomega_theta[:, None], cfg.n_phi, axis=1)
    domega_2d = np.sin(theta)[:, None] * dtheta * dphi
    return float(np.sum(dpdomega_2d * domega_2d))


def main() -> None:
    cfg = DipoleConfig()

    dtheta = np.pi / cfg.n_theta
    theta = (np.arange(cfg.n_theta) + 0.5) * dtheta

    # Path A: analytic dP/dOmega and sphere integration.
    dpdomega_analytic = angular_power_density(theta, cfg)
    p_numeric_analytic = integrate_total_power_from_domega(dpdomega_analytic, cfg)
    p_theory = total_power_theory(cfg)

    # Path B: reconstruct dP/dOmega from far-field E_theta and averaged Poynting flux.
    period = 1.0 / cfg.frequency_hz
    t = np.linspace(0.0, period, cfg.n_time, endpoint=False)
    e_theta = far_field_e_theta(theta, t, cfg)
    s_r = (e_theta**2) / (MU0 * C0)  # instantaneous radial Poynting magnitude
    s_r_avg = np.mean(s_r, axis=1)
    dpdomega_from_fields = cfg.observation_radius_m**2 * s_r_avg
    p_numeric_fields = integrate_total_power_from_domega(dpdomega_from_fields, cfg)

    # Shape validation: normalized pattern should follow sin^2(theta).
    norm_pattern = dpdomega_analytic / np.max(dpdomega_analytic)
    norm_sin2 = (np.sin(theta) ** 2) / np.max(np.sin(theta) ** 2)
    max_shape_error = float(np.max(np.abs(norm_pattern - norm_sin2)))

    # Agreement checks.
    np.testing.assert_allclose(p_numeric_analytic, p_theory, rtol=2e-5, atol=0.0)
    np.testing.assert_allclose(p_numeric_fields, p_theory, rtol=2e-5, atol=0.0)
    np.testing.assert_allclose(dpdomega_from_fields, dpdomega_analytic, rtol=2e-5, atol=0.0)
    np.testing.assert_allclose(max_shape_error, 0.0, atol=1e-14, rtol=0.0)

    deg = np.array([0.5, 30.0, 60.0, 90.0, 120.0, 150.0, 179.5])
    sample_theta = np.deg2rad(deg)
    sample_dpdomega = angular_power_density(sample_theta, cfg)

    print("Dipole Radiation MVP")
    print(
        f"Config: p0={cfg.p0_coulomb_meter:.3e} C*m, f={cfg.frequency_hz:.3e} Hz, "
        f"r={cfg.observation_radius_m:.2f} m"
    )
    print(f"Grid: n_theta={cfg.n_theta}, n_phi={cfg.n_phi}, n_time={cfg.n_time}")
    print("--- Total power checks ---")
    print(f"P_theory                 : {p_theory:.6e} W")
    print(f"P_numeric (analytic dΩ)  : {p_numeric_analytic:.6e} W")
    print(f"P_numeric (field average): {p_numeric_fields:.6e} W")
    print(
        f"RelErr analytic/theory   : "
        f"{abs(p_numeric_analytic - p_theory) / p_theory:.3e}"
    )
    print(
        f"RelErr field/theory      : "
        f"{abs(p_numeric_fields - p_theory) / p_theory:.3e}"
    )
    print("--- Angular samples dP/dΩ (W/sr) ---")
    for d, v in zip(deg, sample_dpdomega):
        print(f"theta={d:6.1f} deg -> {v:.6e}")


if __name__ == "__main__":
    main()

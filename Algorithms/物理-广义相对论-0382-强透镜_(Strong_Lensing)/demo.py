"""Strong lensing MVP using a singular isothermal sphere (SIS) lens.

The script demonstrates a source-level, auditable pipeline:
1) compute angular-diameter distances in flat LCDM;
2) compute SIS Einstein angle;
3) solve SIS lens equation for image multiplicity and magnifications;
4) estimate strong-lensing probability via Monte Carlo source sampling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import quad

# SI constants
C_LIGHT = 299_792_458.0
MPC_M = 3.085677581491367e22
RAD_TO_ARCSEC = 206_264.80624709636
ARCSEC_TO_RAD = 1.0 / RAD_TO_ARCSEC


@dataclass(frozen=True)
class Cosmology:
    """Flat LCDM cosmology parameters."""

    h0_km_s_mpc: float = 70.0
    omega_m: float = 0.3
    omega_lambda: float = 0.7


@dataclass(frozen=True)
class SISLensConfig:
    """Single-lens setup for strong lensing demonstration."""

    sigma_v_km_s: float
    z_lens: float
    z_source: float


def _validate_positive_finite(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value}")


def _h0_si(cosmo: Cosmology) -> float:
    return cosmo.h0_km_s_mpc * 1_000.0 / MPC_M


def _e_of_z(z: float, cosmo: Cosmology) -> float:
    return np.sqrt(cosmo.omega_m * (1.0 + z) ** 3 + cosmo.omega_lambda)


def comoving_distance_m(z: float, cosmo: Cosmology) -> float:
    """Line-of-sight comoving distance in a flat LCDM universe."""
    if not np.isfinite(z) or z < 0.0:
        raise ValueError(f"z must be finite and >= 0, got {z}")
    if np.isclose(z, 0.0):
        return 0.0
    integral, _ = quad(lambda zp: 1.0 / _e_of_z(zp, cosmo), 0.0, z, epsabs=1e-11, epsrel=1e-11, limit=200)
    return (C_LIGHT / _h0_si(cosmo)) * integral


def angular_diameter_distance_m(z: float, cosmo: Cosmology) -> float:
    return comoving_distance_m(z, cosmo) / (1.0 + z)


def angular_diameter_distance_between_m(z1: float, z2: float, cosmo: Cosmology) -> float:
    if z2 <= z1:
        raise ValueError(f"require z2 > z1, got z1={z1}, z2={z2}")
    dc1 = comoving_distance_m(z1, cosmo)
    dc2 = comoving_distance_m(z2, cosmo)
    return (dc2 - dc1) / (1.0 + z2)


def einstein_radius_sis_rad(sigma_v_km_s: float, d_ls_m: float, d_s_m: float) -> float:
    """Einstein angle for SIS: theta_E = 4*pi*(sigma_v/c)^2 * D_ls/D_s."""
    _validate_positive_finite("sigma_v_km_s", sigma_v_km_s)
    _validate_positive_finite("d_ls_m", d_ls_m)
    _validate_positive_finite("d_s_m", d_s_m)
    sigma_si = sigma_v_km_s * 1_000.0
    return 4.0 * np.pi * (sigma_si / C_LIGHT) ** 2 * (d_ls_m / d_s_m)


def sis_image_positions(beta_rad: float, theta_e_rad: float) -> tuple[float, float | None]:
    """Return outer and inner image angles for beta >= 0 in SIS model.

    Lens equation (1D signed form):
    beta = theta - theta_E * sign(theta)
    """
    _validate_positive_finite("theta_e_rad", theta_e_rad)
    if not np.isfinite(beta_rad) or beta_rad < 0.0:
        raise ValueError(f"beta_rad must be finite and >= 0, got {beta_rad}")

    theta_plus = beta_rad + theta_e_rad
    if beta_rad < theta_e_rad:
        theta_minus = beta_rad - theta_e_rad  # negative parity image
        return theta_plus, theta_minus
    return theta_plus, None


def sis_lens_equation_residual(theta_rad: float, beta_rad: float, theta_e_rad: float) -> float:
    if np.isclose(theta_rad, 0.0):
        return np.inf
    return theta_rad - theta_e_rad * np.sign(theta_rad) - beta_rad


def sis_magnifications(beta_rad: float, theta_e_rad: float) -> tuple[float, float | None, float]:
    """Return (mu_plus, mu_minus_or_none, total_abs_mag)."""
    _validate_positive_finite("theta_e_rad", theta_e_rad)
    if beta_rad < 0.0 or not np.isfinite(beta_rad):
        raise ValueError(f"beta_rad must be finite and >= 0, got {beta_rad}")

    if np.isclose(beta_rad, 0.0):
        return np.inf, -np.inf, np.inf

    mu_plus = 1.0 + theta_e_rad / beta_rad
    if beta_rad < theta_e_rad:
        mu_minus = 1.0 - theta_e_rad / beta_rad
        mu_total = abs(mu_plus) + abs(mu_minus)
        return mu_plus, mu_minus, mu_total

    return mu_plus, None, abs(mu_plus)


def strong_lensing_cross_section_source_plane(theta_e_rad: float, d_s_m: float) -> float:
    """Strong-lensing area in source plane (m^2) for SIS (beta_crit = theta_E)."""
    _validate_positive_finite("theta_e_rad", theta_e_rad)
    _validate_positive_finite("d_s_m", d_s_m)
    beta_crit_m = theta_e_rad * d_s_m
    return np.pi * beta_crit_m * beta_crit_m


def simulate_sis_table(theta_e_arcsec: float, beta_arcsec_values: np.ndarray) -> pd.DataFrame:
    theta_e_rad = theta_e_arcsec * ARCSEC_TO_RAD
    rows: list[dict[str, float | int]] = []

    for beta_arcsec in beta_arcsec_values:
        beta_rad = float(beta_arcsec) * ARCSEC_TO_RAD
        theta_p, theta_m = sis_image_positions(beta_rad, theta_e_rad)
        mu_p, mu_m, mu_total = sis_magnifications(beta_rad, theta_e_rad)

        residuals = [abs(sis_lens_equation_residual(theta_p, beta_rad, theta_e_rad))]
        n_images = 1
        theta_m_arcsec = np.nan
        mu_m_out = np.nan

        if theta_m is not None:
            residuals.append(abs(sis_lens_equation_residual(theta_m, beta_rad, theta_e_rad)))
            n_images = 2
            theta_m_arcsec = float(theta_m * RAD_TO_ARCSEC)
            mu_m_out = float(mu_m)

        rows.append(
            {
                "beta_arcsec": float(beta_arcsec),
                "theta_plus_arcsec": float(theta_p * RAD_TO_ARCSEC),
                "theta_minus_arcsec": theta_m_arcsec,
                "n_images": n_images,
                "image_separation_arcsec": float(abs(theta_p - theta_m) * RAD_TO_ARCSEC) if theta_m is not None else 0.0,
                "mu_plus": float(mu_p),
                "mu_minus": mu_m_out,
                "mu_total_abs": float(mu_total),
                "is_strong_lensing": int(n_images >= 2),
                "max_abs_residual": float(max(residuals)),
            }
        )

    return pd.DataFrame(rows)


def monte_carlo_strong_fraction(
    theta_e_arcsec: float,
    beta_max_arcsec: float,
    n_samples: int = 50_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Estimate strong-lensing fraction and compare with analytic area ratio.

    Sources are sampled uniformly in a disk of radius beta_max.
    """
    _validate_positive_finite("theta_e_arcsec", theta_e_arcsec)
    _validate_positive_finite("beta_max_arcsec", beta_max_arcsec)
    if n_samples < 1_000:
        raise ValueError("n_samples should be >= 1000 for stable Monte Carlo statistics")

    rng = np.random.default_rng(seed)
    u = rng.random(n_samples)
    r = beta_max_arcsec * np.sqrt(u)
    is_strong = r < theta_e_arcsec
    empirical = float(is_strong.mean())

    analytic = min(1.0, (theta_e_arcsec / beta_max_arcsec) ** 2)
    sigma = float(np.sqrt(max(analytic * (1.0 - analytic), 1e-16) / n_samples))
    return empirical, analytic, sigma


def main() -> None:
    cosmo = Cosmology(h0_km_s_mpc=70.0, omega_m=0.3, omega_lambda=0.7)
    cfg = SISLensConfig(sigma_v_km_s=260.0, z_lens=0.5, z_source=2.0)

    _validate_positive_finite("sigma_v_km_s", cfg.sigma_v_km_s)
    if cfg.z_source <= cfg.z_lens:
        raise ValueError("z_source must be larger than z_lens")

    d_l = angular_diameter_distance_m(cfg.z_lens, cosmo)
    d_s = angular_diameter_distance_m(cfg.z_source, cosmo)
    d_ls = angular_diameter_distance_between_m(cfg.z_lens, cfg.z_source, cosmo)

    theta_e_rad = einstein_radius_sis_rad(cfg.sigma_v_km_s, d_ls, d_s)
    theta_e_arcsec = theta_e_rad * RAD_TO_ARCSEC

    beta_samples = np.linspace(0.10 * theta_e_arcsec, 2.50 * theta_e_arcsec, 12)
    table = simulate_sis_table(theta_e_arcsec, beta_samples)

    sigma_strong_m2 = strong_lensing_cross_section_source_plane(theta_e_rad, d_s)
    sigma_strong_kpc2 = sigma_strong_m2 / (3.085677581491367e19**2)

    beta_max = 3.0 * theta_e_arcsec
    empirical_p, analytic_p, mc_sigma = monte_carlo_strong_fraction(
        theta_e_arcsec=theta_e_arcsec,
        beta_max_arcsec=beta_max,
        n_samples=60_000,
        seed=2026,
    )

    print("=== Strong Lensing MVP (SIS Lens) ===")
    print(f"Velocity dispersion sigma_v : {cfg.sigma_v_km_s:.1f} km/s")
    print(f"Lens redshift z_l           : {cfg.z_lens:.3f}")
    print(f"Source redshift z_s         : {cfg.z_source:.3f}")
    print(f"D_l  = {d_l / MPC_M:.3f} Mpc")
    print(f"D_s  = {d_s / MPC_M:.3f} Mpc")
    print(f"D_ls = {d_ls / MPC_M:.3f} Mpc")
    print(f"Einstein angle theta_E      : {theta_e_arcsec:.4f} arcsec")
    print(f"Strong cross-section (src)  : {sigma_strong_kpc2:.4f} kpc^2")
    print()
    print("Sampled source offsets and image configurations:")
    print(table.to_string(index=False, justify="center", float_format=lambda x: f"{x: .6e}"))
    print()
    print(f"Monte Carlo strong fraction (beta_max = 3 theta_E): {empirical_p:.6f}")
    print(f"Analytic area ratio prediction                    : {analytic_p:.6f}")
    print(f"Monte Carlo 1-sigma uncertainty                    : {mc_sigma:.6f}")

    max_resid = float(table["max_abs_residual"].max())
    if max_resid > 1e-12:
        raise RuntimeError(f"SIS lens equation residual too large: {max_resid:.3e}")

    inside_mask = table["beta_arcsec"] < theta_e_arcsec
    outside_mask = ~inside_mask
    if not bool((table.loc[inside_mask, "n_images"] == 2).all()):
        raise RuntimeError("Expected two images for beta < theta_E in SIS model")
    if not bool((table.loc[outside_mask, "n_images"] == 1).all()):
        raise RuntimeError("Expected one image for beta >= theta_E in SIS model")

    # Agreement gate: allow 4 sigma + tiny numerical slack.
    if abs(empirical_p - analytic_p) > 4.0 * mc_sigma + 0.005:
        raise RuntimeError("Monte Carlo strong-lensing fraction deviates from analytic prediction")


if __name__ == "__main__":
    main()

"""Point-mass gravitational lensing MVP.

This demo implements a transparent, source-level minimal model for
gravitational lensing in the thin-lens approximation:
1) compute angular-diameter distances from a flat LCDM background;
2) compute Einstein radius of a point lens;
3) solve the scalar lens equation for image positions and magnifications;
4) run a tiny 2D ray-mapping example for a Gaussian source.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import quad

# Physical constants (SI)
G_NEWTON = 6.67430e-11  # m^3 kg^-1 s^-2
C_LIGHT = 299_792_458.0  # m s^-1
MPC_M = 3.085677581491367e22  # m
SOLAR_MASS_KG = 1.98847e30  # kg
RAD_TO_ARCSEC = 206_264.80624709636
ARCSEC_TO_RAD = 1.0 / RAD_TO_ARCSEC


@dataclass(frozen=True)
class Cosmology:
    """Flat LCDM parameters."""

    h0_km_s_mpc: float = 70.0
    omega_m: float = 0.3
    omega_lambda: float = 0.7


def _validate_positive(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value}")


def _h0_si(cosmo: Cosmology) -> float:
    return cosmo.h0_km_s_mpc * 1_000.0 / MPC_M


def _e_of_z(z: float, cosmo: Cosmology) -> float:
    return np.sqrt(cosmo.omega_m * (1.0 + z) ** 3 + cosmo.omega_lambda)


def comoving_distance_m(z: float, cosmo: Cosmology) -> float:
    """Line-of-sight comoving distance in meters."""
    if z < 0.0 or not np.isfinite(z):
        raise ValueError(f"z must be finite and >= 0, got {z}")
    if z == 0.0:
        return 0.0

    integrand = lambda zp: 1.0 / _e_of_z(zp, cosmo)
    integral, _ = quad(integrand, 0.0, z, epsabs=1e-11, epsrel=1e-11, limit=200)
    return (C_LIGHT / _h0_si(cosmo)) * integral


def angular_diameter_distance_m(z: float, cosmo: Cosmology) -> float:
    """Observer-to-object angular diameter distance."""
    return comoving_distance_m(z, cosmo) / (1.0 + z)


def angular_diameter_distance_between_m(z1: float, z2: float, cosmo: Cosmology) -> float:
    """Angular diameter distance between two redshifts (z2 > z1)."""
    if z2 <= z1:
        raise ValueError(f"require z2 > z1, got z1={z1}, z2={z2}")
    dc1 = comoving_distance_m(z1, cosmo)
    dc2 = comoving_distance_m(z2, cosmo)
    return (dc2 - dc1) / (1.0 + z2)


def einstein_radius_rad(mass_kg: float, d_l: float, d_s: float, d_ls: float) -> float:
    """Einstein angle in radians for a point lens."""
    _validate_positive("mass_kg", mass_kg)
    _validate_positive("d_l", d_l)
    _validate_positive("d_s", d_s)
    _validate_positive("d_ls", d_ls)
    factor = (4.0 * G_NEWTON * mass_kg / (C_LIGHT**2)) * (d_ls / (d_l * d_s))
    return np.sqrt(factor)


def image_positions_point_lens(beta_rad: float, theta_e_rad: float) -> tuple[float, float]:
    """Solve theta^2 - beta*theta - theta_E^2 = 0."""
    _validate_positive("theta_e_rad", theta_e_rad)
    if not np.isfinite(beta_rad):
        raise ValueError(f"beta_rad must be finite, got {beta_rad}")
    disc = np.sqrt(beta_rad * beta_rad + 4.0 * theta_e_rad * theta_e_rad)
    theta_plus = 0.5 * (beta_rad + disc)
    theta_minus = 0.5 * (beta_rad - disc)
    return theta_plus, theta_minus


def magnifications_point_lens(beta_rad: float, theta_e_rad: float) -> tuple[float, float, float]:
    """Return (mu_plus, mu_minus, |mu_plus|+|mu_minus|)."""
    _validate_positive("theta_e_rad", theta_e_rad)
    u = beta_rad / theta_e_rad
    if np.isclose(u, 0.0):
        return np.inf, -np.inf, np.inf

    common = (u * u + 2.0) / (2.0 * u * np.sqrt(u * u + 4.0))
    mu_plus = 0.5 + common
    mu_minus = 0.5 - common  # negative parity image
    mu_total = abs(mu_plus) + abs(mu_minus)
    return mu_plus, mu_minus, mu_total


def lens_equation_residual(theta_rad: float, beta_rad: float, theta_e_rad: float) -> float:
    """Residual of beta = theta - theta_E^2/theta."""
    if np.isclose(theta_rad, 0.0):
        return np.inf
    return theta_rad - (theta_e_rad * theta_e_rad) / theta_rad - beta_rad


def simulate_point_lens_table(theta_e_arcsec: float, beta_arcsec_values: np.ndarray) -> pd.DataFrame:
    theta_e_rad = theta_e_arcsec * ARCSEC_TO_RAD
    rows = []
    for beta_arcsec in beta_arcsec_values:
        beta_rad = beta_arcsec * ARCSEC_TO_RAD
        theta_p, theta_m = image_positions_point_lens(beta_rad, theta_e_rad)
        mu_p, mu_m, mu_tot = magnifications_point_lens(beta_rad, theta_e_rad)
        res_p = lens_equation_residual(theta_p, beta_rad, theta_e_rad)
        res_m = lens_equation_residual(theta_m, beta_rad, theta_e_rad)
        rows.append(
            {
                "beta_arcsec": float(beta_arcsec),
                "theta_plus_arcsec": float(theta_p * RAD_TO_ARCSEC),
                "theta_minus_arcsec": float(theta_m * RAD_TO_ARCSEC),
                "mu_plus": float(mu_p),
                "mu_minus": float(mu_m),
                "mu_total_abs": float(mu_tot),
                "max_abs_residual": float(max(abs(res_p), abs(res_m))),
            }
        )
    return pd.DataFrame(rows)


def ray_map_gaussian_source(
    theta_e_arcsec: float,
    source_center_arcsec: tuple[float, float],
    source_sigma_arcsec: float = 0.10,
    grid_half_size_arcsec: float = 2.5,
    grid_points: int = 321,
) -> tuple[float, float]:
    """Return (unlensed_flux, lensed_flux) on an image-plane grid."""
    _validate_positive("theta_e_arcsec", theta_e_arcsec)
    _validate_positive("source_sigma_arcsec", source_sigma_arcsec)
    _validate_positive("grid_half_size_arcsec", grid_half_size_arcsec)
    if grid_points < 11:
        raise ValueError("grid_points must be >= 11")

    grid_1d = np.linspace(-grid_half_size_arcsec, grid_half_size_arcsec, grid_points)
    theta_x, theta_y = np.meshgrid(grid_1d, grid_1d, indexing="xy")
    r2 = theta_x * theta_x + theta_y * theta_y
    safe_r2 = np.maximum(r2, 1e-12)

    # Lens equation in vector form: beta = theta - alpha(theta), alpha ~ theta_E^2 * theta / |theta|^2
    alpha_factor = theta_e_arcsec * theta_e_arcsec / safe_r2
    beta_x = theta_x - alpha_factor * theta_x
    beta_y = theta_y - alpha_factor * theta_y

    cx, cy = source_center_arcsec
    s2 = source_sigma_arcsec * source_sigma_arcsec
    unlensed = np.exp(-0.5 * ((theta_x - cx) ** 2 + (theta_y - cy) ** 2) / s2)
    lensed = np.exp(-0.5 * ((beta_x - cx) ** 2 + (beta_y - cy) ** 2) / s2)

    # Grid cell area in arcsec^2 is common factor, so ratio can be computed from sums.
    return float(unlensed.sum()), float(lensed.sum())


def main() -> None:
    cosmo = Cosmology(h0_km_s_mpc=70.0, omega_m=0.3, omega_lambda=0.7)

    lens_mass_msun = 1.0e12
    z_lens = 0.5
    z_source = 2.0

    d_l = angular_diameter_distance_m(z_lens, cosmo)
    d_s = angular_diameter_distance_m(z_source, cosmo)
    d_ls = angular_diameter_distance_between_m(z_lens, z_source, cosmo)

    theta_e_rad = einstein_radius_rad(lens_mass_msun * SOLAR_MASS_KG, d_l, d_s, d_ls)
    theta_e_arcsec = theta_e_rad * RAD_TO_ARCSEC

    beta_samples = np.linspace(0.05, 1.20, 10)  # arcsec
    df = simulate_point_lens_table(theta_e_arcsec, beta_samples)

    unlensed_flux, lensed_flux = ray_map_gaussian_source(
        theta_e_arcsec=theta_e_arcsec,
        source_center_arcsec=(0.35, 0.15),
        source_sigma_arcsec=0.10,
        grid_half_size_arcsec=2.5,
        grid_points=321,
    )
    flux_mag_est = lensed_flux / unlensed_flux

    print("=== Gravitational Lensing MVP (Point Lens) ===")
    print(f"Lens mass      : {lens_mass_msun:.3e} Msun")
    print(f"Redshift lens  : z_l = {z_lens:.3f}")
    print(f"Redshift source: z_s = {z_source:.3f}")
    print(f"D_l  = {d_l / MPC_M:.3f} Mpc")
    print(f"D_s  = {d_s / MPC_M:.3f} Mpc")
    print(f"D_ls = {d_ls / MPC_M:.3f} Mpc")
    print(f"Einstein radius theta_E = {theta_e_arcsec:.4f} arcsec")
    print()
    print("Image solutions for selected source offsets:")
    print(df.to_string(index=False, justify="center", float_format=lambda x: f"{x: .6e}"))
    print()
    print(f"2D Gaussian source flux (unlensed): {unlensed_flux:.6e}")
    print(f"2D Gaussian source flux (lensed)  : {lensed_flux:.6e}")
    print(f"Estimated total magnification     : {flux_mag_est:.6f}")

    max_residual = float(df["max_abs_residual"].max())
    if max_residual > 1e-12:
        raise RuntimeError(f"lens equation residual too large: {max_residual:.3e}")
    if not np.isfinite(flux_mag_est) or flux_mag_est <= 1.0:
        raise RuntimeError("unexpected magnification estimate from 2D ray mapping")


if __name__ == "__main__":
    main()

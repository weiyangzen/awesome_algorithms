"""Horizon problem MVP: CMB causal patch size without/with inflation."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from scipy import integrate


@dataclass(frozen=True)
class HorizonParams:
    """Minimal LCDM-like parameter set for a pedagogical horizon-problem demo."""

    h0_km_s_mpc: float = 67.4
    omega_m: float = 0.315
    omega_r: float = 9.2e-5
    z_dec: float = 1089.0
    a_min: float = 1.0e-8
    c_km_s: float = 299792.458


def omega_lambda(params: HorizonParams) -> float:
    """Infer flat-universe dark-energy density fraction."""
    value = 1.0 - params.omega_m - params.omega_r
    if value <= 0.0:
        raise ValueError("Require omega_lambda > 0 for this MVP.")
    return value


def e_of_a(a: float, params: HorizonParams) -> float:
    """Dimensionless expansion rate E(a)=H(a)/H0 under flat LCDM."""
    ol = omega_lambda(params)
    return float(np.sqrt(params.omega_r / a**4 + params.omega_m / a**3 + ol))


def comoving_distance_mpc(a_start: float, a_end: float, params: HorizonParams) -> float:
    """Compute comoving radial distance c/H0 * int da/(a^2 E(a))."""
    if not (0.0 < a_start < a_end <= 1.0):
        raise ValueError("Require 0 < a_start < a_end <= 1.")

    def integrand(a: float) -> float:
        return 1.0 / (a * a * e_of_a(a, params))

    integral, _ = integrate.quad(integrand, a_start, a_end, epsabs=1e-10, epsrel=1e-8, limit=200)
    prefactor = params.c_km_s / params.h0_km_s_mpc
    return prefactor * integral


def particle_horizon_mpc(a: float, params: HorizonParams) -> float:
    """Comoving particle horizon at scale factor a."""
    return comoving_distance_mpc(params.a_min, a, params)


def horizon_angle_rad(horizon_mpc: float, distance_to_lss_mpc: float) -> float:
    """Small-angle estimate for the causally connected CMB patch angle."""
    if distance_to_lss_mpc <= 0.0:
        raise ValueError("distance_to_lss_mpc must be positive.")
    return horizon_mpc / distance_to_lss_mpc


def causal_patch_count(theta_rad: float) -> float:
    """Approximate number of independent patches on the full sky: 4/theta^2."""
    if theta_rad <= 0.0:
        raise ValueError("theta_rad must be positive.")
    return 4.0 / (theta_rad * theta_rad)


def required_efolds(theta_initial: float, theta_target: float) -> float:
    """Minimal N such that theta_initial * exp(N) >= theta_target."""
    if not (theta_initial > 0.0 and theta_target > 0.0):
        raise ValueError("Angles must be positive.")
    if theta_initial >= theta_target:
        return 0.0
    return float(math.log(theta_target / theta_initial))


def build_efold_table(theta_initial: float, n_grid: np.ndarray) -> pd.DataFrame:
    """Evaluate effective causal angle and patch count for selected e-fold values."""
    rows: list[dict[str, float | bool]] = []
    for n_efolds in n_grid:
        theta_eff = float(theta_initial * np.exp(float(n_efolds)))
        theta_eff = min(theta_eff, math.pi)
        patches = causal_patch_count(theta_eff)
        rows.append(
            {
                "N_efolds": float(n_efolds),
                "theta_eff_deg": float(np.degrees(theta_eff)),
                "causal_patches": patches,
                "single_patch_sky": bool(patches <= 1.0),
            }
        )
    return pd.DataFrame(rows)


def run_demo() -> None:
    params = HorizonParams()
    a_dec = 1.0 / (1.0 + params.z_dec)

    chi_hor_dec = particle_horizon_mpc(a_dec, params)
    chi_to_lss = comoving_distance_mpc(a_dec, 1.0, params)
    theta_std = horizon_angle_rad(chi_hor_dec, chi_to_lss)
    patches_std = causal_patch_count(theta_std)

    n_area_criterion = required_efolds(theta_std, theta_target=2.0)
    n_antipodal_criterion = required_efolds(theta_std, theta_target=math.pi)

    n_grid = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    table = build_efold_table(theta_std, n_grid)

    print("=== Horizon Problem MVP ===")
    print(f"H0 = {params.h0_km_s_mpc:.2f} km/s/Mpc")
    print(f"Omega_m = {params.omega_m:.6f}, Omega_r = {params.omega_r:.6e}, Omega_Lambda = {omega_lambda(params):.6f}")
    print(f"z_dec = {params.z_dec:.1f}, a_dec = {a_dec:.3e}")
    print(f"Comoving particle horizon at decoupling = {chi_hor_dec:.3f} Mpc")
    print(f"Comoving distance to last-scattering surface = {chi_to_lss:.3f} Mpc")
    print(f"Standard-horizon angular scale at decoupling = {np.degrees(theta_std):.3f} deg")
    print(f"Estimated independent causal patches on CMB sky = {patches_std:.1f}")

    print("\nInflation e-fold lower bounds (geometric toy criteria):")
    print(f"N_min for area criterion (theta_eff >= 2 rad): {n_area_criterion:.3f}")
    print(f"N_min for antipodal criterion (theta_eff >= pi rad): {n_antipodal_criterion:.3f}")

    print("\nPatch count vs e-folds:")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3e}"))

    print("\nSensitivity to decoupling redshift:")
    for z_dec in [900.0, 1089.0, 1400.0]:
        p = HorizonParams(z_dec=z_dec)
        a_d = 1.0 / (1.0 + p.z_dec)
        chi_h = particle_horizon_mpc(a_d, p)
        chi_lss = comoving_distance_mpc(a_d, 1.0, p)
        theta = horizon_angle_rad(chi_h, chi_lss)
        n_req = required_efolds(theta, 2.0)
        print(
            f"z_dec={z_dec:>6.1f} -> theta={np.degrees(theta):.3f} deg, "
            f"patches={causal_patch_count(theta):.1f}, N_area={n_req:.3f}"
        )


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()

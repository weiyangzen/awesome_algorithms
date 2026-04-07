"""Minimal runnable MVP for Press-Schechter halo mass function.

This demo computes a Press-Schechter halo mass function from a simple linear
matter power spectrum model, normalized to a target sigma8.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.integrate import simpson
from scipy.special import erfc

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Cosmology:
    """Flat LCDM parameters for the MVP."""

    omega_m: float = 0.315
    omega_lambda: float = 0.685
    h: float = 0.674
    n_s: float = 0.965
    sigma8: float = 0.811
    delta_c: float = 1.686

    @property
    def rho_crit0(self) -> float:
        """Critical density today in Msun / Mpc^3."""
        return 2.775e11 * self.h**2

    @property
    def rho_m0(self) -> float:
        """Mean matter density today in Msun / Mpc^3."""
        return self.omega_m * self.rho_crit0


def e_z(z: float, cosmo: Cosmology) -> float:
    """Dimensionless Hubble parameter E(z)=H(z)/H0 for flat LCDM."""
    return float(np.sqrt(cosmo.omega_m * (1.0 + z) ** 3 + cosmo.omega_lambda))


def growth_factor_lcdm(z: float, cosmo: Cosmology) -> float:
    """Approximate linear growth factor D(z), normalized to D(0)=1.

    Uses the Carroll-Press-Turner fitting form for g(z), then D(z)=g(z)/(1+z).
    """

    def g_of_z(z_value: float) -> float:
        ez2 = e_z(z_value, cosmo) ** 2
        omega_m_z = cosmo.omega_m * (1.0 + z_value) ** 3 / ez2
        omega_l_z = cosmo.omega_lambda / ez2
        numerator = 5.0 * omega_m_z / 2.0
        denominator = (
            omega_m_z ** (4.0 / 7.0)
            - omega_l_z
            + (1.0 + omega_m_z / 2.0) * (1.0 + omega_l_z / 70.0)
        )
        return float(numerator / denominator)

    g0 = g_of_z(0.0)
    gz = g_of_z(z)
    return float(gz / (g0 * (1.0 + z)))


def transfer_bbks(k_mpc_inv: FloatArray, cosmo: Cosmology) -> FloatArray:
    """BBKS transfer function approximation.

    Input k in 1/Mpc. Internally converted to h/Mpc for the BBKS q variable.
    """
    k_hmpc = k_mpc_inv / cosmo.h
    gamma = cosmo.omega_m * cosmo.h
    q = np.maximum(k_hmpc / gamma, 1e-12)

    c0 = np.log(1.0 + 2.34 * q) / (2.34 * q)
    c1 = 1.0 + 3.89 * q + (16.1 * q) ** 2 + (5.46 * q) ** 3 + (6.71 * q) ** 4
    return c0 / np.power(c1, 0.25)


def power_spectrum_unnormalized(k_mpc_inv: FloatArray, cosmo: Cosmology) -> FloatArray:
    """Linear-theory-shaped spectrum with unit amplitude."""
    t = transfer_bbks(k_mpc_inv, cosmo)
    return np.power(k_mpc_inv, cosmo.n_s) * np.square(t)


def top_hat_window(x: FloatArray) -> FloatArray:
    """Fourier-space spherical top-hat window W(x)."""
    out = np.empty_like(x, dtype=np.float64)
    small = np.abs(x) < 1e-3

    xs = x[small]
    out[small] = 1.0 - xs**2 / 10.0 + xs**4 / 280.0

    xl = x[~small]
    out[~small] = 3.0 * (np.sin(xl) - xl * np.cos(xl)) / np.power(xl, 3)
    return out


def sigma_r(k_mpc_inv: FloatArray, pk: FloatArray, radii_mpc: FloatArray) -> FloatArray:
    """RMS density fluctuation sigma(R) from P(k)."""
    x = np.outer(radii_mpc, k_mpc_inv)
    w = top_hat_window(x)
    integrand = (np.square(k_mpc_inv)[None, :] * pk[None, :] * np.square(w)) / (
        2.0 * np.pi**2
    )
    sigma2 = simpson(integrand, x=k_mpc_inv, axis=1)
    return np.sqrt(np.maximum(sigma2, 0.0))


def mass_to_radius(mass_msun: FloatArray, cosmo: Cosmology) -> FloatArray:
    """Map halo mass M to top-hat radius R via M=(4/3)pi rho_m0 R^3."""
    return np.power(3.0 * mass_msun / (4.0 * np.pi * cosmo.rho_m0), 1.0 / 3.0)


def normalize_power_to_sigma8(
    k_mpc_inv: FloatArray, cosmo: Cosmology
) -> tuple[float, FloatArray, float]:
    """Set power spectrum amplitude so sigma(8/h Mpc)=sigma8 target."""
    pk_unit = power_spectrum_unnormalized(k_mpc_inv, cosmo)

    r8 = 8.0 / cosmo.h
    sigma8_unit = float(sigma_r(k_mpc_inv, pk_unit, np.array([r8], dtype=np.float64))[0])

    amplitude = (cosmo.sigma8 / sigma8_unit) ** 2
    pk = amplitude * pk_unit
    return amplitude, pk, sigma8_unit


def sigma_m(
    mass_msun: FloatArray,
    k_mpc_inv: FloatArray,
    pk: FloatArray,
    cosmo: Cosmology,
    z: float,
) -> FloatArray:
    """Mass variance sigma(M,z)=D(z)*sigma(M,0)."""
    radii = mass_to_radius(mass_msun, cosmo)
    sigma0 = sigma_r(k_mpc_inv, pk, radii)
    return growth_factor_lcdm(z, cosmo) * sigma0


def press_schechter_dndm(
    mass_msun: FloatArray,
    sigma_mass: FloatArray,
    cosmo: Cosmology,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Press-Schechter differential mass function dn/dM.

    Returns:
    - dn_dM: number density per mass [Mpc^-3 Msun^-1]
    - nu: peak-height parameter delta_c/sigma
    - dlnsigma_dlnm: slope term used in the model
    """
    sigma_safe = np.clip(sigma_mass, 1e-12, None)
    nu = cosmo.delta_c / sigma_safe

    lnm = np.log(mass_msun)
    lns = np.log(sigma_safe)
    dlnsigma_dlnm = np.gradient(lns, lnm)

    multiplicity = np.sqrt(2.0 / np.pi) * nu * np.exp(-0.5 * nu**2)
    dn_dM = (cosmo.rho_m0 / np.square(mass_msun)) * multiplicity * np.abs(
        dlnsigma_dlnm
    )
    return dn_dM, nu, dlnsigma_dlnm


def format_report_table(df: pd.DataFrame) -> str:
    """Pretty-print helper for deterministic terminal output."""
    return df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6e}",
        justify="center",
    )


def main() -> None:
    cosmo = Cosmology()

    # Integration and mass grids: small enough for fast MVP, dense enough to be stable.
    k = np.logspace(-4, 2, 4096, dtype=np.float64)  # 1/Mpc
    masses = np.logspace(10, 16, 120, dtype=np.float64)  # Msun

    amplitude, pk, sigma8_unit = normalize_power_to_sigma8(k, cosmo)

    sigma_z0 = sigma_m(masses, k, pk, cosmo, z=0.0)
    sigma_z1 = sigma_m(masses, k, pk, cosmo, z=1.0)

    dn_dM_z0, nu_z0, _ = press_schechter_dndm(masses, sigma_z0, cosmo)
    dn_dM_z1, nu_z1, _ = press_schechter_dndm(masses, sigma_z1, cosmo)

    collapsed_fraction_gt_m_z0 = erfc(nu_z0 / np.sqrt(2.0))

    # Finite-range mass conservation check: 
    # integral[(M/rho_m0) dn/dM dM] should approach 1 if mass range is wide enough.
    mass_fraction_integral = simpson((masses / cosmo.rho_m0) * dn_dM_z0, x=masses)

    sample_indices = np.linspace(0, masses.size - 1, 8, dtype=int)
    table = pd.DataFrame(
        {
            "M(Msun)": masses[sample_indices],
            "sigma(z=0)": sigma_z0[sample_indices],
            "sigma(z=1)": sigma_z1[sample_indices],
            "nu(z=0)": nu_z0[sample_indices],
            "nu(z=1)": nu_z1[sample_indices],
            "dn/dM(z=0)": dn_dM_z0[sample_indices],
            "dn/dM(z=1)": dn_dM_z1[sample_indices],
            "F(>M,z=0)": collapsed_fraction_gt_m_z0[sample_indices],
        }
    )

    print("Press-Schechter Theory MVP")
    print("=" * 72)
    print(
        "Cosmology: "
        f"Omega_m={cosmo.omega_m}, Omega_L={cosmo.omega_lambda}, "
        f"h={cosmo.h}, n_s={cosmo.n_s}, sigma8={cosmo.sigma8}"
    )
    print(f"delta_c = {cosmo.delta_c:.3f}")
    print(f"rho_m0 = {cosmo.rho_m0:.6e} Msun/Mpc^3")
    print(f"Power normalization amplitude A = {amplitude:.6e}")
    print(f"sigma8 (A=1 before normalization) = {sigma8_unit:.6e}")
    print(f"sigma8 (after normalization)      = {sigma_r(k, pk, np.array([8.0 / cosmo.h]))[0]:.6e}")
    print()
    print("Mass-function sample table:")
    print(format_report_table(table))
    print()
    print(
        "Finite-range mass-fraction check at z=0: "
        f"integral[(M/rho_m0) dn/dM dM] = {mass_fraction_integral:.6f}"
    )


if __name__ == "__main__":
    main()

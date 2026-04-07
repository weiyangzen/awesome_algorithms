"""Minimal runnable MVP for the cosmological Halo Model.

The script builds a nonlinear matter power spectrum at z=0 with:
1) a sigma8-normalized linear spectrum,
2) Sheth-Tormen halo abundance and halo bias,
3) NFW profile Fourier kernel,
4) 1-halo + 2-halo decomposition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.integrate import simpson
from scipy.special import sici

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Cosmology:
    """Flat LCDM parameter set for the MVP."""

    omega_m: float = 0.315
    omega_lambda: float = 0.685
    h: float = 0.674
    n_s: float = 0.965
    sigma8: float = 0.811
    delta_c: float = 1.686

    # Halo-structure hyperparameters.
    delta_vir_mean: float = 200.0
    c0: float = 9.0
    c_mass_pivot: float = 1e12  # Msun
    c_mass_slope: float = 0.1

    @property
    def rho_crit0(self) -> float:
        """Critical density in Msun/Mpc^3."""
        return 2.775e11 * self.h**2

    @property
    def rho_m0(self) -> float:
        """Mean matter density in Msun/Mpc^3."""
        return self.omega_m * self.rho_crit0


def e_z(z: float, cosmo: Cosmology) -> float:
    """Dimensionless Hubble factor E(z)=H(z)/H0."""
    return float(np.sqrt(cosmo.omega_m * (1.0 + z) ** 3 + cosmo.omega_lambda))


def growth_factor_lcdm(z: float, cosmo: Cosmology) -> float:
    """Carroll-Press-Turner approximation for linear growth D(z), D(0)=1."""

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
    """BBKS transfer function approximation."""
    k_hmpc = k_mpc_inv / cosmo.h
    gamma = cosmo.omega_m * cosmo.h
    q = np.maximum(k_hmpc / gamma, 1e-12)

    c0 = np.log(1.0 + 2.34 * q) / (2.34 * q)
    c1 = 1.0 + 3.89 * q + (16.1 * q) ** 2 + (5.46 * q) ** 3 + (6.71 * q) ** 4
    return c0 / np.power(c1, 0.25)


def linear_power_unnormalized(k_mpc_inv: FloatArray, cosmo: Cosmology) -> FloatArray:
    """Linear matter spectrum shape with unit amplitude."""
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
    """RMS fluctuation sigma(R) from P(k)."""
    x = np.outer(radii_mpc, k_mpc_inv)
    w = top_hat_window(x)
    integrand = (np.square(k_mpc_inv)[None, :] * pk[None, :] * np.square(w)) / (
        2.0 * np.pi**2
    )
    sigma2 = simpson(integrand, x=k_mpc_inv, axis=1)
    return np.sqrt(np.maximum(sigma2, 0.0))


def mass_to_top_hat_radius(mass_msun: FloatArray, cosmo: Cosmology) -> FloatArray:
    """Map mass to top-hat radius via M=(4/3)pi rho_m0 R^3."""
    return np.power(3.0 * mass_msun / (4.0 * np.pi * cosmo.rho_m0), 1.0 / 3.0)


def normalize_linear_power_to_sigma8(
    k_mpc_inv: FloatArray,
    cosmo: Cosmology,
) -> tuple[float, FloatArray, float]:
    """Normalize P(k) so that sigma(8/h Mpc)=sigma8 target."""
    pk_unit = linear_power_unnormalized(k_mpc_inv, cosmo)
    r8 = 8.0 / cosmo.h
    sigma8_unit = float(sigma_r(k_mpc_inv, pk_unit, np.array([r8], dtype=np.float64))[0])
    amplitude = (cosmo.sigma8 / sigma8_unit) ** 2
    return amplitude, amplitude * pk_unit, sigma8_unit


def sigma_m(
    mass_msun: FloatArray,
    k_mpc_inv: FloatArray,
    pk: FloatArray,
    cosmo: Cosmology,
    z: float,
) -> FloatArray:
    """Mass variance sigma(M,z)."""
    radii = mass_to_top_hat_radius(mass_msun, cosmo)
    sigma0 = sigma_r(k_mpc_inv, pk, radii)
    return growth_factor_lcdm(z, cosmo) * sigma0


def sheth_tormen_mass_function_and_bias(
    mass_msun: FloatArray,
    sigma_mass: FloatArray,
    cosmo: Cosmology,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Return dn/dM, halo bias, peak height nu and slope dlnsigma/dlnM."""
    # Standard ST parameters.
    a = 0.707
    p = 0.3
    a_norm = 0.3222

    sigma_safe = np.clip(sigma_mass, 1e-12, None)
    nu = cosmo.delta_c / sigma_safe

    dlnsigma_dlnm = np.gradient(np.log(sigma_safe), np.log(mass_msun))

    anu2 = a * np.square(nu)
    f_nu = (
        a_norm
        * np.sqrt(2.0 * a / np.pi)
        * nu
        * np.exp(-0.5 * anu2)
        * (1.0 + np.power(np.maximum(anu2, 1e-14), -p))
    )
    dn_dM = (cosmo.rho_m0 / np.square(mass_msun)) * f_nu * np.abs(dlnsigma_dlnm)

    bias = (
        1.0
        + (anu2 - 1.0) / cosmo.delta_c
        + (2.0 * p) / (cosmo.delta_c * (1.0 + np.power(np.maximum(anu2, 1e-14), p)))
    )
    return dn_dM, bias, nu, dlnsigma_dlnm


def virial_radius_mpc(mass_msun: FloatArray, cosmo: Cosmology) -> FloatArray:
    """Halo virial radius with Delta=200 wrt mean density."""
    return np.power(
        3.0 * mass_msun / (4.0 * np.pi * cosmo.delta_vir_mean * cosmo.rho_m0),
        1.0 / 3.0,
    )


def concentration_mz(mass_msun: FloatArray, z: float, cosmo: Cosmology) -> FloatArray:
    """Simple concentration-mass relation."""
    c = cosmo.c0 / (1.0 + z) * np.power(mass_msun / cosmo.c_mass_pivot, -cosmo.c_mass_slope)
    return np.clip(c, 2.0, 25.0)


def nfw_u_of_km(
    k_mpc_inv: FloatArray,
    mass_msun: FloatArray,
    z: float,
    cosmo: Cosmology,
) -> FloatArray:
    """Fourier transform of normalized NFW profile, u(k|M), shape [M, K]."""
    c = concentration_mz(mass_msun, z, cosmo)
    r_vir = virial_radius_mpc(mass_msun, cosmo)
    r_s = r_vir / c

    x = r_s[:, None] * k_mpc_inv[None, :]
    x_safe = np.maximum(x, 1e-8)
    x1_safe = np.maximum((1.0 + c)[:, None] * x, 1e-8)

    si_x, ci_x = sici(x_safe)
    si_x1, ci_x1 = sici(x1_safe)

    norm = np.log(1.0 + c) - c / (1.0 + c)
    term = (
        np.sin(x) * (si_x1 - si_x)
        + np.cos(x) * (ci_x1 - ci_x)
        - np.sin(c[:, None] * x) / ((1.0 + c)[:, None] * x_safe)
    )
    u = term / norm[:, None]
    return np.where(x < 1e-4, 1.0, u)


def halo_power_terms(
    masses: FloatArray,
    dn_dM: FloatArray,
    bias: FloatArray,
    u_km: FloatArray,
    cosmo: Cosmology,
) -> tuple[FloatArray, FloatArray]:
    """Compute P_1h(k) and the 2-halo prefactor I_2(k)."""
    mass_weight = masses / cosmo.rho_m0

    p1h_integrand = dn_dM[:, None] * np.square(mass_weight[:, None]) * np.square(u_km)
    p1h = simpson(p1h_integrand, x=masses, axis=0)

    i2_integrand = dn_dM[:, None] * bias[:, None] * mass_weight[:, None] * u_km
    i2 = simpson(i2_integrand, x=masses, axis=0)
    return p1h, i2


def renormalize_mass_and_bias_constraints(
    masses: FloatArray,
    dn_dM: FloatArray,
    bias: FloatArray,
    cosmo: Cosmology,
) -> tuple[FloatArray, FloatArray, float, float]:
    """Enforce finite-range mass and bias consistency constraints.

    For a truncated mass grid, raw integrals typically violate:
    - ∫ n(M) M/rho_m dM = 1
    - ∫ n(M) b(M) M/rho_m dM = 1
    We apply simple multiplicative corrections so the low-k limit is stable.
    """
    mass_weight = masses / cosmo.rho_m0
    mass_int_raw = float(simpson(dn_dM * mass_weight, x=masses))
    dn_norm = dn_dM / np.clip(mass_int_raw, 1e-12, None)

    bias_int_raw = float(simpson(dn_norm * bias * mass_weight, x=masses))
    bias_norm = bias / np.clip(bias_int_raw, 1e-12, None)
    return dn_norm, bias_norm, mass_int_raw, bias_int_raw


def format_report_table(df: pd.DataFrame) -> str:
    """Deterministic table formatting for terminal output."""
    return df.to_string(index=False, float_format=lambda x: f"{x:.6e}", justify="center")


def main() -> None:
    cosmo = Cosmology()
    z = 0.0

    # Grids for sigma(M) integration and halo integrals.
    k_sigma = np.logspace(-4, 2, 4096, dtype=np.float64)  # 1/Mpc
    masses = np.logspace(10, 16, 240, dtype=np.float64)  # Msun
    k_eval = np.logspace(-2.5, 1.2, 140, dtype=np.float64)  # 1/Mpc

    amplitude, pk_sigma, sigma8_unit = normalize_linear_power_to_sigma8(k_sigma, cosmo)
    sigma_mass = sigma_m(masses, k_sigma, pk_sigma, cosmo, z=z)

    dn_dM_raw, bias_raw, nu, _ = sheth_tormen_mass_function_and_bias(masses, sigma_mass, cosmo)
    dn_dM, bias, mass_int_raw, bias_int_raw = renormalize_mass_and_bias_constraints(
        masses, dn_dM_raw, bias_raw, cosmo
    )
    u_km = nfw_u_of_km(k_eval, masses, z=z, cosmo=cosmo)

    p1h, i2 = halo_power_terms(masses, dn_dM, bias, u_km, cosmo)
    p_lin = np.interp(np.log(k_eval), np.log(k_sigma), pk_sigma)
    p2h = np.square(i2) * p_lin
    p_tot = p1h + p2h
    delta2_tot = np.power(k_eval, 3) * p_tot / (2.0 * np.pi**2)

    # Consistency diagnostics over the finite mass range [1e10, 1e16] Msun.
    mass_fraction = simpson(dn_dM * masses / cosmo.rho_m0, x=masses)
    bias_weighted_mass_fraction = simpson(dn_dM * bias * masses / cosmo.rho_m0, x=masses)
    low_k_ratio = p_tot[0] / p_lin[0]

    sample_ids = np.linspace(0, k_eval.size - 1, 9, dtype=int)
    table = pd.DataFrame(
        {
            "k(1/Mpc)": k_eval[sample_ids],
            "P_lin": p_lin[sample_ids],
            "P_1h": p1h[sample_ids],
            "P_2h": p2h[sample_ids],
            "P_total": p_tot[sample_ids],
            "Delta2_total": delta2_tot[sample_ids],
        }
    )

    print("Halo Model MVP (z=0)")
    print("=" * 78)
    print(
        "Cosmology: "
        f"Omega_m={cosmo.omega_m}, Omega_L={cosmo.omega_lambda}, "
        f"h={cosmo.h}, n_s={cosmo.n_s}, sigma8={cosmo.sigma8}"
    )
    print(f"rho_m0 = {cosmo.rho_m0:.6e} Msun/Mpc^3")
    print(f"delta_c = {cosmo.delta_c:.3f}, Delta_vir = {cosmo.delta_vir_mean:.1f}")
    print(f"Power normalization amplitude A = {amplitude:.6e}")
    print(f"sigma8 (A=1 before normalization) = {sigma8_unit:.6e}")
    sigma8_after = sigma_r(k_sigma, pk_sigma, np.array([8.0 / cosmo.h], dtype=np.float64))[0]
    print(f"sigma8 (after normalization)      = {sigma8_after:.6e}")
    print()
    print("Finite-mass-range diagnostics:")
    print(f"Raw integral n(M) M/rho_m dM           = {mass_int_raw:.6f}")
    print(f"Raw integral n(M) b(M) M/rho_m dM      = {bias_int_raw:.6f}")
    print(f"Renorm integral n(M) M/rho_m dM        = {mass_fraction:.6f}")
    print(f"Renorm integral n(M) b(M) M/rho_m dM   = {bias_weighted_mass_fraction:.6f}")
    print(f"Low-k ratio P_total/P_lin (first point)= {low_k_ratio:.6f}")
    print(f"nu range: [{np.min(nu):.4f}, {np.max(nu):.4f}]")
    print()
    print("Power-spectrum sample table:")
    print(format_report_table(table))


if __name__ == "__main__":
    main()

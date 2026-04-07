"""Monopole problem MVP: Kibble production + inflationary dilution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize


@dataclass(frozen=True)
class MonopoleParams:
    """Physical and observational parameters for a minimal monopole model."""

    t_c_gev: float = 1.0e16  # GUT symmetry-breaking scale
    g_star: float = 100.0  # relativistic degrees of freedom
    kappa: float = 1.0  # monopoles per horizon volume (order-one Kibble factor)
    m_pl_gev: float = 1.22e19  # Planck mass
    s0_cm3: float = 2891.2  # present entropy density [cm^-3]
    c_cm_s: float = 2.99792458e10
    v_over_c: float = 1.0e-3  # halo-like monopole speed scale
    parker_flux_bound: float = 1.0e-16  # [cm^-2 s^-1 sr^-1]


def hubble_rate_radiation(t_gev: float, g_star: float, m_pl_gev: float) -> float:
    """H(T) in radiation domination with natural units (GeV)."""
    return 1.66 * np.sqrt(g_star) * t_gev**2 / m_pl_gev


def entropy_density_radiation(t_gev: float, g_star: float) -> float:
    """Entropy density s(T) in natural units (GeV^3)."""
    return (2.0 * np.pi**2 / 45.0) * g_star * t_gev**3


def initial_monopole_yield(params: MonopoleParams) -> float:
    """Y_M = n_M / s from one-per-horizon Kibble estimate at T_c."""
    h_c = hubble_rate_radiation(params.t_c_gev, params.g_star, params.m_pl_gev)
    n_m_c = params.kappa * h_c**3
    s_c = entropy_density_radiation(params.t_c_gev, params.g_star)
    return n_m_c / s_c


def present_flux_from_yield(y_m: float, params: MonopoleParams) -> float:
    """Convert yield to isotropic differential flux [cm^-2 s^-1 sr^-1]."""
    n0_cm3 = y_m * params.s0_cm3
    v_cm_s = params.v_over_c * params.c_cm_s
    return n0_cm3 * v_cm_s / (4.0 * np.pi)


def flux_after_inflation(flux0: float, n_efolds: float) -> float:
    """Inflationary dilution: number density ~ exp(-3N)."""
    return flux0 * np.exp(-3.0 * n_efolds)


def required_efolds_numeric(flux0: float, flux_bound: float) -> float:
    """Find N such that flux_after_inflation(flux0, N) = flux_bound."""
    if flux0 <= flux_bound:
        return 0.0

    def residual(n_efolds: float) -> float:
        return flux_after_inflation(flux0, n_efolds) - flux_bound

    n_guess = np.log(flux0 / flux_bound) / 3.0
    n_hi = max(1.0, n_guess + 5.0)
    return float(optimize.brentq(residual, a=0.0, b=n_hi, xtol=1e-12, rtol=1e-10, maxiter=200))


def required_efolds_analytic(flux0: float, flux_bound: float) -> float:
    """Closed-form check for the same root."""
    if flux0 <= flux_bound:
        return 0.0
    return float(np.log(flux0 / flux_bound) / 3.0)


def build_scan_table(params: MonopoleParams, n_grid: np.ndarray) -> tuple[pd.DataFrame, float, float]:
    """Create scan table over e-fold values."""
    y0 = initial_monopole_yield(params)
    flux0 = present_flux_from_yield(y0, params)

    rows: list[dict[str, float | bool]] = []
    for n_efolds in n_grid:
        flux_n = flux_after_inflation(flux0, float(n_efolds))
        rows.append(
            {
                "N_efolds": float(n_efolds),
                "flux_cm^-2_s^-1_sr^-1": flux_n,
                "passes_parker_bound": bool(flux_n <= params.parker_flux_bound),
            }
        )
    return pd.DataFrame(rows), y0, flux0


def run_demo() -> None:
    params = MonopoleParams()
    n_grid = np.array([0.0, 5.0, 10.0, 12.0, 14.0, 15.0, 20.0, 30.0, 60.0])

    table, y0, flux0 = build_scan_table(params, n_grid)
    n_req_num = required_efolds_numeric(flux0, params.parker_flux_bound)
    n_req_ana = required_efolds_analytic(flux0, params.parker_flux_bound)

    print("=== Monopole Problem MVP ===")
    print(f"GUT scale T_c = {params.t_c_gev:.3e} GeV")
    print(f"Kibble factor kappa = {params.kappa:.2f}")
    print(f"Initial yield Y_M(T_c) = {y0:.3e}")
    print(f"No-inflation flux F0 = {flux0:.3e} cm^-2 s^-1 sr^-1")
    print(f"Parker bound F_bound = {params.parker_flux_bound:.3e} cm^-2 s^-1 sr^-1")
    print(f"Required e-folds (numeric brentq) = {n_req_num:.6f}")
    print(f"Required e-folds (analytic)      = {n_req_ana:.6f}")
    print(f"Consistency |numeric-analytic|   = {abs(n_req_num - n_req_ana):.3e}")

    print("\nFlux scan over selected e-fold values:")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3e}"))

    n_ref = 60.0
    flux_ref = flux_after_inflation(flux0, n_ref)
    print(f"\nReference check at N={n_ref:.0f}: F(N)={flux_ref:.3e}")
    print(f"Pass bound at N={n_ref:.0f}: {flux_ref <= params.parker_flux_bound}")

    # Sensitivity to Kibble prefactor uncertainty.
    print("\nSensitivity to kappa (order-of-magnitude uncertainty):")
    for kappa in [0.1, 1.0, 10.0]:
        p = MonopoleParams(kappa=kappa)
        y_k = initial_monopole_yield(p)
        f_k = present_flux_from_yield(y_k, p)
        n_k = required_efolds_numeric(f_k, p.parker_flux_bound)
        print(f"kappa={kappa:>4.1f} -> Y0={y_k:.3e}, F0={f_k:.3e}, N_required={n_k:.4f}")


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()

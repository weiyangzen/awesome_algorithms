"""Flatness problem MVP: curvature-deviation growth vs inflationary suppression."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize


@dataclass(frozen=True)
class FlatnessParams:
    """Minimal parameter set for a pedagogical flatness-problem calculation."""

    t0_gev: float = 2.348e-13  # CMB temperature today in GeV
    t_reh_gev: float = 1.0e15  # reheating temperature scale in GeV
    g_s0: float = 3.91  # entropy dof today
    g_s_reh: float = 106.75  # entropy dof near high-T SM
    z_eq: float = 3400.0  # matter-radiation equality redshift
    omega_k_bound_today: float = 1.0e-2  # target bound for |Omega_k| today
    delta_pre_inflation: float = 1.0  # assumed O(1) pre-inflation |Omega-1|


def scale_factor_reheating(params: FlatnessParams) -> float:
    """Estimate a_reh from entropy conservation: aT g_s^{1/3} ~ const."""
    entropy_factor = (params.g_s0 / params.g_s_reh) ** (1.0 / 3.0)
    return (params.t0_gev / params.t_reh_gev) * entropy_factor


def scale_factor_equality(z_eq: float) -> float:
    """Convert equality redshift to scale factor with a0=1."""
    return 1.0 / (1.0 + z_eq)


def flatness_growth_factor_no_inflation(a_reh: float, a_eq: float) -> tuple[float, float, float]:
    """Growth factors for |Omega-1| from reheating to today.

    Radiation era: |Omega-1| ~ a^2
    Matter era:    |Omega-1| ~ a
    """
    if not (0.0 < a_reh < a_eq < 1.0):
        raise ValueError("Require 0 < a_reh < a_eq < 1 for consistent epoch ordering.")

    growth_rad = (a_eq / a_reh) ** 2
    growth_mat = 1.0 / a_eq
    growth_total = growth_rad * growth_mat
    return growth_rad, growth_mat, growth_total


def omega_deviation_today(delta_pre_inflation: float, growth_total: float, n_efolds: float) -> float:
    """Propagate |Omega-1| to today with inflationary suppression exp(-2N)."""
    return delta_pre_inflation * np.exp(-2.0 * n_efolds) * growth_total


def required_efolds_analytic(
    delta_pre_inflation: float,
    growth_total: float,
    omega_k_bound_today: float,
) -> float:
    """Closed-form minimal N from delta_pre*exp(-2N)*G <= bound."""
    raw = delta_pre_inflation * growth_total / omega_k_bound_today
    if raw <= 1.0:
        return 0.0
    return float(0.5 * np.log(raw))


def required_efolds_numeric(
    delta_pre_inflation: float,
    growth_total: float,
    omega_k_bound_today: float,
) -> float:
    """Numerically solve delta_pre*exp(-2N)*G - bound = 0 for N."""
    deviation_no_inflation = delta_pre_inflation * growth_total
    if deviation_no_inflation <= omega_k_bound_today:
        return 0.0

    def residual(n_efolds: float) -> float:
        return omega_deviation_today(delta_pre_inflation, growth_total, n_efolds) - omega_k_bound_today

    n_guess = required_efolds_analytic(delta_pre_inflation, growth_total, omega_k_bound_today)
    n_hi = max(1.0, n_guess + 5.0)
    return float(optimize.brentq(residual, a=0.0, b=n_hi, xtol=1e-12, rtol=1e-10, maxiter=200))


def build_scan_table(params: FlatnessParams, n_grid: np.ndarray) -> tuple[pd.DataFrame, float, float, float, float]:
    """Evaluate |Omega-1| today across selected inflation e-fold values."""
    a_reh = scale_factor_reheating(params)
    a_eq = scale_factor_equality(params.z_eq)
    growth_rad, growth_mat, growth_total = flatness_growth_factor_no_inflation(a_reh, a_eq)

    rows: list[dict[str, float | bool]] = []
    for n_efolds in n_grid:
        omega_today = omega_deviation_today(params.delta_pre_inflation, growth_total, float(n_efolds))
        rows.append(
            {
                "N_efolds": float(n_efolds),
                "omega_minus_1_today": omega_today,
                "passes_bound": bool(omega_today <= params.omega_k_bound_today),
            }
        )

    return pd.DataFrame(rows), a_reh, growth_rad, growth_mat, growth_total


def run_demo() -> None:
    params = FlatnessParams()
    n_grid = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 55.0, 60.0, 70.0])

    table, a_reh, growth_rad, growth_mat, growth_total = build_scan_table(params, n_grid)
    a_eq = scale_factor_equality(params.z_eq)

    omega_no_inflation = omega_deviation_today(params.delta_pre_inflation, growth_total, n_efolds=0.0)
    n_req_num = required_efolds_numeric(
        params.delta_pre_inflation,
        growth_total,
        params.omega_k_bound_today,
    )
    n_req_ana = required_efolds_analytic(
        params.delta_pre_inflation,
        growth_total,
        params.omega_k_bound_today,
    )

    print("=== Flatness Problem MVP ===")
    print(f"T_reh = {params.t_reh_gev:.3e} GeV")
    print(f"a_reh = {a_reh:.3e}")
    print(f"z_eq = {params.z_eq:.1f} -> a_eq = {a_eq:.3e}")
    print(f"Growth in radiation era = {growth_rad:.3e}")
    print(f"Growth in matter era    = {growth_mat:.3e}")
    print(f"Total no-inflation growth G = {growth_total:.3e}")
    print(f"Assumed pre-inflation |Omega-1| = {params.delta_pre_inflation:.3e}")
    print(f"Predicted today without inflation = {omega_no_inflation:.3e}")
    print(f"Target bound today = {params.omega_k_bound_today:.3e}")
    print(f"Required e-folds (numeric brentq) = {n_req_num:.6f}")
    print(f"Required e-folds (analytic)      = {n_req_ana:.6f}")
    print(f"Consistency |numeric-analytic|   = {abs(n_req_num - n_req_ana):.3e}")

    print("\nOmega deviation scan over selected e-fold values:")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3e}"))

    print("\nSensitivity to reheating temperature:")
    for t_reh in [1.0e9, 1.0e12, 1.0e15]:
        p = FlatnessParams(t_reh_gev=t_reh)
        a_reh_p = scale_factor_reheating(p)
        a_eq_p = scale_factor_equality(p.z_eq)
        _, _, g_total_p = flatness_growth_factor_no_inflation(a_reh_p, a_eq_p)
        n_p = required_efolds_numeric(p.delta_pre_inflation, g_total_p, p.omega_k_bound_today)
        print(f"T_reh={t_reh:>8.1e} GeV -> a_reh={a_reh_p:.3e}, G={g_total_p:.3e}, N_required={n_p:.3f}")

    print("\nSensitivity to assumed pre-inflation deviation:")
    for delta0 in [0.1, 1.0, 10.0]:
        p = FlatnessParams(delta_pre_inflation=delta0)
        a_reh_p = scale_factor_reheating(p)
        a_eq_p = scale_factor_equality(p.z_eq)
        _, _, g_total_p = flatness_growth_factor_no_inflation(a_reh_p, a_eq_p)
        n_p = required_efolds_numeric(p.delta_pre_inflation, g_total_p, p.omega_k_bound_today)
        print(f"|Omega-1|_pre={delta0:>4.1f} -> G={g_total_p:.3e}, N_required={n_p:.3f}")


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()

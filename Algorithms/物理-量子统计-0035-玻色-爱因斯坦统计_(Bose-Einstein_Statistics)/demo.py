"""Minimal runnable MVP for Bose-Einstein statistics.

The script computes the chemical potential and condensate fraction of an
ideal 3D Bose gas (in normalized units) across a set of temperatures.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import integrate, optimize, special


@dataclass(frozen=True)
class BEConfig:
    """Configuration for the Bose-Einstein statistics MVP."""

    tc: float = 1.0
    total_density: float = 1.0
    temperatures: tuple[float, ...] = (0.40, 0.70, 1.00, 1.30, 1.80)
    energy_levels: tuple[float, ...] = (0.02, 0.10, 0.30, 0.80, 1.50, 3.00)
    t_upper: float = 14.0
    quad_epsabs: float = 1e-9
    quad_epsrel: float = 1e-8
    alpha_upper_init: float = 60.0


def density_prefactor_for_tc(tc: float, total_density: float) -> float:
    """Return DOS prefactor A so that N_ex(Tc, mu=0)=total_density.

    For a 3D ideal Bose gas with g(epsilon)=A*sqrt(epsilon),
    N_ex(T,mu) = A*T^(3/2)*Gamma(3/2)*Li_{3/2}(exp(mu/T)).
    At the critical point mu=0, this becomes
    N_ex(Tc,0) = A*Tc^(3/2)*Gamma(3/2)*zeta(3/2).
    """
    gamma_3_over_2 = float(special.gamma(1.5))
    zeta_3_over_2 = float(special.zeta(1.5, 1.0))
    return total_density / (tc**1.5 * gamma_3_over_2 * zeta_3_over_2)


def _bose_integrand_t(t: float, alpha: float) -> float:
    """Integrand after substitution epsilon = t^2.

    I(alpha) = int_0^inf sqrt(x)/(exp(x+alpha)-1) dx
             = int_0^inf 2*t^2/(exp(t^2+alpha)-1) dt
    where alpha = -mu/T >= 0 for bosons.
    """
    u = t * t + alpha
    if u < 1e-12:
        # limit_{u->0} 2*t^2/expm1(u) with alpha=0 and u=t^2 is 2.
        return 2.0
    return 2.0 * t * t / np.expm1(u)


def bose_integral(alpha: float, cfg: BEConfig) -> float:
    """Numerically evaluate the Bose integral I(alpha)."""
    value, _ = integrate.quad(
        _bose_integrand_t,
        0.0,
        cfg.t_upper,
        args=(alpha,),
        epsabs=cfg.quad_epsabs,
        epsrel=cfg.quad_epsrel,
        limit=400,
    )
    return float(value)


def excited_density(T: float, mu: float, prefactor_A: float, cfg: BEConfig) -> float:
    """Compute excited-state density N_ex(T, mu)."""
    alpha = -mu / T
    if alpha < -1e-12:
        raise ValueError("For bosons, alpha=-mu/T should be non-negative.")
    alpha = max(alpha, 0.0)
    return float(prefactor_A * (T**1.5) * bose_integral(alpha, cfg))


def solve_mu_above_tc(T: float, prefactor_A: float, cfg: BEConfig) -> float:
    """Solve mu(T)<0 for T>Tc from N_ex(T, mu)=N_total."""

    def residual(alpha: float) -> float:
        return prefactor_A * (T**1.5) * bose_integral(alpha, cfg) - cfg.total_density

    low = 1e-12
    high = cfg.alpha_upper_init
    f_low = residual(low)
    if f_low < 0.0:
        raise RuntimeError("Residual at alpha~0 is negative; expected T>Tc branch.")

    f_high = residual(high)
    while f_high > 0.0:
        high *= 2.0
        if high > 1e5:
            raise RuntimeError("Failed to bracket chemical potential root.")
        f_high = residual(high)

    alpha_star = optimize.brentq(residual, low, high, xtol=1e-12, rtol=1e-10, maxiter=200)
    return float(-alpha_star * T)


def analyze_temperatures(cfg: BEConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute thermodynamic summary and occupation samples."""
    prefactor_A = density_prefactor_for_tc(cfg.tc, cfg.total_density)
    energies = np.array(cfg.energy_levels, dtype=float)

    summary_records: list[dict[str, float | bool]] = []
    occupancy_records: list[dict[str, float]] = []

    for T in cfg.temperatures:
        if T <= cfg.tc + 1e-12:
            mu = 0.0
        else:
            mu = solve_mu_above_tc(T, prefactor_A, cfg)

        n_ex = excited_density(T, mu, prefactor_A, cfg)
        n0 = max(cfg.total_density - n_ex, 0.0)
        cond_frac = n0 / cfg.total_density
        theory_cond_frac = max(1.0 - (T / cfg.tc) ** 1.5, 0.0)

        summary_records.append(
            {
                "T": float(T),
                "mu": float(mu),
                "N_ex": float(n_ex),
                "N0": float(n0),
                "cond_frac": float(cond_frac),
                "theory_cond_frac": float(theory_cond_frac),
                "mu_over_T": float(mu / T),
                "is_condensed": bool(cond_frac > 1e-8),
            }
        )

        # Occupation numbers for representative energy levels.
        occupations = 1.0 / np.expm1((energies - mu) / T)
        occ_row: dict[str, float] = {"T": float(T), "mu": float(mu)}
        for e, occ in zip(energies, occupations):
            occ_row[f"n(e={e:.2f})"] = float(occ)
        occupancy_records.append(occ_row)

    summary_df = pd.DataFrame.from_records(summary_records)
    occupancy_df = pd.DataFrame.from_records(occupancy_records)
    return summary_df, occupancy_df


def run_consistency_checks(summary_df: pd.DataFrame, cfg: BEConfig) -> None:
    """Basic physical sanity checks for the generated results."""
    below_tc = summary_df.loc[summary_df["T"] < cfg.tc - 1e-12].sort_values("T")
    if len(below_tc) >= 2:
        cond = below_tc["cond_frac"].to_numpy()
        # As temperature rises below Tc, condensate fraction should not increase.
        assert np.all(np.diff(cond) <= 5e-4), "Condensate fraction should decrease below Tc."

        theory = below_tc["theory_cond_frac"].to_numpy()
        max_err = float(np.max(np.abs(cond - theory)))
        assert max_err < 8e-3, f"Condensate fraction deviates too much from theory: {max_err:.3e}"

    above_tc = summary_df.loc[summary_df["T"] > cfg.tc + 1e-12].sort_values("T")
    if not above_tc.empty:
        mu = above_tc["mu"].to_numpy()
        assert np.all(mu < 0.0), "mu should be negative above Tc."
        assert float(above_tc["cond_frac"].max()) < 5e-3, "Condensate should vanish above Tc."
        # For increasing T above Tc, mu typically becomes more negative.
        assert np.all(np.diff(mu) < 0.0), "mu(T) should decrease (more negative) with T above Tc."


def main() -> None:
    cfg = BEConfig()
    summary_df, occupancy_df = analyze_temperatures(cfg)
    run_consistency_checks(summary_df, cfg)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)

    print("=== Bose-Einstein Statistics MVP ===")
    print("Normalized units: k_B=1, Tc=1, total density=1")
    print()

    print("[Thermodynamic summary]")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x: .6f}"))
    print()

    print("[Sample occupation numbers]")
    print(occupancy_df.to_string(index=False, float_format=lambda x: f"{x: .6f}"))


if __name__ == "__main__":
    main()

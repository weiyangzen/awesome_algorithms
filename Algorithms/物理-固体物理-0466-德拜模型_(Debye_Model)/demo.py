"""Minimal runnable MVP for Debye Model in solid-state physics.

This script computes the molar constant-volume heat capacity C_V(T)
using the Debye model numerical integral:

    C_V = 9 R (T/theta_D)^3 * integral_0^{theta_D/T}
          [x^4 e^x / (e^x - 1)^2] dx

The implementation avoids black-box solvers and uses explicit Simpson
integration plus two sanity checks:
- Low-temperature law: C_V ~ (12*pi^4/5) R (T/theta_D)^3
- High-temperature law: C_V -> 3R (Dulong-Petit)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

R_GAS = 8.31446261815324  # J/(mol*K)


@dataclass(frozen=True)
class DebyeConfig:
    theta_debye: float = 343.0  # K, representative for Cu-like scale
    t_min: float = 2.0
    t_max: float = 2200.0
    n_temps: int = 120
    simpson_intervals: int = 4000
    integration_cutoff: float = 80.0


def simpson_integral(y: np.ndarray, a: float, b: float) -> float:
    """Composite Simpson integration on an evenly spaced grid.

    Parameters
    ----------
    y : np.ndarray
        Function values on n+1 points where n must be even.
    a, b : float
        Integration range.
    """
    n = y.size - 1
    if n <= 0 or n % 2 != 0:
        raise ValueError("Simpson rule needs an even number of intervals.")
    h = (b - a) / n
    odd_sum = np.sum(y[1:-1:2])
    even_sum = np.sum(y[2:-1:2])
    return float((h / 3.0) * (y[0] + y[-1] + 4.0 * odd_sum + 2.0 * even_sum))


def debye_cv_integrand(x: np.ndarray) -> np.ndarray:
    """Dimensionless integrand for Debye heat capacity.

    f(x) = x^4 * exp(x) / (exp(x)-1)^2
         = x^4 * exp(-x) / (1-exp(-x))^2
    """
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)

    # Series-stable near x=0: f(x) ~ x^2
    small = x < 1.0e-6
    out[small] = x[small] * x[small]

    if np.any(~small):
        xs = x[~small]
        exp_minus = np.exp(-xs)
        denom = -np.expm1(-xs)  # 1 - exp(-x), numerically stable for small x
        out[~small] = (xs**4) * exp_minus / (denom * denom)

    return out


def debye_heat_capacity_molar(
    temperatures: np.ndarray,
    theta_debye: float,
    simpson_intervals: int,
    integration_cutoff: float,
) -> np.ndarray:
    """Compute Debye molar C_V for each temperature (J/mol/K)."""
    temps = np.asarray(temperatures, dtype=float)
    if np.any(temps <= 0.0):
        raise ValueError("All temperatures must be positive.")

    cv = np.empty_like(temps)

    n = simpson_intervals
    if n % 2 != 0:
        raise ValueError("simpson_intervals must be even.")

    for i, t in enumerate(temps):
        upper_exact = theta_debye / t
        upper = min(upper_exact, integration_cutoff)

        x = np.linspace(0.0, upper, n + 1)
        y = debye_cv_integrand(x)
        integral_value = simpson_integral(y, 0.0, upper)

        cv[i] = 9.0 * R_GAS * (t / theta_debye) ** 3 * integral_value

    return cv


def build_result_table(cfg: DebyeConfig) -> pd.DataFrame:
    temperatures = np.geomspace(cfg.t_min, cfg.t_max, cfg.n_temps)
    cv = debye_heat_capacity_molar(
        temperatures=temperatures,
        theta_debye=cfg.theta_debye,
        simpson_intervals=cfg.simpson_intervals,
        integration_cutoff=cfg.integration_cutoff,
    )

    df = pd.DataFrame(
        {
            "T_K": temperatures,
            "Cv_J_per_molK": cv,
            "Cv_over_3R": cv / (3.0 * R_GAS),
            "Cv_over_T3": cv / (temperatures**3),
        }
    )
    return df


def run_sanity_checks(cfg: DebyeConfig) -> dict[str, float]:
    # Low-T T^3 law coefficient
    low_t = 0.03 * cfg.theta_debye
    cv_low = debye_heat_capacity_molar(
        temperatures=np.array([low_t]),
        theta_debye=cfg.theta_debye,
        simpson_intervals=cfg.simpson_intervals,
        integration_cutoff=cfg.integration_cutoff,
    )[0]
    cv_low_asym = (12.0 * math.pi**4 / 5.0) * R_GAS * (low_t / cfg.theta_debye) ** 3
    low_rel_err = abs(cv_low - cv_low_asym) / cv_low_asym

    # High-T Dulong-Petit limit
    high_t = 8.0 * cfg.theta_debye
    cv_high = debye_heat_capacity_molar(
        temperatures=np.array([high_t]),
        theta_debye=cfg.theta_debye,
        simpson_intervals=cfg.simpson_intervals,
        integration_cutoff=cfg.integration_cutoff,
    )[0]
    cv_dp = 3.0 * R_GAS
    high_rel_err = abs(cv_high - cv_dp) / cv_dp

    # Non-negativity over representative range
    probe_t = np.geomspace(cfg.t_min, cfg.t_max, 32)
    probe_cv = debye_heat_capacity_molar(
        temperatures=probe_t,
        theta_debye=cfg.theta_debye,
        simpson_intervals=cfg.simpson_intervals,
        integration_cutoff=cfg.integration_cutoff,
    )
    min_cv = float(np.min(probe_cv))

    # Assert expected physical behaviors
    assert low_rel_err < 0.05, f"Low-T asymptotic mismatch too large: {low_rel_err:.4f}"
    assert high_rel_err < 0.02, f"High-T Dulong-Petit mismatch too large: {high_rel_err:.4f}"
    assert min_cv > 0.0, "Heat capacity should stay positive in the tested range."

    return {
        "low_t_K": float(low_t),
        "low_rel_err": float(low_rel_err),
        "high_t_K": float(high_t),
        "high_rel_err": float(high_rel_err),
        "min_cv": min_cv,
    }


def main() -> None:
    cfg = DebyeConfig()
    df = build_result_table(cfg)
    checks = run_sanity_checks(cfg)

    print("=== Debye Model (Molar Heat Capacity) ===")
    print(f"theta_D = {cfg.theta_debye:.2f} K")
    print("\nSelected sample points:")

    sample_idx = np.linspace(0, len(df) - 1, 8, dtype=int)
    sample_df = df.iloc[sample_idx].copy()
    with pd.option_context("display.max_rows", 20, "display.width", 120):
        print(sample_df.to_string(index=False, float_format=lambda x: f"{x:10.6f}"))

    print("\nSanity checks:")
    print(
        "- Low-T T^3 law at T={:.3f} K: relative error = {:.4%}".format(
            checks["low_t_K"], checks["low_rel_err"]
        )
    )
    print(
        "- High-T 3R limit at T={:.3f} K: relative error = {:.4%}".format(
            checks["high_t_K"], checks["high_rel_err"]
        )
    )
    print("- Minimum C_V in probe range: {:.6f} J/(mol*K)".format(checks["min_cv"]))


if __name__ == "__main__":
    main()

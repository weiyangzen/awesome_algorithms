"""Minimal runnable MVP for Einstein Model in solid-state physics.

This script computes temperature-dependent molar quantities for the
Einstein crystal model (single characteristic vibration frequency):

1) Thermal internal energy (zero-point term excluded):
   U_th(T) = 3 R Theta_E / (exp(Theta_E/T) - 1)

2) Constant-volume molar heat capacity:
   C_V(T) = 3 R * x^2 * exp(x) / (exp(x) - 1)^2
   where x = Theta_E / T.

The implementation is explicit and non-black-box:
- stable evaluation of Bose occupancy and C_V kernel
- temperature sweep table
- physics sanity checks (low-T suppression, high-T 3R limit, monotonicity)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

R_GAS = 8.31446261815324  # J/(mol*K)


@dataclass(frozen=True)
class EinsteinConfig:
    theta_einstein: float = 220.0  # K, representative scale
    t_min: float = 2.0
    t_max: float = 2500.0
    n_temps: int = 140


def bose_occupancy(x: np.ndarray) -> np.ndarray:
    """Return n_B(x)=1/(exp(x)-1) with numerically stable branches."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)

    if np.any(x <= 0.0):
        raise ValueError("Dimensionless x must be positive.")

    small = x < 1.0e-4
    if np.any(small):
        xs = x[small]
        # Series: 1/x - 1/2 + x/12 - x^3/720
        out[small] = 1.0 / xs - 0.5 + xs / 12.0 - (xs**3) / 720.0

    if np.any(~small):
        xb = x[~small]
        exp_minus = np.exp(-xb)
        denom = -np.expm1(-xb)  # 1 - exp(-x), stable for x->0+
        out[~small] = exp_minus / denom

    return out


def cv_ratio_over_3r(x: np.ndarray) -> np.ndarray:
    """Return C_V/(3R) for Einstein model at x=Theta_E/T."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)

    if np.any(x <= 0.0):
        raise ValueError("Dimensionless x must be positive.")

    small = x < 1.0e-4
    if np.any(small):
        xs = x[small]
        # Series around x=0: 1 - x^2/12 + x^4/240
        out[small] = 1.0 - (xs**2) / 12.0 + (xs**4) / 240.0

    if np.any(~small):
        xb = x[~small]
        exp_minus = np.exp(-xb)
        denom = -np.expm1(-xb)
        out[~small] = (xb**2) * exp_minus / (denom * denom)

    return out


def einstein_heat_capacity_molar(temperatures: np.ndarray, theta_einstein: float) -> np.ndarray:
    """Compute Einstein-model molar C_V(T) in J/(mol*K)."""
    temps = np.asarray(temperatures, dtype=float)
    if np.any(temps <= 0.0):
        raise ValueError("All temperatures must be positive.")
    if theta_einstein <= 0.0:
        raise ValueError("theta_einstein must be positive.")

    x = theta_einstein / temps
    return 3.0 * R_GAS * cv_ratio_over_3r(x)


def einstein_internal_energy_thermal_molar(
    temperatures: np.ndarray, theta_einstein: float
) -> np.ndarray:
    """Compute thermal part of internal energy U_th(T) in J/mol.

    Zero-point term (3/2 R Theta_E) is intentionally excluded.
    """
    temps = np.asarray(temperatures, dtype=float)
    if np.any(temps <= 0.0):
        raise ValueError("All temperatures must be positive.")
    if theta_einstein <= 0.0:
        raise ValueError("theta_einstein must be positive.")

    x = theta_einstein / temps
    n_b = bose_occupancy(x)
    return 3.0 * R_GAS * theta_einstein * n_b


def build_result_table(cfg: EinsteinConfig) -> pd.DataFrame:
    temperatures = np.geomspace(cfg.t_min, cfg.t_max, cfg.n_temps)
    cv = einstein_heat_capacity_molar(temperatures, cfg.theta_einstein)
    u_th = einstein_internal_energy_thermal_molar(temperatures, cfg.theta_einstein)

    df = pd.DataFrame(
        {
            "T_K": temperatures,
            "Cv_J_per_molK": cv,
            "Cv_over_3R": cv / (3.0 * R_GAS),
            "U_th_J_per_mol": u_th,
            "Cv_over_T3": cv / (temperatures**3),
        }
    )
    return df


def run_sanity_checks(cfg: EinsteinConfig) -> dict[str, float]:
    # Low-T asymptotic: C_V ~ 3R * x^2 * exp(-x), x=Theta_E/T
    low_t = max(cfg.t_min, 0.08 * cfg.theta_einstein)
    x_low = cfg.theta_einstein / low_t
    cv_low = einstein_heat_capacity_molar(np.array([low_t]), cfg.theta_einstein)[0]
    cv_low_asym = 3.0 * R_GAS * (x_low**2) * math.exp(-x_low)
    low_rel_err = abs(cv_low - cv_low_asym) / cv_low_asym

    # High-T Dulong-Petit: C_V -> 3R
    high_t = min(cfg.t_max, 10.0 * cfg.theta_einstein)
    cv_high = einstein_heat_capacity_molar(np.array([high_t]), cfg.theta_einstein)[0]
    cv_dp = 3.0 * R_GAS
    high_rel_err = abs(cv_high - cv_dp) / cv_dp

    # Monotonic increase with T for Einstein C_V
    probe_t = np.geomspace(max(cfg.t_min, 0.03 * cfg.theta_einstein), cfg.t_max, 220)
    probe_cv = einstein_heat_capacity_molar(probe_t, cfg.theta_einstein)
    diffs = np.diff(probe_cv)
    monotonic_violations = int(np.sum(diffs < -1.0e-11))

    # Thermal internal energy should be close to 0 at very low T.
    ultra_low_t = max(cfg.t_min, 0.02 * cfg.theta_einstein)
    u_ultra_low = einstein_internal_energy_thermal_molar(
        np.array([ultra_low_t]), cfg.theta_einstein
    )[0]

    assert low_rel_err < 0.08, f"Low-T asymptotic mismatch too large: {low_rel_err:.4f}"
    assert high_rel_err < 0.012, f"High-T 3R mismatch too large: {high_rel_err:.4f}"
    assert monotonic_violations == 0, "C_V(T) should be monotonic non-decreasing."
    assert u_ultra_low < 0.2, "U_th at ultra-low T should stay near 0."

    return {
        "low_t_K": float(low_t),
        "low_rel_err": float(low_rel_err),
        "high_t_K": float(high_t),
        "high_rel_err": float(high_rel_err),
        "monotonic_violations": float(monotonic_violations),
        "ultra_low_t_K": float(ultra_low_t),
        "u_ultra_low": float(u_ultra_low),
    }


def main() -> None:
    cfg = EinsteinConfig()
    df = build_result_table(cfg)
    checks = run_sanity_checks(cfg)

    print("=== Einstein Model (Molar Lattice Heat Capacity) ===")
    print(f"Theta_E = {cfg.theta_einstein:.2f} K")
    print("\nSelected sample points:")

    sample_idx = np.linspace(0, len(df) - 1, 8, dtype=int)
    sample_df = df.iloc[sample_idx].copy()
    with pd.option_context("display.max_rows", 20, "display.width", 150):
        print(sample_df.to_string(index=False, float_format=lambda x: f"{x:12.6f}"))

    print("\nSanity checks:")
    print(
        "- Low-T asymptotic at T={:.3f} K: relative error = {:.4%}".format(
            checks["low_t_K"], checks["low_rel_err"]
        )
    )
    print(
        "- High-T 3R limit at T={:.3f} K: relative error = {:.4%}".format(
            checks["high_t_K"], checks["high_rel_err"]
        )
    )
    print("- Monotonic violations in probe grid: {:.0f}".format(checks["monotonic_violations"]))
    print(
        "- U_th at ultra-low T={:.3f} K: {:.6f} J/mol".format(
            checks["ultra_low_t_K"], checks["u_ultra_low"]
        )
    )


if __name__ == "__main__":
    main()

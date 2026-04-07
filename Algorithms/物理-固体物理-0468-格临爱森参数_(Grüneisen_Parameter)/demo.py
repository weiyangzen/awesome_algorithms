"""Minimal runnable MVP for Gruneisen parameter in solid-state physics.

This script demonstrates two equivalent ways to obtain the Gruneisen parameter:
1) Mode definition (finite-difference form):
   gamma_i = -(V / theta_i) * (d theta_i / dV)
2) Thermodynamic identity:
   gamma(T) = alpha(T) * K_T * V / C_V(T)

The implementation is explicit (no black-box material package):
- synthetic volume-dependent mode temperatures theta_i(V),
- finite-difference gamma_i estimation,
- heat-capacity-weighted gamma(T),
- inversion check through gamma = alpha K_T V / C_V.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

R_GAS = 8.31446261815324  # J/(mol*K)


@dataclass(frozen=True)
class GruneisenConfig:
    """Physical and numerical configuration for the MVP."""

    # Molar reference volume and isothermal bulk modulus.
    v0_m3_per_mol: float = 1.00e-5
    bulk_modulus_pa: float = 1.50e11

    # Finite-difference volume step: V+/- = V0 * (1 +/- finite_diff_fraction)
    finite_diff_fraction: float = 1.0e-2

    # Three representative phonon-mode Einstein temperatures at V0.
    theta_modes_k: tuple[float, ...] = (120.0, 260.0, 540.0)

    # Ground-truth mode Gruneisen parameters used to generate theta(V).
    gamma_modes_true: tuple[float, ...] = (1.10, 1.55, 2.05)

    # Degeneracy per mode; sum=3 gives C_V -> 3R at high temperature.
    degeneracies: tuple[float, ...] = (1.0, 1.0, 1.0)

    # Temperature sweep.
    t_min: float = 5.0
    t_max: float = 1200.0
    n_temps: int = 140


def unpack_mode_arrays(cfg: GruneisenConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = np.asarray(cfg.theta_modes_k, dtype=float)
    gamma = np.asarray(cfg.gamma_modes_true, dtype=float)
    degeneracy = np.asarray(cfg.degeneracies, dtype=float)

    if theta.ndim != 1 or theta.size == 0:
        raise ValueError("theta_modes_k must be a non-empty 1D sequence")
    if gamma.shape != theta.shape or degeneracy.shape != theta.shape:
        raise ValueError("theta_modes_k, gamma_modes_true, degeneracies must have equal length")
    if np.any(theta <= 0.0):
        raise ValueError("all theta_modes_k must be positive")
    if np.any(gamma <= 0.0):
        raise ValueError("all gamma_modes_true must be positive")
    if np.any(degeneracy <= 0.0):
        raise ValueError("all degeneracies must be positive")

    return theta, gamma, degeneracy


def validate_config(cfg: GruneisenConfig) -> None:
    if cfg.v0_m3_per_mol <= 0.0:
        raise ValueError("v0_m3_per_mol must be positive")
    if cfg.bulk_modulus_pa <= 0.0:
        raise ValueError("bulk_modulus_pa must be positive")
    if not (0.0 < cfg.finite_diff_fraction < 0.2):
        raise ValueError("finite_diff_fraction should be in (0, 0.2)")
    if cfg.t_min <= 0.0 or cfg.t_max <= cfg.t_min:
        raise ValueError("temperature range must satisfy 0 < t_min < t_max")
    if cfg.n_temps < 16:
        raise ValueError("n_temps must be at least 16")

    unpack_mode_arrays(cfg)


def cv_ratio_over_r(x: np.ndarray) -> np.ndarray:
    """Stable evaluation of Einstein mode C_V / R at x=theta/T.

    Formula:
        C_V_mode / R = x^2 * exp(x)/(exp(x)-1)^2
                       = x^2 * exp(-x)/(1-exp(-x))^2
    """
    x = np.asarray(x, dtype=float)
    if np.any(x <= 0.0):
        raise ValueError("x must be positive")

    out = np.empty_like(x)
    small = x < 1.0e-4

    if np.any(small):
        xs = x[small]
        # Series around x=0: 1 - x^2/12 + x^4/240
        out[small] = 1.0 - (xs**2) / 12.0 + (xs**4) / 240.0

    if np.any(~small):
        xb = x[~small]
        exp_minus = np.exp(-xb)
        denom = -np.expm1(-xb)  # 1 - exp(-x), stable for x->0+
        out[~small] = (xb**2) * exp_minus / (denom * denom)

    return out


def einstein_mode_cv_molar(temperatures: np.ndarray, theta_mode: float, degeneracy: float) -> np.ndarray:
    """Mode heat capacity in J/(mol*K) using Einstein kernel."""
    temps = np.asarray(temperatures, dtype=float)
    if np.any(temps <= 0.0):
        raise ValueError("temperatures must be positive")
    if theta_mode <= 0.0 or degeneracy <= 0.0:
        raise ValueError("theta_mode and degeneracy must be positive")

    x = theta_mode / temps
    return degeneracy * R_GAS * cv_ratio_over_r(x)


def theta_at_volume(theta0: np.ndarray, gamma: np.ndarray, volume: float, v0: float) -> np.ndarray:
    """Return theta_i(V) = theta_i(V0) * (V/V0)^(-gamma_i)."""
    return theta0 * (volume / v0) ** (-gamma)


def estimate_mode_gamma_fd(
    theta_minus: np.ndarray,
    theta_zero: np.ndarray,
    theta_plus: np.ndarray,
    v_minus: float,
    v_zero: float,
    v_plus: float,
) -> np.ndarray:
    """Estimate gamma_i via central finite difference at V0."""
    dtheta_dv = (theta_plus - theta_minus) / (v_plus - v_minus)
    return -(v_zero / theta_zero) * dtheta_dv


def build_mode_summary_table(
    cfg: GruneisenConfig,
    theta0: np.ndarray,
    gamma_true: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    v0 = cfg.v0_m3_per_mol
    dv_frac = cfg.finite_diff_fraction

    v_minus = v0 * (1.0 - dv_frac)
    v_plus = v0 * (1.0 + dv_frac)

    theta_minus = theta_at_volume(theta0, gamma_true, v_minus, v0)
    theta_plus = theta_at_volume(theta0, gamma_true, v_plus, v0)

    gamma_fd = estimate_mode_gamma_fd(theta_minus, theta0, theta_plus, v_minus, v0, v_plus)
    rel_err = np.abs(gamma_fd - gamma_true) / gamma_true

    df = pd.DataFrame(
        {
            "mode_id": np.arange(1, theta0.size + 1, dtype=int),
            "theta0_K": theta0,
            "gamma_true": gamma_true,
            "gamma_fd": gamma_fd,
            "rel_err": rel_err,
        }
    )
    return df, gamma_fd


def compute_bulk_properties(
    temperatures: np.ndarray,
    theta0: np.ndarray,
    gamma_mode: np.ndarray,
    degeneracy: np.ndarray,
    bulk_modulus_pa: float,
    volume_m3_per_mol: float,
) -> pd.DataFrame:
    """Compute C_V(T), weighted gamma(T), alpha(T), and inverse-recovered gamma(T)."""
    cv_modes = np.vstack(
        [
            einstein_mode_cv_molar(temperatures, theta_mode=t, degeneracy=g)
            for t, g in zip(theta0, degeneracy)
        ]
    )

    cv_total = np.sum(cv_modes, axis=0)
    gamma_weighted = np.sum(gamma_mode[:, None] * cv_modes, axis=0) / cv_total

    # Thermodynamic identity: gamma = alpha * K_T * V / C_V
    alpha = gamma_weighted * cv_total / (bulk_modulus_pa * volume_m3_per_mol)
    gamma_recovered = alpha * bulk_modulus_pa * volume_m3_per_mol / cv_total

    gamma_recovery_rel_err = np.abs(gamma_recovered - gamma_weighted) / gamma_weighted

    return pd.DataFrame(
        {
            "T_K": temperatures,
            "Cv_J_per_molK": cv_total,
            "gamma_weighted": gamma_weighted,
            "alpha_per_K": alpha,
            "gamma_recovered": gamma_recovered,
            "gamma_recovery_rel_err": gamma_recovery_rel_err,
        }
    )


def run_sanity_checks(
    cfg: GruneisenConfig,
    mode_table: pd.DataFrame,
    temp_table: pd.DataFrame,
    gamma_true: np.ndarray,
    degeneracy: np.ndarray,
) -> dict[str, float]:
    max_mode_rel_err = float(mode_table["rel_err"].max())

    gamma_expected_high_t = float(np.sum(gamma_true * degeneracy) / np.sum(degeneracy))
    gamma_high_t = float(temp_table["gamma_weighted"].iloc[-1])
    gamma_high_t_rel_err = abs(gamma_high_t - gamma_expected_high_t) / gamma_expected_high_t

    cv_high_t = float(temp_table["Cv_J_per_molK"].iloc[-1])
    cv_high_t_target = float(np.sum(degeneracy) * R_GAS)
    cv_high_t_rel_err = abs(cv_high_t - cv_high_t_target) / cv_high_t_target

    max_gamma_recovery_rel_err = float(temp_table["gamma_recovery_rel_err"].max())

    min_cv = float(temp_table["Cv_J_per_molK"].min())
    min_alpha = float(temp_table["alpha_per_K"].min())

    assert max_mode_rel_err < 8.0e-4, f"mode gamma finite-difference error too large: {max_mode_rel_err:.3e}"
    assert gamma_high_t_rel_err < 3.0e-2, f"high-T gamma limit mismatch: {gamma_high_t_rel_err:.3e}"
    assert cv_high_t_rel_err < 2.0e-2, f"high-T Cv limit mismatch: {cv_high_t_rel_err:.3e}"
    assert max_gamma_recovery_rel_err < 1.0e-12, (
        "thermodynamic inversion mismatch too large: "
        f"{max_gamma_recovery_rel_err:.3e}"
    )
    assert min_cv > 0.0, "C_V must be positive"
    assert min_alpha > 0.0, "alpha must be positive for positive gamma"

    return {
        "max_mode_rel_err": max_mode_rel_err,
        "gamma_high_t_rel_err": float(gamma_high_t_rel_err),
        "cv_high_t_rel_err": float(cv_high_t_rel_err),
        "max_gamma_recovery_rel_err": max_gamma_recovery_rel_err,
        "min_cv_J_per_molK": min_cv,
        "min_alpha_per_K": min_alpha,
        "gamma_high_t": gamma_high_t,
        "gamma_high_t_expected": gamma_expected_high_t,
    }


def main() -> None:
    cfg = GruneisenConfig()
    validate_config(cfg)

    theta0, gamma_true, degeneracy = unpack_mode_arrays(cfg)

    mode_table, gamma_fd = build_mode_summary_table(cfg, theta0=theta0, gamma_true=gamma_true)

    temperatures = np.geomspace(cfg.t_min, cfg.t_max, cfg.n_temps)
    temp_table = compute_bulk_properties(
        temperatures=temperatures,
        theta0=theta0,
        gamma_mode=gamma_fd,
        degeneracy=degeneracy,
        bulk_modulus_pa=cfg.bulk_modulus_pa,
        volume_m3_per_mol=cfg.v0_m3_per_mol,
    )

    checks = run_sanity_checks(cfg, mode_table, temp_table, gamma_true, degeneracy)

    print("=== Gruneisen Parameter MVP ===")
    print(
        {
            "V0_m3_per_mol": cfg.v0_m3_per_mol,
            "bulk_modulus_GPa": cfg.bulk_modulus_pa / 1.0e9,
            "finite_diff_fraction": cfg.finite_diff_fraction,
            "n_modes": int(theta0.size),
            "n_temps": cfg.n_temps,
            "temperature_range_K": (cfg.t_min, cfg.t_max),
        }
    )

    print("\n[mode_gamma_estimation]")
    with pd.option_context("display.width", 140, "display.max_rows", 20):
        print(mode_table.to_string(index=False, float_format=lambda x: f"{x:12.6f}"))

    print("\n[temperature_samples]")
    sample_idx = np.linspace(0, len(temp_table) - 1, 8, dtype=int)
    sample_df = temp_table.iloc[sample_idx].copy()
    sample_df["alpha_1e5_per_K"] = sample_df["alpha_per_K"] * 1.0e5
    show_cols = [
        "T_K",
        "Cv_J_per_molK",
        "gamma_weighted",
        "alpha_1e5_per_K",
        "gamma_recovered",
    ]
    with pd.option_context("display.width", 160, "display.max_rows", 20):
        print(sample_df[show_cols].to_string(index=False, float_format=lambda x: f"{x:12.6f}"))

    print("\n[sanity_checks]")
    for key, value in checks.items():
        print(f"- {key}: {value:.6e}")


if __name__ == "__main__":
    main()

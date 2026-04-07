"""Minimal runnable MVP for Doppler Effect for Light.

This script demonstrates the relativistic Doppler model for light by:
1) Forward mapping between radial velocity beta=v/c and observed wavelength/frequency.
2) Inverse mapping from measured redshift z back to beta.
3) Quantifying when classical approximation z≈beta is acceptable.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


C_LIGHT = 299_792_458.0  # m/s


def _as_beta_array(beta: np.ndarray | float) -> np.ndarray:
    """Return beta as float64 numpy array and validate |beta| < 1."""
    beta_arr = np.asarray(beta, dtype=np.float64)
    if np.any(np.abs(beta_arr) >= 1.0):
        raise ValueError("beta must satisfy |beta| < 1 for physical light-speed limits")
    return beta_arr


def doppler_factor_from_beta(beta: np.ndarray | float) -> np.ndarray:
    """Return Doppler factor D = sqrt((1+beta)/(1-beta)), beta>0 means receding source."""
    beta_arr = _as_beta_array(beta)
    return np.sqrt((1.0 + beta_arr) / (1.0 - beta_arr))


def observed_wavelength(rest_wavelength: float, beta: np.ndarray | float) -> np.ndarray:
    """Return observed wavelength lambda_obs for given rest wavelength lambda_0."""
    return float(rest_wavelength) * doppler_factor_from_beta(beta)


def observed_frequency(rest_frequency: float, beta: np.ndarray | float) -> np.ndarray:
    """Return observed frequency f_obs for given rest frequency f_0."""
    return float(rest_frequency) / doppler_factor_from_beta(beta)


def redshift_from_beta(beta: np.ndarray | float) -> np.ndarray:
    """Return relativistic redshift z where 1+z = sqrt((1+beta)/(1-beta))."""
    return doppler_factor_from_beta(beta) - 1.0


def beta_from_redshift(redshift: np.ndarray | float) -> np.ndarray:
    """Return radial beta from redshift using beta=((1+z)^2-1)/((1+z)^2+1)."""
    z = np.asarray(redshift, dtype=np.float64)
    k = 1.0 + z
    if np.any(k <= 0.0):
        raise ValueError("redshift z must satisfy 1+z > 0")
    return (k * k - 1.0) / (k * k + 1.0)


def classical_redshift_approx(beta: np.ndarray | float) -> np.ndarray:
    """Return non-relativistic approximation z≈beta for |beta|<<1."""
    return _as_beta_array(beta)


def run_inverse_consistency_demo() -> Dict[str, float]:
    """Validate forward/inverse mappings over a wide velocity range."""
    beta_grid = np.linspace(-0.85, 0.85, 4001, dtype=np.float64)
    z_grid = redshift_from_beta(beta_grid)
    beta_recovered = beta_from_redshift(z_grid)

    max_abs_err = float(np.max(np.abs(beta_recovered - beta_grid)))
    mean_abs_err = float(np.mean(np.abs(beta_recovered - beta_grid)))

    assert max_abs_err < 1e-12, f"beta inversion error too large: {max_abs_err:.3e}"

    return {
        "beta_min": float(beta_grid.min()),
        "beta_max": float(beta_grid.max()),
        "max_abs_beta_err": max_abs_err,
        "mean_abs_beta_err": mean_abs_err,
    }


def run_classical_vs_relativistic_demo() -> Dict[str, float]:
    """Quantify low-speed validity and high-speed breakdown of classical approximation."""
    beta_small = np.linspace(-0.01, 0.01, 2001, dtype=np.float64)
    z_rel_small = redshift_from_beta(beta_small)
    z_cls_small = classical_redshift_approx(beta_small)
    small_speed_max_abs_err = float(np.max(np.abs(z_rel_small - z_cls_small)))

    beta_high = np.array([0.1, 0.3, 0.6], dtype=np.float64)
    z_rel_high = redshift_from_beta(beta_high)
    z_cls_high = classical_redshift_approx(beta_high)
    high_speed_abs_gaps = np.abs(z_rel_high - z_cls_high)

    gap_at_0p3 = float(high_speed_abs_gaps[1])
    gap_at_0p6 = float(high_speed_abs_gaps[2])

    assert small_speed_max_abs_err < 5.2e-5, "low-speed approximation error unexpectedly large"
    assert gap_at_0p3 > 0.05, "expected relativistic correction at beta=0.3 is not visible"

    return {
        "small_speed_max_abs_z_error": small_speed_max_abs_err,
        "high_speed_gap_beta_0.3": gap_at_0p3,
        "high_speed_gap_beta_0.6": gap_at_0p6,
    }


def run_spectral_line_velocity_estimation_demo() -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Estimate radial velocity from noisy wavelength observations of one spectral line."""
    rng = np.random.default_rng(7)

    rest_wavelength_nm = 656.281  # H-alpha line
    rest_frequency_hz = C_LIGHT / (rest_wavelength_nm * 1e-9)

    true_beta = np.array([-0.05, -0.015, 0.0, 0.03, 0.12, 0.35], dtype=np.float64)
    lambda_true_nm = observed_wavelength(rest_wavelength_nm, true_beta)

    measurement_noise_nm = rng.normal(loc=0.0, scale=0.002, size=true_beta.shape)
    lambda_measured_nm = lambda_true_nm + measurement_noise_nm

    z_measured = lambda_measured_nm / rest_wavelength_nm - 1.0
    beta_est = beta_from_redshift(z_measured)

    abs_beta_err = np.abs(beta_est - true_beta)
    beta_mae = float(mean_absolute_error(true_beta, beta_est))
    beta_max_err = float(np.max(abs_beta_err))

    freq_true_hz = observed_frequency(rest_frequency_hz, true_beta)
    light_speed_reconstruction = freq_true_hz * lambda_true_nm * 1e-9
    c_identity_max_abs_err = float(np.max(np.abs(light_speed_reconstruction - C_LIGHT)))

    table = pd.DataFrame(
        {
            "beta_true": true_beta,
            "lambda_true_nm": lambda_true_nm,
            "lambda_measured_nm": lambda_measured_nm,
            "z_measured": z_measured,
            "beta_est": beta_est,
            "abs_beta_error": abs_beta_err,
        }
    )

    assert beta_mae < 2.0e-5, f"velocity estimation MAE too large: {beta_mae:.3e}"
    assert beta_max_err < 5.0e-5, f"velocity estimation max error too large: {beta_max_err:.3e}"
    assert c_identity_max_abs_err < 1e-6, "lambda*f=c identity violated beyond numerical tolerance"

    metrics = {
        "rest_wavelength_nm": rest_wavelength_nm,
        "noise_std_nm": 0.002,
        "beta_mae": beta_mae,
        "beta_max_abs_err": beta_max_err,
        "lambda_f_c_identity_max_abs_err": c_identity_max_abs_err,
    }
    return table, metrics


def main() -> None:
    print("=== Demo A: Forward/Inverse consistency (relativistic Doppler) ===")
    report_a = run_inverse_consistency_demo()
    for key, value in report_a.items():
        print(f"{key:>30s}: {value:.10f}")

    print("\n=== Demo B: Classical vs relativistic redshift ===")
    report_b = run_classical_vs_relativistic_demo()
    for key, value in report_b.items():
        print(f"{key:>30s}: {value:.10f}")

    print("\n=== Demo C: Spectral-line velocity estimation ===")
    table_c, report_c = run_spectral_line_velocity_estimation_demo()
    print(table_c.to_string(index=False, justify="right", float_format=lambda x: f"{x:.8f}"))
    print("\nMetrics:")
    for key, value in report_c.items():
        print(f"{key:>30s}: {value:.10f}")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()

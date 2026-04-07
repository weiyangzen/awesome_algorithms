"""Minimal runnable MVP for Relativistic Doppler Effect (longitudinal case).

Sign convention:
- beta > 0: source receding from observer (redshift).
- beta < 0: source approaching observer (blueshift).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

C_LIGHT = 299_792_458.0  # m/s


def validate_beta_array(beta: np.ndarray) -> np.ndarray:
    """Validate and return beta as float64 1D array with |beta|<1."""
    arr = np.asarray(beta, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("beta must be a 1D array.")
    if arr.size == 0:
        raise ValueError("beta array must be non-empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("beta must contain only finite values.")
    if np.any(np.abs(arr) >= 1.0):
        raise ValueError("all |beta| values must be < 1.")
    return arr


def doppler_frequency_ratio(beta: np.ndarray) -> np.ndarray:
    """Observed/emitted frequency ratio for longitudinal relativistic Doppler."""
    beta = validate_beta_array(beta)
    return np.sqrt((1.0 - beta) / (1.0 + beta))


def doppler_wavelength_ratio(beta: np.ndarray) -> np.ndarray:
    """Observed/emitted wavelength ratio for longitudinal relativistic Doppler."""
    beta = validate_beta_array(beta)
    return np.sqrt((1.0 + beta) / (1.0 - beta))


def relativistic_redshift(beta: np.ndarray) -> np.ndarray:
    """Return z = lambda_obs/lambda_emit - 1 for longitudinal SR Doppler."""
    return doppler_wavelength_ratio(beta) - 1.0


def beta_from_redshift(z: np.ndarray) -> np.ndarray:
    """Invert longitudinal SR Doppler relation to recover beta from redshift z."""
    z_arr = np.asarray(z, dtype=np.float64)
    if z_arr.ndim != 1:
        raise ValueError("z must be a 1D array.")
    if z_arr.size == 0:
        raise ValueError("z array must be non-empty.")
    if not np.all(np.isfinite(z_arr)):
        raise ValueError("z must contain only finite values.")
    if np.any(z_arr <= -1.0):
        raise ValueError("z must be > -1 for physical wavelength ratios.")

    ratio_sq = (1.0 + z_arr) ** 2
    return (ratio_sq - 1.0) / (ratio_sq + 1.0)


def classical_redshift_approx(beta: np.ndarray) -> np.ndarray:
    """Classical low-speed approximation z≈beta."""
    return validate_beta_array(beta).copy()


def build_summary_table(betas: np.ndarray, emitted_wavelength_nm: float) -> pd.DataFrame:
    """Compute and package SR Doppler quantities for display."""
    beta = validate_beta_array(betas)
    if emitted_wavelength_nm <= 0.0:
        raise ValueError("emitted_wavelength_nm must be positive.")

    freq_ratio = doppler_frequency_ratio(beta)
    wavelength_ratio = doppler_wavelength_ratio(beta)
    z_rel = wavelength_ratio - 1.0
    beta_recovered = beta_from_redshift(z_rel)
    z_classical = classical_redshift_approx(beta)

    observed_wavelength_nm = emitted_wavelength_nm * wavelength_ratio
    velocity_km_s = beta * C_LIGHT / 1_000.0

    return pd.DataFrame(
        {
            "beta": beta,
            "velocity_km_s": velocity_km_s,
            "freq_ratio_obs_over_emit": freq_ratio,
            "wavelength_ratio_obs_over_emit": wavelength_ratio,
            "z_relativistic": z_rel,
            "beta_recovered_from_z": beta_recovered,
            "z_classical_approx": z_classical,
            "abs_error_classical": np.abs(z_rel - z_classical),
            "lambda_emit_nm": np.full(beta.shape, emitted_wavelength_nm),
            "lambda_obs_nm": observed_wavelength_nm,
        }
    )


def main() -> None:
    # Includes approaching, rest, and receding source samples.
    betas = np.array([-0.8, -0.4, -0.1, 0.0, 0.1, 0.4, 0.8], dtype=np.float64)
    emitted_wavelength_nm = 656.28  # Hydrogen-alpha line (nm)

    table = build_summary_table(betas=betas, emitted_wavelength_nm=emitted_wavelength_nm)
    print("Relativistic Doppler Effect MVP (longitudinal)")
    print(table.to_string(index=False, float_format=lambda x: f"{x: .6f}"))

    beta = table["beta"].to_numpy()
    z_rel = table["z_relativistic"].to_numpy()
    beta_rec = table["beta_recovered_from_z"].to_numpy()
    freq_ratio = table["freq_ratio_obs_over_emit"].to_numpy()
    wavelength_ratio = table["wavelength_ratio_obs_over_emit"].to_numpy()

    # Non-interactive validation checks.
    assert np.allclose(beta, beta_rec, atol=1e-12, rtol=1e-12), "z->beta inverse mismatch."
    assert np.allclose(freq_ratio * wavelength_ratio, 1.0, atol=1e-12, rtol=1e-12), "f and lambda ratios must be reciprocal."
    assert np.all(z_rel[beta > 0.0] > 0.0), "receding source must have z>0."
    assert np.all(z_rel[beta < 0.0] < 0.0), "approaching source must have z<0."
    assert np.all(np.diff(z_rel) > 0.0), "z should increase monotonically with beta."

    low_speed = np.abs(beta) <= 0.1
    assert np.max(np.abs(z_rel[low_speed] - beta[low_speed])) < 6e-3, "classical approximation should be close at low speed."


if __name__ == "__main__":
    main()

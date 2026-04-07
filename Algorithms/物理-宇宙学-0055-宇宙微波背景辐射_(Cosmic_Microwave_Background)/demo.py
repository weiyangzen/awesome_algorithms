"""Minimal runnable MVP for Cosmic Microwave Background (PHYS-0055).

This script builds an auditable CMB pipeline without black-box cosmology solvers:
1) Synthesize a FIRAS-like blackbody spectrum and recover CMB temperature.
2) Build a toy acoustic angular power spectrum D_ell and sample noisy C_ell.
3) Detect acoustic peak positions and compute angular correlation snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import constants, optimize, signal, special


@dataclass(frozen=True)
class CMBConfig:
    """Configuration for the end-to-end CMB MVP demo."""

    true_temperature_k: float = 2.7255
    calibration_gain: float = 1.0
    frequency_min_ghz: float = 30.0
    frequency_max_ghz: float = 900.0
    frequency_count: int = 220
    relative_noise_std: float = 0.008

    ell_min: int = 2
    ell_max: int = 1200
    cosmic_variance_scale: float = 0.35

    seed_spectrum: int = 55
    seed_cl: int = 550


def planck_radiance_hz(nu_hz: np.ndarray, temperature_k: float) -> np.ndarray:
    """Planck blackbody spectral radiance B_nu(T), SI units.

    B_nu(T) = (2 h nu^3 / c^2) / (exp(h nu / kT) - 1)
    """
    if temperature_k <= 0.0:
        raise ValueError("temperature must be positive")

    x = constants.h * nu_hz / (constants.k * temperature_k)
    x = np.clip(x, 1e-12, 700.0)
    numerator = 2.0 * constants.h * nu_hz**3 / (constants.c**2)
    return numerator / np.expm1(x)


def make_synthetic_spectrum(cfg: CMBConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic FIRAS-like CMB spectrum with multiplicative Gaussian noise."""
    rng = np.random.default_rng(cfg.seed_spectrum)

    nu_ghz = np.linspace(cfg.frequency_min_ghz, cfg.frequency_max_ghz, cfg.frequency_count)
    nu_hz = nu_ghz * 1e9

    clean = cfg.calibration_gain * planck_radiance_hz(nu_hz, cfg.true_temperature_k)
    sigma = cfg.relative_noise_std * np.maximum(clean, 1e-30)
    observed = clean + rng.normal(loc=0.0, scale=sigma)
    return nu_hz, observed, sigma


def fit_temperature_and_gain(
    nu_hz: np.ndarray,
    observed: np.ndarray,
    sigma: np.ndarray,
    t0: float = 2.7,
    g0: float = 1.0,
) -> Tuple[float, float, float]:
    """Recover (T, gain) by weighted nonlinear least squares."""

    def residuals(params: np.ndarray) -> np.ndarray:
        t_k, gain = float(params[0]), float(params[1])
        model = gain * planck_radiance_hz(nu_hz, t_k)
        return (model - observed) / np.clip(sigma, 1e-30, None)

    result = optimize.least_squares(
        residuals,
        x0=np.array([t0, g0], dtype=np.float64),
        bounds=([1.0, 0.8], [5.0, 1.2]),
        max_nfev=400,
    )
    if not result.success:
        raise RuntimeError(f"least_squares failed: {result.message}")

    t_hat = float(result.x[0])
    g_hat = float(result.x[1])
    rms_residual = float(np.sqrt(np.mean(result.fun**2)))
    return t_hat, g_hat, rms_residual


def toy_acoustic_d_ell(ell: np.ndarray) -> np.ndarray:
    """Construct a toy LambdaCDM-like D_ell = ell(ell+1)C_ell/(2pi) in uK^2.

    This is not a precision Boltzmann solution; it is a transparent analytical proxy
    with damping tail + acoustic peaks.
    """
    ell = np.asarray(ell, dtype=np.float64)

    baseline = 780.0 * np.power(np.maximum(ell, 2.0) / 80.0, 0.20)
    damping = np.exp(-np.power(ell / 1400.0, 1.22))

    peaks = (
        4900.0 * np.exp(-0.5 * ((ell - 220.0) / 62.0) ** 2)
        + 2400.0 * np.exp(-0.5 * ((ell - 540.0) / 78.0) ** 2)
        + 1600.0 * np.exp(-0.5 * ((ell - 810.0) / 95.0) ** 2)
    )

    d_ell = (baseline + peaks) * damping
    return np.maximum(d_ell, 1e-9)


def d_ell_to_c_ell(ell: np.ndarray, d_ell: np.ndarray) -> np.ndarray:
    """Convert D_ell to C_ell using C_ell = 2pi D_ell / [ell(ell+1)]."""
    ell = np.asarray(ell, dtype=np.float64)
    denom = ell * (ell + 1.0)
    return 2.0 * np.pi * d_ell / np.clip(denom, 1e-12, None)


def c_ell_to_d_ell(ell: np.ndarray, c_ell: np.ndarray) -> np.ndarray:
    """Convert C_ell to D_ell using D_ell = ell(ell+1)C_ell/(2pi)."""
    ell = np.asarray(ell, dtype=np.float64)
    return ell * (ell + 1.0) * c_ell / (2.0 * np.pi)


def sample_observed_c_ell(cfg: CMBConfig, ell: np.ndarray, c_ell_true: np.ndarray) -> np.ndarray:
    """Sample noisy C_ell using Gaussianized cosmic variance.

    sigma(C_ell) ~= sqrt(2/(2ell+1)) * C_ell
    """
    rng = np.random.default_rng(cfg.seed_cl)

    sigma = np.sqrt(2.0 / (2.0 * ell + 1.0)) * c_ell_true
    sigma *= cfg.cosmic_variance_scale

    noisy = c_ell_true + rng.normal(loc=0.0, scale=sigma)
    return np.maximum(noisy, 1e-12)


def detect_acoustic_peaks(ell: np.ndarray, d_ell_obs: np.ndarray) -> pd.DataFrame:
    """Detect major acoustic peaks from smoothed D_ell curve."""
    # Savitzky-Golay smoothing to suppress random scatter while keeping peak locations.
    smooth = signal.savgol_filter(d_ell_obs, window_length=41, polyorder=3, mode="interp")
    idx, props = signal.find_peaks(smooth, prominence=120.0, distance=90)

    peak_ell = ell[idx]
    peak_amp = smooth[idx]
    peak_prom = props["prominences"]

    frame = pd.DataFrame(
        {
            "ell": peak_ell.astype(int),
            "D_ell_uK2": peak_amp,
            "prominence": peak_prom,
        }
    )
    frame = frame.sort_values("D_ell_uK2", ascending=False).reset_index(drop=True)
    return frame


def angular_correlation(theta_deg: np.ndarray, ell: np.ndarray, c_ell: np.ndarray) -> np.ndarray:
    """Compute C(theta) = sum_l (2l+1)/(4pi) C_l P_l(cos theta)."""
    mu = np.cos(np.deg2rad(theta_deg))
    out = np.zeros_like(mu, dtype=np.float64)

    for l_val, cl_val in zip(ell, c_ell):
        prefactor = (2.0 * l_val + 1.0) / (4.0 * np.pi) * cl_val
        out += prefactor * special.eval_legendre(int(l_val), mu)
    return out


def main() -> None:
    cfg = CMBConfig()

    print("CMB MVP (PHYS-0055)")
    print("=" * 66)

    # Part A: spectrum -> temperature estimation
    nu_hz, intensity_obs, sigma = make_synthetic_spectrum(cfg)
    t_hat, g_hat, rms_residual = fit_temperature_and_gain(
        nu_hz,
        intensity_obs,
        sigma,
        t0=2.6,
        g0=1.02,
    )

    print("[A] Blackbody Spectrum Fit")
    print(f"True T_CMB [K]            : {cfg.true_temperature_k:.6f}")
    print(f"Estimated T_CMB [K]       : {t_hat:.6f}")
    print(f"Estimated gain            : {g_hat:.6f}")
    print(f"Weighted RMS residual     : {rms_residual:.6f}")

    # Part B: toy angular spectrum -> peak extraction
    ell = np.arange(cfg.ell_min, cfg.ell_max + 1, dtype=np.int64)
    d_ell_true = toy_acoustic_d_ell(ell)
    c_ell_true = d_ell_to_c_ell(ell, d_ell_true)
    c_ell_obs = sample_observed_c_ell(cfg, ell, c_ell_true)
    d_ell_obs = c_ell_to_d_ell(ell, c_ell_obs)

    peak_table = detect_acoustic_peaks(ell, d_ell_obs)
    top_peaks = peak_table.head(3).sort_values("ell").reset_index(drop=True)

    print("\n[B] Toy Angular Power Spectrum")
    print(f"Ell range                 : [{cfg.ell_min}, {cfg.ell_max}]")
    print("Top 3 peaks (by amplitude):")
    print(top_peaks.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # First acoustic peak: choose strongest peak in [150, 320]
    first_window = peak_table[(peak_table["ell"] >= 150) & (peak_table["ell"] <= 320)]
    if first_window.empty:
        raise RuntimeError("Failed to detect first acoustic peak in expected ell window.")
    first_peak_ell = int(first_window.iloc[0]["ell"])

    # Part C: angular correlation from observed C_ell
    theta_grid = np.array([0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0], dtype=np.float64)
    c_theta = angular_correlation(theta_grid, ell, c_ell_obs)
    corr_df = pd.DataFrame({"theta_deg": theta_grid, "C_theta_uK2": c_theta})

    print("\n[C] Angular Correlation Snapshot")
    print(corr_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"Detected first peak ell    : {first_peak_ell}")

    # Validation checks (deterministic with fixed seeds)
    checks = {
        "temperature fit error < 0.03 K": abs(t_hat - cfg.true_temperature_k) < 0.03,
        "gain estimate near 1.0": abs(g_hat - cfg.calibration_gain) < 0.03,
        "first peak in [190, 250]": 190 <= first_peak_ell <= 250,
        "C(0 deg) > C(60 deg)": c_theta[0] > c_theta[2],
    }

    print("\nChecks:")
    for name, ok in checks.items():
        print(f"- {name}: {'OK' if ok else 'FAIL'}")

    if not all(checks.values()):
        raise SystemExit("Validation failed.")

    print("=" * 66)
    print("Run completed successfully.")


if __name__ == "__main__":
    main()

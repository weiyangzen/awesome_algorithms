"""Minimal MVP for Baryon Acoustic Oscillations (BAO).

The script builds a transparent BAO pipeline without external cosmology black boxes:
1. Construct a smooth no-wiggle spectrum P_nw(k).
2. Inject damped sinusoidal BAO wiggles to get P_true(k).
3. Add controlled Gaussian noise to obtain mock observed P_obs(k).
4. Estimate the acoustic scale r_s from ratio wiggles P_obs/P_nw - 1.
5. Fourier-transform P_obs(k) -> xi(r) and detect the BAO bump position.

All parameters are fixed in code (no interactive input), so
`uv run python demo.py` is fully reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import simpson
from scipy.signal import find_peaks
from scipy.special import spherical_jn


@dataclass(frozen=True)
class BAOParams:
    """Configuration for a compact BAO demonstration in h-scaled units."""

    k_min: float = 5.0e-3
    k_max: float = 0.6
    n_k: int = 2500

    r_min: float = 40.0
    r_max: float = 180.0
    n_r: int = 400

    n_s: float = 0.96
    k_turn: float = 0.2
    smooth_power_alpha: float = 3.0

    sound_horizon_true: float = 105.0  # Mpc/h (effective BAO standard ruler)
    bao_amplitude: float = 0.08
    sigma_nl: float = 7.0  # Mpc/h; nonlinear damping proxy

    noise_fraction: float = 0.02
    random_seed: int = 42

    rs_search_min: float = 90.0
    rs_search_max: float = 120.0
    rs_search_points: int = 601

    peak_search_min: float = 80.0
    peak_search_max: float = 130.0
    baseline_poly_degree: int = 4


def build_k_grid(params: BAOParams) -> np.ndarray:
    return np.linspace(params.k_min, params.k_max, params.n_k)


def build_r_grid(params: BAOParams) -> np.ndarray:
    return np.linspace(params.r_min, params.r_max, params.n_r)


def smooth_no_wiggle_power(k: np.ndarray, params: BAOParams) -> np.ndarray:
    """Pedagogical smooth baseline spectrum P_nw(k)."""
    return (k**params.n_s) / (1.0 + (k / params.k_turn) ** params.smooth_power_alpha)


def bao_wiggle_factor(k: np.ndarray, r_s: float, amplitude: float, sigma_nl: float) -> np.ndarray:
    """Multiplicative BAO wiggles with Gaussian damping."""
    damping = np.exp(-0.5 * (k * sigma_nl) ** 2)
    return 1.0 + amplitude * np.sin(k * r_s) * damping


def generate_mock_spectrum(params: BAOParams) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build k grid, no-wiggle spectrum, true wiggle spectrum, and noisy observed spectrum."""
    k = build_k_grid(params)
    p_nw = smooth_no_wiggle_power(k, params)
    p_true = p_nw * bao_wiggle_factor(
        k,
        r_s=params.sound_horizon_true,
        amplitude=params.bao_amplitude,
        sigma_nl=params.sigma_nl,
    )

    rng = np.random.default_rng(params.random_seed)
    multiplicative_noise = 1.0 + params.noise_fraction * rng.normal(size=k.size)
    p_obs = p_true * multiplicative_noise
    return k, p_nw, p_true, p_obs


def estimate_sound_horizon_from_ratio(
    k: np.ndarray,
    p_obs: np.ndarray,
    p_nw: np.ndarray,
    params: BAOParams,
) -> tuple[float, float, float, pd.DataFrame]:
    """Fit wiggle ratio y(k)=P_obs/P_nw-1 with damped sine templates across r_s grid.

    For each candidate r_s:
      template t(k) = sin(k*r_s) * exp[-(k*sigma_nl)^2/2]
      best amplitude a = <y,t>/<t,t>
      MSE = mean((y-a*t)^2)
    """
    y = p_obs / p_nw - 1.0
    damping = np.exp(-0.5 * (k * params.sigma_nl) ** 2)

    rs_grid = np.linspace(params.rs_search_min, params.rs_search_max, params.rs_search_points)
    templates = np.sin(np.outer(rs_grid, k)) * damping[None, :]

    s_yt = templates @ y
    s_tt = np.sum(templates * templates, axis=1)
    amp_grid = s_yt / np.clip(s_tt, 1e-14, None)

    # Use algebraic SSE expression to avoid building an extra residual matrix.
    s_yy = float(np.dot(y, y))
    sse_grid = s_yy - (s_yt * s_yt) / np.clip(s_tt, 1e-14, None)
    mse_grid = sse_grid / y.size

    best_idx = int(np.argmin(mse_grid))
    r_s_best = float(rs_grid[best_idx])
    amp_best = float(amp_grid[best_idx])
    mse_best = float(mse_grid[best_idx])

    fit_table = pd.DataFrame(
        {
            "r_s_candidate": rs_grid,
            "amplitude_hat": amp_grid,
            "mse": mse_grid,
        }
    )
    return r_s_best, amp_best, mse_best, fit_table


def power_to_correlation_function(k: np.ndarray, p_k: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Compute xi(r)=1/(2pi^2) * int dk k^2 P(k) j0(kr)."""
    j0 = spherical_jn(0, np.outer(r, k))
    integrand = (k[None, :] ** 2) * p_k[None, :] * j0 / (2.0 * np.pi**2)
    return simpson(integrand, x=k, axis=1)


def remove_broadband_trend(
    r: np.ndarray,
    xi: np.ndarray,
    peak_search_min: float,
    peak_search_max: float,
    degree: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Subtract low-order polynomial baseline outside BAO-peak window."""
    mask_background = (r < peak_search_min) | (r > peak_search_max)
    coeff = np.polyfit(r[mask_background], xi[mask_background], degree)
    baseline = np.polyval(coeff, r)
    residual = xi - baseline
    return baseline, residual


def detect_bao_peak(
    r: np.ndarray,
    residual: np.ndarray,
    peak_search_min: float,
    peak_search_max: float,
) -> tuple[float, float]:
    """Detect BAO bump location from residual xi(r)."""
    mask = (r >= peak_search_min) & (r <= peak_search_max)
    r_win = r[mask]
    y_win = residual[mask]

    peak_indices, props = find_peaks(y_win, prominence=0.0)
    if peak_indices.size == 0:
        idx = int(np.argmax(y_win))
        return float(r_win[idx]), float(y_win[idx])

    # For this pedagogical setup, selecting the tallest in-window local maximum
    # is more robust than selecting by prominence alone.
    _ = props  # keep explicit dependency on find_peaks metadata for auditability.
    idx = int(peak_indices[np.argmax(y_win[peak_indices])])
    return float(r_win[idx]), float(y_win[idx])


def build_sample_table(k: np.ndarray, p_obs: np.ndarray, p_nw: np.ndarray, n_rows: int = 12) -> pd.DataFrame:
    """Small diagnostic table for the observed wiggle ratio."""
    idx = np.linspace(0, k.size - 1, n_rows, dtype=int)
    ratio_minus_one = p_obs / p_nw - 1.0
    return pd.DataFrame(
        {
            "k_h_per_Mpc": k[idx],
            "P_obs": p_obs[idx],
            "P_nw": p_nw[idx],
            "P_obs_over_P_nw_minus_1": ratio_minus_one[idx],
        }
    )


def run_checks(
    params: BAOParams,
    r_s_est: float,
    amp_est: float,
    xi_peak_r: float,
) -> None:
    """Hard checks to ensure the MVP captures BAO scale robustly."""
    rs_err = abs(r_s_est - params.sound_horizon_true)
    assert rs_err < 2.5, f"Recovered r_s is too far from truth: error={rs_err:.3f}"

    # In this setup, correlation-space BAO peak is near ~100-115 Mpc/h.
    assert 95.0 <= xi_peak_r <= 120.0, f"Unexpected xi peak location: {xi_peak_r:.3f}"

    assert amp_est > 0.01, f"Recovered BAO amplitude is too small/non-physical: {amp_est:.4f}"


def run_demo() -> None:
    params = BAOParams()

    k, p_nw, p_true, p_obs = generate_mock_spectrum(params)
    r_s_est, amp_est, mse_best, fit_table = estimate_sound_horizon_from_ratio(k, p_obs, p_nw, params)

    r = build_r_grid(params)
    xi_obs = power_to_correlation_function(k, p_obs, r)
    _, xi_residual = remove_broadband_trend(
        r,
        xi_obs,
        peak_search_min=params.peak_search_min,
        peak_search_max=params.peak_search_max,
        degree=params.baseline_poly_degree,
    )
    xi_peak_r, xi_peak_height = detect_bao_peak(
        r,
        xi_residual,
        peak_search_min=params.peak_search_min,
        peak_search_max=params.peak_search_max,
    )

    run_checks(params, r_s_est, amp_est, xi_peak_r)

    fit_best_row = fit_table.loc[(fit_table["r_s_candidate"] - r_s_est).abs().idxmin()]
    sample_table = build_sample_table(k, p_obs, p_nw)

    print("=== Baryon Acoustic Oscillations (BAO) MVP ===")
    print("Mock spectrum: P(k)=P_nw(k)*[1+A*sin(k*r_s)*exp(-(k*Sigma)^2/2)] + noise")
    print(f"True r_s (input): {params.sound_horizon_true:.3f} Mpc/h")
    print(f"Estimated r_s from ratio fit: {r_s_est:.3f} Mpc/h")
    print(f"Absolute error: {abs(r_s_est - params.sound_horizon_true):.3f} Mpc/h")
    print(f"Estimated wiggle amplitude: {amp_est:.5f}")
    print(f"Best-fit ratio MSE: {mse_best:.6e}")
    print(f"Fit-table row MSE check: {float(fit_best_row['mse']):.6e}")
    print()
    print("Correlation-space BAO bump:")
    print(f"Detected xi(r) peak at r = {xi_peak_r:.3f} Mpc/h")
    print(f"Residual peak height = {xi_peak_height:.6e}")
    print()
    print("Sample points of observed wiggle ratio:")
    with pd.option_context("display.precision", 6, "display.width", 180):
        print(sample_table.to_string(index=False))


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()
